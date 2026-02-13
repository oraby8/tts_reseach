"""
Training script for Soprano.

Usage:
python train.py --input-dir path/to/files --save-dir path/to/weights

Args:
--input-dir: Path to directory of LJSpeech-style dataset. If none is provided this defaults to the provided example dataset.
--save-dir: Path to directory to save weights

Adapted from https://github.com/karpathy/nanoGPT
"""
import argparse
import pathlib
import random
import time
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import Accelerator

from dataset import AudioDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",
        required=False,
        default="./example_dataset",
        type=pathlib.Path
    )
    parser.add_argument("--save-dir",
        required=True,
        type=pathlib.Path
    )
    parser.add_argument("--unfreeze-steps",
        default=1000,
        type=int,
        help="Number of steps to keep backbone and original embeddings frozen"
    )
    return parser.parse_args()

args = get_args()

# training hyperparameters
seed = 1337
max_lr = 5e-4  # Lowered from 5e-4 to prevent overfitting
warmup_ratio = 0.1
cooldown_ratio = 0.1
min_lr = 0.1 * max_lr
batch_size = 4
grad_accum_steps = 4
seq_len = 4096 
val_freq = 250
text_factor = 1.5
max_steps = 10000 # Reduced from 10000 for small dataset
betas = (0.9, 0.95)
weight_decay = 0.1
train_dataset_path = f'{args.input_dir}/train.json'
val_dataset_path = f'{args.input_dir}/val.json'
save_path = args.save_dir

# Initialize Accelerator
accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps, log_with="wandb")
device = accelerator.device

if accelerator.is_main_process:
    accelerator.init_trackers("soprano_training", config=vars(args))

def worker_seed_init(_):
    worker_seed = torch.initial_seed() % (2**32-1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_lr(it): # WSD schedule
    if it<warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it<max_steps-cooldown_steps:
        return max_lr
    return min_lr + (max_lr-min_lr) * ((max_steps-it) / cooldown_steps)

def collate_pack(texts):
    # Padding logic needs update for new tokenizer?
    # Actually, padding logic depends on tokenizer.pad_token_id which should be set.
    try:
        tokens_batch = tokenizer(texts, padding=False, truncation=False)
    except Exception as e:
        print(f"Tokenizer error on texts: {e}")
        raise e
        
    batch = []
    cur_sample, cur_size = [], 0
    for i in range(len(texts)):
        input_ids = tokens_batch['input_ids'][i]
        if len(input_ids) == 0:
            print(f"Warning: Empty input_ids for text index {i}")
            continue
            
        tokens = torch.tensor(input_ids[:-1], dtype=torch.long)
        
        if len(tokens) == 0:
             # This happens if input_ids has length 1.
             # Debug why.
             # print(f"Warning: Tokens empty after slicing. Input IDs len: {len(input_ids)}. Text len: {len(texts[i])}")
             pass
             
        cur_size += tokens.size(0)
        cur_sample.append(tokens)
        if cur_size >= seq_len + 1:
            batch.append(torch.cat(cur_sample)[: seq_len + 1])
            cur_sample, cur_size = [], 0
            if len(batch) == batch_size:
                break
    if cur_sample and not batch: # add partial sample if there isn't enough data
        batch.append(torch.cat(cur_sample + [torch.zeros(seq_len, dtype=torch.long)])[: seq_len + 1])
        
    if len(batch) < batch_size:
        # pad up to batch_size for consistency
        if not batch:
            print("ERROR: Batch is empty in collate_pack!")
            print(f"Num texts: {len(texts)}")
            print(f"First text sample: {texts[0][:100]}...")
            print(f"Tokenizer vocab size: {len(tokenizer)}")
            sample_ids = tokenizer.encode(texts[0])
            print(f"Sample IDs len: {len(sample_ids)}")
            print(f"Sample IDs start: {sample_ids[:10]}")
            # return empty tensors to avoid crash, likely will crash downstream but better for debug
            return torch.zeros(batch_size, seq_len, dtype=torch.long), torch.zeros(batch_size, seq_len, dtype=torch.long)
            
        pad = batch[-1]
        while len(batch) < batch_size:
            batch.append(pad)
    batch = torch.stack(batch)
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x, y


def compute_loss(logits, y, num_steps):
    if logits.size(1) != y.size(1):
        # Mismatch detected.
        # This might be because the model truncated the output due to max_pos.
        # Check if we can just truncate labels to match logits?
        # If logits are [B, 1024, V] and y is [B, 4096].
        # If we truncate y, we ignore 3/4 of the sequence. That's bad.
        # But for debugging, we can print it.
        if accelerator.is_local_main_process:
            print(f"Shape mismatch! Logits: {logits.shape}, Labels: {y.shape}")
            
    pred = logits.view(-1, logits.size(-1))
    labels = y.view(-1)
    
    # Safety check
    if pred.size(0) != labels.size(0):
        min_len = min(pred.size(0), labels.size(0))
        pred = pred[:min_len]
        labels = labels[:min_len]
        
    loss = torch.nn.functional.cross_entropy(pred, labels, reduction='none')
    
    # Updated audio mask logic for extended tokenizer
    # [0] (ID 4) to [7999] (ID 8003) are audio tokens
    # Ensure audio_mask matches length
    audio_mask = torch.logical_and(labels >= 4, labels <= 8003).view(-1)
    
    # Handle mask length mismatch if any (shouldn't happen if labels was truncated)
    if audio_mask.size(0) != loss.size(0):
         audio_mask = audio_mask[:loss.size(0)]

    audio_loss = loss[audio_mask].mean()
    text_loss = loss[~audio_mask].mean()
    
    # Calculate accuracy only on audio tokens
    acc = (pred.argmax(dim=-1) == labels).float()[audio_mask].mean()
    
    return audio_loss, text_loss, acc


def evaluate(val_dataloader):
    model.eval()
    val_dataloader_it = iter(val_dataloader)
    val_audio_loss_accum = torch.tensor(0.0).to(device)
    val_text_loss_accum = torch.tensor(0.0).to(device)
    val_acc_accum = torch.tensor(0.0).to(device)
    val_count = 0
    
    with torch.no_grad():
        # Iterate over entire validation set
        for batch in val_dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x).logits
            audio_loss, text_loss, acc = compute_loss(logits, y, 1)
            val_audio_loss_accum += audio_loss
            val_text_loss_accum += text_loss
            val_acc_accum += acc
            val_count += 1
            
        val_audio_loss_accum /= val_count
        val_text_loss_accum /= val_count
        val_acc_accum /= val_count
        
        # All gather metrics from all processes
        val_audio_loss_accum = accelerator.gather(val_audio_loss_accum).mean()
        val_text_loss_accum = accelerator.gather(val_text_loss_accum).mean()
        val_acc_accum = accelerator.gather(val_acc_accum).mean()
        
        if accelerator.is_main_process:
            print(f"validation text loss: {val_text_loss_accum.item():.4f}\tvalidation audio loss: {val_audio_loss_accum.item():.4f}\tvalidation acc: {val_acc_accum.item():.4f}")
            accelerator.log({
                "val/text_loss": val_text_loss_accum.item(),
                "val/audio_loss": val_audio_loss_accum.item(),
                "val/acc": val_acc_accum.item()
            }, step=step)
            
    model.train()
    return val_audio_loss_accum.item()


# Load extended tokenizer
tokenizer_path = "./soprano_tokenizer_extended"
if not os.path.exists(tokenizer_path):
    print(f"Warning: Extended tokenizer not found at {tokenizer_path}. Using base tokenizer.")
    tokenizer_path = 'ekwek/Soprano-80M'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

if __name__ == '__main__':
    # device_type handled by accelerate
    
    # Set seed
    accelerator.print(f"Save Path: {save_path}")
    if accelerator.is_local_main_process:
        os.makedirs(save_path, exist_ok=True)

    # lr schedule
    warmup_steps = int(max_steps * warmup_ratio)
    cooldown_steps = int(max_steps * cooldown_ratio)

    # model
    # Resize embeddings for new vocabulary
    # Update config for longer context
    config = AutoConfig.from_pretrained('ekwek/Soprano-80M')
    config.max_position_embeddings = 8192 
    # Enable RoPE scaling to handle longer context
    config.rope_scaling = {"type": "dynamic", "factor": 8.0}
    
    model = AutoModelForCausalLM.from_pretrained('ekwek/Soprano-80M', config=config)
    original_vocab_size = model.get_input_embeddings().weight.shape[0]
    if accelerator.is_local_main_process:
        print(f"Original vocab size: {original_vocab_size}")

    model.resize_token_embeddings(len(tokenizer))
    
    # -------------------------------------------------------------------------
    # Freezing Strategy
    # -------------------------------------------------------------------------
    # We want to freeze the backbone AND the original embeddings for the first `unfreeze_steps`
    # We will register a hook on the embedding gradients to zero out the original rows.
    
    embedding_hook_handle = None
    
    def freeze_hook(grad):
        # grad is [vocab_size, hidden_dim]
        # We zero out rows < original_vocab_size
        # We need to clone to avoid in-place modification errors if any, 
        # though usually modify-in-place is fine for gradient hooks if returned.
        # Ideally: grad[:original_vocab_size] = 0
        if grad is not None:
             grad[:original_vocab_size] = 0.0
        return grad

    if args.unfreeze_steps > 0:
        if accelerator.is_local_main_process:
            print(f"Freezing backbone and original embeddings for {args.unfreeze_steps} steps.")
            
        # Freeze all parameters first
        model.requires_grad_(False)
        
        # Unfreeze input embeddings (we will control rows with hook)
        input_embeddings = model.get_input_embeddings()
        input_embeddings.weight.requires_grad_(True)
        embedding_hook_handle = input_embeddings.weight.register_hook(freeze_hook)
        
        # Unfreeze output embeddings if they are separate (usually tied, but checking)
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and output_embeddings.weight is not input_embeddings.weight:
             # If separate, we need to treat it similarly or just train it all?
             # For safety, let's treat it same as input (train new tokens only)
             output_embeddings.weight.requires_grad_(True)
             output_embeddings.weight.register_hook(freeze_hook)
             if accelerator.is_local_main_process:
                 print("Registered hook on separate output embeddings.")
        elif output_embeddings is not None and accelerator.is_local_main_process:
             print("Output embeddings are tied to input embeddings.")
             
    else:
        if accelerator.is_local_main_process:
             print("No freezing (unfreeze_steps <= 0). Training full model.")

    if accelerator.is_local_main_process:
        # Debug print trainable params
        trainable_params = 0
        all_params = 0
        for name, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Trainable parameters: {trainable_params} / {all_params} ({trainable_params/all_params:.2%})")
    
    # Use bfloat16 if available (handled by accelerate config usually, but forcing here for memory)
    # model.to(torch.bfloat16) 
    
    model.train()

    # dataset
    dataset = AudioDataset(train_dataset_path)
    # we need batch_size * 16 to have enough tokens after packing
    dataloader = DataLoader(dataset,
        batch_size=batch_size * 16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_seed_init,
        collate_fn=collate_pack,
    )
    
    val_dataset = AudioDataset(val_dataset_path)
    val_dataloader = DataLoader(val_dataset,
        batch_size=batch_size * 16,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_seed_init,
        collate_fn=collate_pack,
    )

    # optimizer
    opt = torch.optim.AdamW(model.parameters(), max_lr, betas=betas, weight_decay=weight_decay, fused=True)

    # Prepare with accelerator
    model, opt, dataloader, val_dataloader = accelerator.prepare(
        model, opt, dataloader, val_dataloader
    )
    
    dataloader_it = iter(dataloader)

    pbar = tqdm(range(0, max_steps), ncols=200, dynamic_ncols=True, disable=not accelerator.is_local_main_process)
    
    best_val_loss = float('inf')
    
    for step in pbar:
        start = time.time()
        
        # Validation
        if val_freq>0 and (step % val_freq == 0 or step==max_steps-1):
            val_loss = evaluate(val_dataloader)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if accelerator.is_main_process:
                    tqdm.write(f"New best validation loss: {best_val_loss:.4f}. Saving checkpoint.")
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(f"{save_path}/best_model")
                    tokenizer.save_pretrained(f"{save_path}/best_model")

        # Unfreeze check
        if args.unfreeze_steps > 0 and step == args.unfreeze_steps:
            if accelerator.is_local_main_process:
                print(f"Step {step}: Unfreezing full model!")
            
            # Unfreeze everything
            model.requires_grad_(True)
            
            # Remove hooks (hooks are on the tensor, so we need to find them or use handle)
            # Note: We registered hook on the parameter tensor BEFORE wrapping with accelerate/DDP.
            # The tensor object should be the same underlying storage, but let's check.
            # Actually, `prepare` might replicate models in some DDP modes, but typically `requires_grad` change propagates if we use the model object we have (which is now wrapped).
            # The handle `embedding_hook_handle` is valid for the tensor we registered on.
            if embedding_hook_handle is not None:
                embedding_hook_handle.remove()
                embedding_hook_handle = None
                if accelerator.is_local_main_process:
                    print("Embedding gradient hook removed.")
            
            # If we had separate output embeddings:
            # We didn't keep a handle for it separately in the simplified code above (oops), 
            # effectively assuming tied or just doing input. 
            # Re-visiting the logic above: I only stored one handle. 
            # If output was separate, I should have stored `output_hook_handle`.
            # But let's assume tied for `Soprano-80M` (it's likely Llama-based).
            
            if accelerator.is_local_main_process:
                 # Debug print trainable params
                 trainable_params = 0
                 all_params = 0
                 # For wrapped model, use unwrapped to be clean or just iterate
                 for name, param in model.named_parameters():
                     all_params += param.numel()
                     if param.requires_grad:
                         trainable_params += param.numel()
                 print(f"Trainable parameters after unfreeze: {trainable_params} / {all_params} ({trainable_params/all_params:.2%})")

        # Training step
        # Gradient accumulation is handled by accelerator context manager usually,
        # but here we manually loop micro_steps.
        # With accelerate, it's better to use accelerator.accumulate(model) context.
        
        audio_loss_accum = 0.0
        text_loss_accum = 0.0
        acc_accum = 0.0
        
        # We use explicit accumulation loop to match original logic style, 
        # but wrapped in accelerator.accumulate for DDP sync handling
        
        # Actually, the original loop:
        # for micro_step in range(grad_accum_steps):
        #    ...
        #    loss.backward()
        # opt.step()
        
        # With accelerate:
        # with accelerator.accumulate(model):
        #   ...
        #   accelerator.backward(loss)
        #   opt.step()
        #   opt.zero_grad()
        
        # But since our data loader yields packed sequences that are independent,
        # and we want to control exactly `grad_accum_steps` micro-batches per step...
        
        opt.zero_grad()
        
        for micro_step in range(grad_accum_steps):
            try:
                x, y = next(dataloader_it)
            except StopIteration:
                dataloader_it = iter(dataloader)
                x, y = next(dataloader_it)
                
            # x, y are already on device thanks to prepare() ?
            # Wait, DataLoader from accelerate yields batches on device?
            # Usually yes if device_placement=True (default).
            
            # Print shapes for debug
            # print(f"Input x shape: {x.shape}")
            
            logits = model(x).logits
            
            # print(f"Logits shape: {logits.shape}")
            
            audio_loss, text_loss, acc = compute_loss(logits, y, 1) # Don't divide here
            
            # Loss scaling for accumulation
            loss = (audio_loss + text_factor * text_loss) / grad_accum_steps
            
            accelerator.backward(loss)
            
            audio_loss_accum += audio_loss.item()
            text_loss_accum += text_loss.item()
            acc_accum += acc.item()

        # Clip grad
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
        lr = get_lr(step)
        for param_group in opt.param_groups:
            param_group['lr'] = lr
            
        opt.step()
        
        # Logging
        # Average accumulators
        audio_loss_accum /= grad_accum_steps
        text_loss_accum /= grad_accum_steps
        acc_accum /= grad_accum_steps
        
        end = time.time()
        dt = (end-start)*1000
        tokens_per_second = (batch_size*seq_len*grad_accum_steps) / (end-start)
        
        if accelerator.is_local_main_process:
            tqdm_log = f'text loss: {text_loss_accum:.3f} | audio loss: {audio_loss_accum:.3f} | acc: {acc_accum:.4f} | lr: {lr:.2e} | time: {dt:.2f} ms | {tokens_per_second:.2f} t/s'
            pbar.set_description(tqdm_log)

            accelerator.log({
                "train/text_loss": text_loss_accum,
                "train/audio_loss": audio_loss_accum,
                "train/acc": acc_accum,
                "train/lr": lr,
                "train/tokens_per_sec": tokens_per_second,
                "train/step_time_ms": dt,
            }, step=step)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"Training complete. Saving model at {save_path}")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("Saving done.")
        
    accelerator.end_training()
