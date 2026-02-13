import argparse
import torch
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemo.collections.tts.models import AudioCodecModel

def main():
    parser = argparse.ArgumentParser(description="Run inference on Soprano checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to saved checkpoint directory")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--output-file", type=str, default="generated.wav", help="Output audio filename")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    # 1. Load Model and Tokenizer
    print(f"Loading model from {args.checkpoint_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_dir).to(device)
    model.eval()
    print("Soprano model loaded.")

    # 2. Load Audio Codec
    print("Loading Nvidia Audio Codec...")
    codec = AudioCodecModel.from_pretrained("nvidia/audio-codec-44khz").to(device).eval()
    print("Codec loaded.")

    # 3. Prepare Input
    # Format: [STOP][TEXT]{text}[START]
    # Special tokens: [STOP], [TEXT], [START] are usually in the vocab.
    # Based on previous checks: [STOP] is likely mapped to special token or we use the string.
    # Let's construct the string.
    input_text = f"[STOP][TEXT]{args.prompt}[START]"
    
    print(f"Input text: {input_text}")
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # 4. Generate
    print("Generating tokens...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temp,
            top_k=args.top_k,
            pad_token_id=tokenizer.eos_token_id, # Use EOS as pad if pad not set
            eos_token_id=tokenizer.convert_tokens_to_ids("[STOP]")
        )

    # 5. Extract Audio Tokens
    # The output contains input_ids + generated_ids
    generated_ids = output_ids[0][input_ids.shape[1]:]
    
    print(f"Generated {len(generated_ids)} tokens.")
    
    # Filter for audio tokens and convert to codes
    # Audio tokens range: ID 4 ([0]) to ID 8003 ([7999])
    # Mapping: Code Value = ID - 4
    audio_codes = []
    
    audio_token_min_id = 4
    audio_token_max_id = 8003
    
    for token_id in generated_ids:
        token_id = token_id.item()
        if token_id == tokenizer.convert_tokens_to_ids("[STOP]"):
            break
        
        if audio_token_min_id <= token_id <= audio_token_max_id:
            val = token_id - audio_token_min_id
            # Remove offset (val % 1000) to get original codebook index
            # The codec expects raw indices [0-1023] (actually 0-999 for this model?)
            # Model has 1000 entries per codebook.
            code = val % 1000
            audio_codes.append(code)
            
    if not audio_codes:
        print("No audio tokens generated.")
        return

    print(f"Extracted {len(audio_codes)} audio codes.")

    # 6. Reconstruct Audio
    # We need 8 codebooks per timestep.
    # Shape required by codec decode: [B, 8, T]
    
    num_codebooks = 8
    # Truncate to multiple of 8
    num_timesteps = len(audio_codes) // num_codebooks
    if num_timesteps == 0:
        print("Not enough codes for a single timestep.")
        return
        
    trimmed_codes = audio_codes[:num_timesteps * num_codebooks]
    
    # Convert to tensor
    codes_tensor = torch.tensor(trimmed_codes, device=device).view(num_timesteps, num_codebooks)
    
    # Transpose to [1, 8, T]
    # codes_tensor is [T, 8] -> transpose -> [8, T] -> unsqueeze -> [1, 8, T]
    codes_input = codes_tensor.transpose(0, 1).unsqueeze(0)
    
    # lengths tensor is usually required for NeMo decode
    tokens_len = torch.tensor([codes_input.shape[2]], device=device)
    
    print("Decoding audio...")
    with torch.no_grad():
        # Some NeMo versions expect int32 or int64
        codes_input = codes_input.int() 
        # Check codec.decode signature
        # NeMo AudioCodecModel.decode(tokens, tokens_len=None)
        audio_out, _ = codec.decode(tokens=codes_input, tokens_len=tokens_len)
    
    # 7. Save Audio
    audio_data = audio_out.squeeze().cpu().numpy()
    sf.write(args.output_file, audio_data, 44100)
    print(f"Audio saved to {args.output_file}")

if __name__ == "__main__":
    main()
