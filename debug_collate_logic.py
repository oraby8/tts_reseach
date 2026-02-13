from transformers import AutoTokenizer
import torch

tokenizer_path = "/home/kerno/oraby/soprano-factory/soprano_tokenizer_extended"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Simulate collate_pack logic
text = "test"
audio = [i for i in range(100)]
res = f"[STOP][TEXT]{text}[START]{''.join(list(map(lambda x: f'[{x}]', audio)))}[STOP]"
texts = [res] * 64

seq_len = 4096
batch_size = 4

tokens_batch = tokenizer(texts, padding=False, truncation=False)
print(f"Input IDs length: {len(tokens_batch['input_ids'][0])}")

batch = []
cur_sample, cur_size = [], 0
for i in range(len(texts)):
    tokens = torch.tensor(tokens_batch['input_ids'][i][:-1], dtype=torch.long)
    cur_size += tokens.size(0)
    cur_sample.append(tokens)
    if cur_size >= seq_len + 1:
        print("Filled a sequence")
        batch.append(torch.cat(cur_sample)[: seq_len + 1])
        cur_sample, cur_size = [], 0
        if len(batch) == batch_size:
            break

if cur_sample and not batch:
    print("Adding partial sample")
    batch.append(torch.cat(cur_sample + [torch.zeros(seq_len, dtype=torch.long)])[: seq_len + 1])

print(f"Batch len: {len(batch)}")
if len(batch) < batch_size:
    print("Padding batch...")
    if not batch:
        print("Batch is empty!")
    else:
        pad = batch[-1]
        print("Pad ok")
