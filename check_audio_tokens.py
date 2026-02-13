from transformers import AutoTokenizer
import re

tokenizer = AutoTokenizer.from_pretrained('ekwek/Soprano-80M')
vocab = tokenizer.get_vocab()

# Find all tokens that look like [number]
pattern = re.compile(r"^\[(\d+)\]$")
audio_tokens = []
for token, id in vocab.items():
    match = pattern.match(token)
    if match:
        audio_tokens.append(int(match.group(1)))

if audio_tokens:
    print(f"Found {len(audio_tokens)} audio tokens.")
    print(f"Min: {min(audio_tokens)}")
    print(f"Max: {max(audio_tokens)}")
else:
    print("No audio tokens found.")
