from transformers import AutoTokenizer
import json

tokenizer_path = "/home/kerno/oraby/soprano-factory/soprano_tokenizer_extended"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Simulate a dataset item
text = "تجربة"
audio_tokens = [0, 100, 200, 300]
formatted_string = f"[STOP][TEXT]{text}[START]{''.join(list(map(lambda x: f'[{x}]', audio_tokens)))}[STOP]"

print(f"String: {formatted_string}")

tokens = tokenizer.tokenize(formatted_string)
ids = tokenizer.encode(formatted_string)

print(f"Token count: {len(tokens)}")
print(f"ID count: {len(ids)}")
print(f"IDs: {ids}")
print(f"Tokens: {tokens}")

if len(ids) < 2:
    print("WARNING: Too few tokens!")
