from transformers import AutoTokenizer

tokenizer_path = "/home/kerno/oraby/soprano-factory/soprano_tokenizer_extended"
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
except:
    print("Extended tokenizer not found, checking base.")
    tokenizer = AutoTokenizer.from_pretrained("ekwek/Soprano-80M")

id_0 = tokenizer.convert_tokens_to_ids("[0]")
id_7999 = tokenizer.convert_tokens_to_ids("[7999]")

print(f"ID for [0]: {id_0}")
print(f"ID for [7999]: {id_7999}")

# Check IDs for some text tokens
text_token_id = tokenizer.convert_tokens_to_ids("الله")
print(f"ID for 'الله': {text_token_id}")

print(f"Vocab size: {len(tokenizer)}")
