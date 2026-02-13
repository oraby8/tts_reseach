from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('ekwek/Soprano-80M')
print(t.backend_tokenizer.to_str())
