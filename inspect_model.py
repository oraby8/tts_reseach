from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('ekwek/Soprano-80M')
print(model.config)
