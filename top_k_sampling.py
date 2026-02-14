import torch
import torch.nn.functional as f
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"generating on : {device}")
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("initialized the tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
print("initialized the model...")

text = "I am a good boy"
max_new_tokens = 200
temperature = 0.8
top_k = 50
input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

# Top_k sampling
model.eval()
with torch.no_grad():
    for _ in range(max_new_tokens):
        # input_ids dim -> (1, 5)
        outputs = model(input_ids=input_ids)
        # output dim -> (1, 5, vocab_size)
        logits = outputs.logits[0, -1, :]
        # logits dim -> (vocab_size)
        logits = logits / temperature

        if top_k is not None:
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float("-inf"))
            logits = logits.scatter_(dim=0, index=top_k_indices, src=top_k_logits)

        probs = f.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat(
            [input_ids, next_token.unsqueeze(0)], dim=1
        )  # dim next_token.unsqueeze(0) -> (1, 1)
        if (next_token == tokenizer.eos_token_id).all():
            break

print(tokenizer.decode(input_ids[0]))
