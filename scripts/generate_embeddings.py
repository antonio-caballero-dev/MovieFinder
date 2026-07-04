import numpy as np
import pandas as pd
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import torch

model_path = "./model/all-MiniLM-L6-v2"
model = ORTModelForFeatureExtraction.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model_path = "model/all-MiniLM-L6-v2"
csv_path = "data/peliculas_CLEAN.csv"
output_path = "data/embeddings.npy"

df = pd.read_csv(csv_path)
texts = (df["title"] + " " + df["overview"]).tolist()

def encode_batch(texts, batch_size=64):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.numpy())
        print(f"  {min(i+batch_size, len(texts))}/{len(texts)}")
    return np.vstack(all_embeddings)

print("Generando embeddings...")
embeddings = encode_batch(texts)
np.save(output_path, embeddings)
print(f"✅ Guardado: {embeddings.shape}")