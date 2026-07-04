import os
import numpy as np
import pandas as pd
import ast
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import onnxruntime as ort

GENRES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Sci-Fi", 10770: "TV Movie",
    53: "Thriller", 10752: "War", 37: "Western"
}

class OnnxDenseRetriever:
    def __init__(self, csv_path: str, embeddings_path: str, model_dir: str):
        self.df = pd.read_csv(csv_path)
        self.embeddings = np.load(embeddings_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.session = ort.InferenceSession(os.path.join(model_dir, "model.onnx"))

    def _encode(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)
        onnx_inputs = {k: v for k, v in inputs.items() if k in {i.name for i in self.session.get_inputs()}}
        outputs = self.session.run(None, onnx_inputs)
        last_hidden_state = outputs[0]  # (batch, seq, hidden)

        attention_mask = inputs["attention_mask"]
        mask = np.expand_dims(attention_mask, -1).astype(np.float32)
        summed = np.sum(last_hidden_state * mask, axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        emb = summed / counts

        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.clip(norm, a_min=1e-9, a_max=None)
        return emb

    def search(self, query: str, top_k: int = 5, genre_id: int | None = None,
               start_year: int | None = None, end_year: int | None = None) -> list[dict]:
        mask = np.ones(len(self.df), dtype=bool)
        if genre_id:
            mask &= self.df["genre_ids"].astype(str).str.contains(str(genre_id), na=False)
        if start_year:
            mask &= self.df["release_date"].str[:4].astype(int) >= start_year
        if end_year:
            mask &= self.df["release_date"].str[:4].astype(int) <= end_year

        filtered_df = self.df[mask].reset_index(drop=True)
        filtered_embs = self.embeddings[mask]

        query_emb = self._encode(query)
        sims = cosine_similarity(query_emb, filtered_embs)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_idx:
            row = filtered_df.iloc[idx]
            genre_ids = ast.literal_eval(row["genre_ids"]) if isinstance(row["genre_ids"], str) else []
            results.append({
                "id": int(row["id"]),
                "title": row["title"],
                "release_date": row["release_date"],
                "overview": row["overview"],
                "genres": [GENRES.get(g, "Unknown") for g in genre_ids],
                "similarity": float(sims[idx]),
                "poster_url": f"https://image.tmdb.org/t/p/w500{row['poster_path']}" if row.get("poster_path") else None,
            })
        return results