from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
save_path = "./model/paraphrase-multilingual-MiniLM-L12-v2"

model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("✅ Modelo exportado a ONNX")