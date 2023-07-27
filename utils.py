import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'roberta-base-go_emotions/'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model = model.to(DEVICE)

def get_predictions(input_text: str) -> dict:
    label2id = model.config.label2id
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True)
    inputs = inputs.to(DEVICE)
    outputs = model(**inputs)
    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    probs = probs.detach().numpy()
    for i, k in enumerate(label2id.keys()):
        label2id[k] = probs[i]
    label2id = {k: float(v) for k, v in sorted(label2id.items(), key=lambda item: item[1].item(), reverse=True)}
    print(label2id)
    return label2id
