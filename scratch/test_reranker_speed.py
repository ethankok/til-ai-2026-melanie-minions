import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def test_device(device_name):
    print(f"Testing on {device_name}...")
    tok = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
    model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base').to(device_name)
    model.eval()
    
    pairs = [("What is the capital of France?", "Paris is the capital of France.")] * 128
    
    start = time.time()
    with torch.no_grad():
        enc = tok(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        ).to(device_name)
        logits = model(**enc).logits.float().cpu()
    elapsed = time.time() - start
    print(f"Device {device_name} elapsed: {elapsed:.2f}s")

def main():
    test_device('cpu')
    if torch.backends.mps.is_available():
        test_device('mps')

if __name__ == "__main__":
    main()
