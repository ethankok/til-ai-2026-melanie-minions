import time
import torch
from transformers import AutoModel, AutoTokenizer

def test_device(device_name):
    print(f"Testing on {device_name}...")
    tok = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5').to(device_name)
    model.eval()
    
    texts = ["This is a test sentence number " + str(i) for i in range(128)]
    
    start = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device_name)
            hidden = model(**enc).last_hidden_state[:, 0]
    elapsed = time.time() - start
    print(f"Device {device_name} elapsed: {elapsed:.2f}s")

def main():
    test_device('cpu')
    if torch.backends.mps.is_available():
        test_device('mps')

if __name__ == "__main__":
    main()
