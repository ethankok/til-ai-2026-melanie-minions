import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm, trange

# Set PyTorch to use multiple CPU threads for parallel processing
torch.set_num_threads(8)

# Add NLP src to path
sys.path.extend(['nlp/src'])
from nlp_manager import NLPManager, _bm25_tokenize, _zscore

def load_data():
    paths_to_try = [
        Path("data/novice/nlp"),
        Path("/home/jupyter/novice/nlp"),
        Path("/home/jupyter/advanced/nlp")
    ]
    data_dir = None
    for p in paths_to_try:
        if (p / "nlp.jsonl").exists():
            data_dir = p
            break
            
    if data_dir is None:
        raise FileNotFoundError(f"Could not find nlp.jsonl in any of {paths_to_try}")
        
    nlp_jsonl = data_dir / "nlp.jsonl"
    documents_dir = data_dir / "documents"
    
    with open(nlp_jsonl) as f:
        instances = [json.loads(line.strip()) for line in f if line.strip()]
        
    doc_contents = []
    for doc_file in documents_dir.glob("*.txt"):
        with open(doc_file, "r") as f:
            content = f.read()
            doc = {
                "id": doc_file.stem,
                "document": content,
            }
            doc_contents.append(doc)
    return instances, doc_contents

def get_candidate_passages_pool(manager, question: str, max_k: int = 45):
    q_tokens = _bm25_tokenize(question) or [question.lower()]
    bm25_scores = np.asarray(manager.bm25.get_scores(q_tokens), dtype=np.float32)
    
    q_embed = manager._embed_query(question)
    dense_scores = (manager.passage_embeds @ q_embed).numpy().astype(np.float32)
    
    union_set = set()
    
    # 1. Default hybrid
    passage_hybrid = _zscore(bm25_scores) + _zscore(dense_scores)
    
    doc_idxs = []
    doc_hybrid = np.empty(0, dtype=np.float32)
    if manager.doc_bm25 is not None and manager.doc_embeds is not None and manager.documents:
        doc_bm25_scores = np.asarray(
            manager.doc_bm25.get_scores(q_tokens), dtype=np.float32
        )
        doc_dense_scores = (manager.doc_embeds @ q_embed).numpy().astype(np.float32)
        doc_hybrid = _zscore(doc_bm25_scores) + _zscore(doc_dense_scores)
        doc_idxs = manager._top_indices(doc_hybrid, 8)
        
    hybrid_with_prior = passage_hybrid
    if doc_hybrid.size:
        doc_prior = np.asarray(
            [doc_hybrid[d] for d in manager.passage_doc_idx], dtype=np.float32
        )
        hybrid_with_prior = hybrid_with_prior + 0.65 * _zscore(doc_prior)
        union_set.update(manager._top_indices(hybrid_with_prior, max_k))
        
    union_set.update(manager._top_indices(passage_hybrid, max_k))
    
    # Seed doc best passages
    for didx in doc_idxs[:4]:
        candidates = (
            manager.doc_passage_idxs[didx]
            if didx < len(manager.doc_passage_idxs) else []
        )
        if not candidates:
            continue
        best_pidx = max(candidates, key=lambda pidx: hybrid_with_prior[pidx])
        union_set.add(best_pidx)
        
    return list(union_set), doc_idxs

def custom_retrieve(manager, question: str, k: int, doc_prior_weight: float, bm25_weight: float, dense_weight: float, top_k_doc_retrieve: int, top_k_doc_seed: int):
    q_tokens = _bm25_tokenize(question) or [question.lower()]
    bm25_scores = np.asarray(manager.bm25.get_scores(q_tokens), dtype=np.float32)

    q_embed = manager._embed_query(question)
    dense_scores = (manager.passage_embeds @ q_embed).numpy().astype(np.float32)

    passage_hybrid = bm25_weight * _zscore(bm25_scores) + dense_weight * _zscore(dense_scores)

    doc_idxs: list[int] = []
    doc_hybrid = np.empty(0, dtype=np.float32)
    if manager.doc_bm25 is not None and manager.doc_embeds is not None and manager.documents:
        doc_bm25_scores = np.asarray(
            manager.doc_bm25.get_scores(q_tokens), dtype=np.float32
        )
        doc_dense_scores = (manager.doc_embeds @ q_embed).numpy().astype(np.float32)
        doc_hybrid = bm25_weight * _zscore(doc_bm25_scores) + dense_weight * _zscore(doc_dense_scores)
        doc_idxs = manager._top_indices(doc_hybrid, top_k_doc_retrieve)

    hybrid = passage_hybrid
    if doc_hybrid.size:
        doc_prior = np.asarray(
            [doc_hybrid[d] for d in manager.passage_doc_idx], dtype=np.float32
        )
        hybrid = hybrid + doc_prior_weight * _zscore(doc_prior)

    passage_idxs = manager._top_indices(hybrid, k)

    seen = set(passage_idxs)
    for didx in doc_idxs[:top_k_doc_seed]:
        candidates = (
            manager.doc_passage_idxs[didx]
            if didx < len(manager.doc_passage_idxs) else []
        )
        if not candidates:
            continue
        best_pidx = max(candidates, key=lambda pidx: hybrid[pidx])
        if best_pidx not in seen:
            passage_idxs.append(best_pidx)
            seen.add(best_pidx)

    return passage_idxs, doc_idxs

def main():
    print("Loading data...")
    instances, doc_contents = load_data()
    print(f"Loaded {len(instances)} instances, {len(doc_contents)} documents.")
    
    print("Initializing NLPManager...")
    manager = NLPManager()
    if torch.cuda.is_available():
        manager.device = torch.device("cuda")
        print("Using CUDA device for acceleration.")
        rerank_batch = 256
    else:
        manager.device = torch.device("cpu")
        print("Using CPU device (with multi-threading).")
        rerank_batch = 128
    
    manager._answerer_mode = "extractive"
    manager._init_extractive_qa = lambda: None
    
    manager.load_corpus(doc_contents)
    print("Corpus loaded successfully.")
    
    # Step 1: Pre-collect candidate pools and run batch reranking once to populate cache
    print("Building candidate passage pool and pre-scoring via reranker...")
    
    all_pairs = []
    question_candidates = []
    
    for q_idx, instance in enumerate(tqdm(instances, desc="Constructing candidate pools")):
        q = instance["question"]
        candidates, doc_idxs = get_candidate_passages_pool(manager, q, max_k=45)
        question_candidates.append((candidates, doc_idxs))
        for pidx in candidates:
            all_pairs.append((q_idx, pidx, q, manager.passages[pidx]))
            
    print(f"Total pairs to score: {len(all_pairs)}")
    
    # Score in batches
    rerank_cache = {}
    
    with torch.no_grad():
        for start in trange(0, len(all_pairs), rerank_batch, desc="Reranking candidate passages"):
            chunk = all_pairs[start : start + rerank_batch]
            enc = manager._rerank_tok(
                [c[2] for c in chunk],
                [c[3] for c in chunk],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(manager.device)
            
            logits = manager._rerank_model(**enc).logits.float().cpu()
            if logits.ndim == 1:
                batch_scores = logits.tolist()
            elif logits.shape[-1] == 1:
                batch_scores = logits[:, 0].tolist()
            else:
                batch_scores = (logits[:, -1] - logits[:, 0]).tolist()
                
            for idx, score in enumerate(batch_scores):
                q_idx, pidx = chunk[idx][0], chunk[idx][1]
                rerank_cache[(q_idx, pidx)] = score
                
    print("Cached reranker scores.")
    
    # Step 2: Evaluation function using cache
    def evaluate_with_cache(doc_prior_weight, top_k_retrieve, bm25_weight=1.0, dense_weight=1.0, top_k_doc_retrieve=8, top_k_doc_seed=4):
        hits = 0
        total = len(instances)
        
        for q_idx, instance in enumerate(instances):
            q = instance["question"]
            gt_docs = set(instance["source_docs"])
            
            # Retrieve using custom settings
            retrieved, doc_candidates = custom_retrieve(
                manager, q, top_k_retrieve, doc_prior_weight, bm25_weight, dense_weight, top_k_doc_retrieve, top_k_doc_seed
            )
            
            # Rerank retrieved passages using cached scores
            scores = [rerank_cache.get((q_idx, pidx), -999.0) for pidx in retrieved]
            order = sorted(range(len(retrieved)), key=lambda idx: -scores[idx])
            reranked = [retrieved[idx] for idx in order]
            
            pred_docs = manager._top_doc_ids(
                reranked, fallback=retrieved, doc_fallback=doc_candidates
            )
            
            pred_docs_top3 = set(pred_docs[:3])
            if pred_docs_top3 & gt_docs:
                hits += 1
                
        return hits / total, hits

    print("\nEvaluating default configuration...")
    hr, hits = evaluate_with_cache(0.35, 30, 1.0, 1.0)
    print(f"Default: Hit Rate = {hr:.4f} ({hits}/{len(instances)})")
    
    best_hr = hr
    best_params = (0.35, 30, 1.0, 1.0)
    
    # Define sweep grid
    doc_prior_weights = [0.1, 0.25, 0.35, 0.45, 0.6]
    top_k_retrieves = [30, 40, 50]
    bm25_weights = [0.5, 0.8, 1.0, 1.2, 1.5]
    dense_weights = [0.5, 0.8, 1.0, 1.2, 1.5]
    
    # Run a quick sweep over prior weights and retrieve K
    print("\nSweeping doc_prior_weight and top_k_retrieve...")
    for dpw in doc_prior_weights:
        for tkr in top_k_retrieves:
            hr, hits = evaluate_with_cache(dpw, tkr, 1.0, 1.0)
            print(f"dpw={dpw}, tkr={tkr}: Hit Rate = {hr:.4f} ({hits}/{len(instances)})")
            if hr > best_hr:
                best_hr = hr
                best_params = (dpw, tkr, 1.0, 1.0)
                
    # Sweep bm25/dense weights using the best dpw/tkr found
    dpw, tkr, _, _ = best_params
    print(f"\nSweeping lexical vs dense weights with dpw={dpw}, tkr={tkr}...")
    for bw in bm25_weights:
        for dw in dense_weights:
            if bw == 1.0 and dw == 1.0:
                continue
            hr, hits = evaluate_with_cache(dpw, tkr, bw, dw)
            print(f"bw={bw}, dw={dw}: Hit Rate = {hr:.4f} ({hits}/{len(instances)})")
            if hr > best_hr:
                best_hr = hr
                best_params = (dpw, tkr, bw, dw)
                
    print("\n" + "="*50)
    print("Sweep complete!")
    print(f"Best Hit Rate: {best_hr:.4f}")
    print(f"Best parameters: dpw={best_params[0]}, tkr={best_params[1]}, bw={best_params[2]}, dw={best_params[3]}")
    print("Related to: Default dpw=0.35, tkr=30, bw=1.0, dw=1.0")
    print("="*50)

if __name__ == "__main__":
    main()
