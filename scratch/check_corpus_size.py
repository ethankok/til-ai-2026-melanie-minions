import sys
sys.path.extend(['nlp/src', 'training/nlp'])
from sweep_retrieval import load_data
from nlp_manager import NLPManager

instances, doc_contents = load_data()
manager = NLPManager()
manager._init_extractive_qa = lambda: None
manager.device = "cpu"

# Run just the chunking part of load_corpus
manager.documents = []
manager.doc_ids = []
for idx, raw in enumerate(doc_contents):
    from nlp_manager import _parse_doc_payload
    doc_id, text = _parse_doc_payload(raw, idx)
    manager.doc_ids.append(doc_id)
    manager.documents.append(text)

manager.passages = []
for doc in manager.documents:
    for chunk in manager._chunk_document(doc):
        manager.passages.append(chunk)

print(f"Number of documents: {len(manager.documents)}")
print(f"Number of passages: {len(manager.passages)}")
