# NLP

Your NLP challenge is retrieval-augmented question answering over a test corpus loaded at runtime.

This README mirrors the official Wiki challenge specification. If this file ever conflicts with the Wiki, the Wiki wins: <https://github.com/til-ai/til-26/wiki/Challenge-specifications#nlp>

## Fictional world context

The documents are set in the fictional cyberpunk world of Clairos. This matters because questions and answers can contain in-world proper nouns, slang, organizations, and locations that normal QA models may not know.

## Training data

In the track data directory's `nlp/` folder:

- `documents/` contains the plain-text training corpus.
- `nlp.jsonl` contains question-answer pairs and metadata.
- `models/nlp_eval.zip` contains the answer-equivalence evaluator weights.

## Track variations

Questions are categorized into levels:

| Level | Name | What it tests |
|---|---|---|
| L1 | Direct extraction | Single fact stated directly in one document |
| L2 | Inference / combination | Combine 2-3 facts, simple arithmetic, or identify what the doc does not state |
| L3 | Cross-document | Requires facts from two or more documents; Advanced only |
| L4 | Unanswerable | No corpus reference; expected empty answer and empty documents; Advanced only |
| L5 | False premise | Premise contradicts a referenced document; expected empty answer and non-empty documents; Advanced only |

## Scoring

Each challenge score blends accuracy and speed: 75% accuracy/reward + 25% speed. Qualifier speed uses `t_max = 30 minutes` for the whole test set.

NLP scoring has two parts:

1. Retrieval: the top 3 returned document IDs are checked for overlap with the target document(s). If retrieval fails, the case scores `0`.
2. Answer equivalence: answerable questions are checked by a ModernBERT-base answer-equivalence model on `(question, reference, candidate)` triples. Candidate answers are cleaned of non-printable characters and truncated to 64 tokens. Equivalence probability must exceed `0.9`.

If retrieval succeeds but the answer is not equivalent, the case gets `0.4` for successful retrieval.

Advanced unanswerable handling:

- L4: return `answer: ""` and `documents: []`.
- L5: return `answer: ""` and a non-empty `documents` list.
- Empty/non-empty answer or document lists in the wrong situation score as wrong.

## Corpus setup request

The first request loads the test corpus:

```JSON
{
  "instances": [
    {
      "documents": [
        "Text of document one.",
        "Text of document two."
      ]
    }
  ]
}
```

After embedding/chunking/indexing/etc, return:

```JSON
{
  "predictions": [{"status": "loaded"}]
}
```

That response signals the evaluator that the system is ready for questions.

## Question input

Questions are sent via POST to `/nlp` on port `5004`.

```JSON
{
  "instances": [
    {
      "question": "QUESTION_TEXT"
    }
  ]
}
```

The length of `instances` is variable.

## Output

Your route handler must return:

```Python
{
    "predictions": [
        {"documents": ["DOC-0001"], "answer": "Answer one."},
        {"documents": ["DOC-0002"], "answer": "Answer two."}
    ]
}
```

Each item in `predictions` contains the relevant document IDs and predicted answer for the corresponding question.

The `k`-th prediction must correspond to the `k`-th input instance. The length of `predictions` must equal the length of `instances`.
