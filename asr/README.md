# ASR

Your ASR challenge is to transcribe a noisy recording of one speaker into text.

This README mirrors the official Wiki challenge specification. If this file ever conflicts with the Wiki, the Wiki wins: <https://github.com/til-ai/til-26/wiki/Challenge-specifications#asr>

## Track variations

- Novice: English speech only, with varied accents.
- Advanced: roughly even mix of English, Malay, Tamil, and Chinese. Non-English clips can still contain English fictional slang terms. Advanced audio has more noise than Novice.

The audio is set in the same fictional world as the NLP corpus, so transcripts include in-world slang/proper nouns that may not be normal dictionary words.

## Scoring

Each challenge score blends accuracy and speed: 75% accuracy/reward + 25% speed. Qualifier speed uses `t_max = 30 minutes` for the whole test set.

ASR accuracy:

- English, Tamil, Malay: `max(0, 1 - WER)` using JiWER.
- Chinese: `max(0, 1 - CER)` using JiWER.
- Final ASR accuracy averages the per-language error rates.

Official transcript normalization before scoring:

```python
# English / Tamil / Malay
jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.SubstituteRegexes({"-": " ", "—": " ", "–": " "}),
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

# Chinese
jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.SubstituteRegexes({"-": "", "—": "", "–": ""}),
    jiwer.RemoveWhiteSpace(replace_by_space=False),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfChars(),
])
```

## Input

The input is sent via a POST request to the `/asr` route on port `5001`.

```JSON
{
  "instances": [
    {
      "key": 0,
      "b64": "BASE64_ENCODED_AUDIO"
    }
  ]
}
```

`b64` contains base64-encoded WAV bytes. The length of `instances` is variable.

## Output

Your route handler must return:

```Python
{
    "predictions": [
        "Predicted transcript one.",
        "Predicted transcript two."
    ]
}
```

The `k`-th prediction must correspond to the `k`-th input instance. The length of `predictions` must equal the length of `instances`.
