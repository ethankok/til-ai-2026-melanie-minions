# Noise

Noise is the adversarial image-noising component attached to the CV challenge. During Finals, a team may get the chance to add acceptable perturbations to an opposing team's CV input image before their CV model processes it.

This README mirrors the official Wiki challenge specification. If this file ever conflicts with the Wiki, the Wiki wins: <https://github.com/til-ai/til-26/wiki/Challenge-specifications#noise>

## Track variations

None.

## Scoring

There is an evaluator that checks whether added noise stays within acceptable limits using:

- SSIM
- RMSE L2 norm

Per the official specification, Noise is not directly rewarded in Qualifiers. It can matter indirectly in Finals by making opponents' CV inputs harder while staying within the allowed perturbation limits.

## Input

The input is sent via a POST request to the `/noise` route on port `5003`.

```JSON
{
  "instances": [
    {
      "key": 0,
      "b64": "BASE64_ENCODED_IMAGE"
    }
  ]
}
```

`b64` contains base64-encoded JPEG bytes. The length of `instances` is variable.

## Output

Your route handler must return:

```JSON
{
    "predictions": [
        "BASE64_ENCODED_IMAGE"
    ]
}
```

Each string in `predictions` is the adversarially noised version of the corresponding input image.

The `k`-th prediction must correspond to the `k`-th input instance. The length of `predictions` must equal the length of `instances`.
