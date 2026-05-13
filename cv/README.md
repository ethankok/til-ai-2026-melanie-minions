# CV

Your CV challenge is to detect and classify every target object in a JPEG image.

This README mirrors the official Wiki challenge specification. If this file ever conflicts with the Wiki, the Wiki wins: <https://github.com/til-ai/til-26/wiki/Challenge-specifications#cv>

## Target list

`category_id` must be the index in this exact list:

| Category index | Object type |
|---:|---|
| 0 | cargo aircraft |
| 1 | commercial aircraft |
| 2 | drone |
| 3 | fighter jet |
| 4 | fighter plane |
| 5 | helicopter |
| 6 | light aircraft |
| 7 | missile |
| 8 | truck |
| 9 | car |
| 10 | tank |
| 11 | bus |
| 12 | van |
| 13 | cargo ship |
| 14 | yacht |
| 15 | cruise ship |
| 16 | warship |
| 17 | sailboat |

## Track variations

Advanced images contain more noise and smaller targets than Novice images.

## Scoring

Each challenge score blends accuracy and speed: 75% accuracy/reward + 25% speed. Qualifier speed uses `t_max = 30 minutes` for the whole test set.

CV accuracy is mean average precision across IoU thresholds `0.50, 0.55, ..., 0.95` (`mAP@.5:.05:.95`).

## Bounding box format

Output boxes are LTWH:

- `l`: left/top-left x coordinate in pixels
- `t`: top/top-left y coordinate in pixels
- `w`: width in pixels
- `h`: height in pixels

Coordinates are zero-indexed. `[0, 0, w, h]` starts at the image's top-left corner.

YOLO-style outputs must be converted. For example, Ultralytics `xywh` uses center-x/center-y/width/height, so convert to LTWH before returning.

## Input

The input is sent via a POST request to the `/cv` route on port `5002`.

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

```Python
{
    "predictions": [
        [
            {
                "bbox": [l, t, w, h],
                "category_id": category_id
            }
        ]
    ]
}
```

If no objects are detected in a scene, return an empty list for that scene:

```Python
{"predictions": [[]]}
```

The `k`-th prediction list must correspond to the `k`-th input instance. The length of `predictions` must equal the length of `instances`.
