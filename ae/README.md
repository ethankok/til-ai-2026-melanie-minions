# AE

Your AE challenge is to direct one agent through a 16×16 grid game map while interacting with other agents and completing objectives.

This README mirrors the official Wiki challenge specification. If this file ever conflicts with the Wiki, the Wiki wins: <https://github.com/til-ai/til-26/wiki/Challenge-specifications#ae>

## Gameplay overview

- Each match is played by six teams.
- Each team has one robot in the environment.
- The game advances in discrete time steps.
- Evaluation episodes end after 200 time steps.
- During Qualifiers, ASR/CV/NLP are scored separately from AE. AE can still receive rewards for activating challenges, but no actual ASR/CV/NLP request is sent and their performance does not affect AE score.

## Track variations

- Novice: fixed map.
- Advanced: map varies per match, so agents need to generalize to novel environments.

## Scoring

Each challenge score blends accuracy/reward and speed: 75% reward + 25% speed. Qualifier speed uses `t_max = 30 minutes` for the whole test set.

AE reward score is:

```text
sum(all rewards during evaluation) / number_of_rounds / 1000
```

## Input

The input is sent via a POST request to the `/ae` route on port `5005`. The length of `instances` is always 1.

```JSON
{
  "instances": [
    {
      "observation": {
        "agent_viewcone": [[[0]], [[0]]],
        "base_viewcone": [[[0]], [[0]]],
        "direction": 0,
        "location": [0, 0],
        "base_location": [0, 0],
        "health": [60.0],
        "frozen_ticks": 0,
        "base_health": [100.0],
        "team_resources": [0.0],
        "team_bombs": 0,
        "step": 0,
        "action_mask": [1, 1, 1, 1, 1, 0]
      }
    }
  ]
}
```

Official spec shapes:

- `agent_viewcone`: `7 × 5 × 25` float32 array, oriented relative to the agent's facing direction.
- `base_viewcone`: `5 × 5 × 25` float32 array, centered on the team base.
- `action_mask`: length-6 legal-action mask.

The observation is partially observable. See the dedicated AE environment docs for channel meanings and the action space: <https://github.com/til-ai/til-26/wiki/AE-with-til_environment>

## Reset

During Qualifier evaluation, a GET request is sent to `/reset` when a round ends. Clear any persistent state in your manager when this happens. The server may also treat step-0 observations as a reset signal.

## Output

Your route handler must return:

```Python
{
    "predictions": [
        {
            "action": 0
        }
    ]
}
```

`action` is an integer from the environment action space. Always respect `action_mask`; illegal actions can lose the entire episode/submission even if the planner is otherwise good.
