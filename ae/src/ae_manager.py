"""Baseline AE manager.

A tiny rule-based policy that respects the action mask when present. It explores
by moving forward when legal, occasionally rotates, and places bombs sparingly
when legal. Replace this with a proper planner/RL policy later.
"""


class AEManager:
    """Valid rule-based autonomous-exploration baseline."""

    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    PLACE_BOMB = 5

    def __init__(self):
        self.turn_counter = 0

    def _legal(self, observation: dict, action: int) -> bool:
        mask = observation.get("action_mask")
        if mask is None:
            return True
        try:
            return bool(mask[action])
        except Exception:
            return True

    def _first_legal(self, observation: dict, actions: list[int]) -> int:
        for action in actions:
            if self._legal(observation, action):
                return action
        return self.STAY

    def ae(self, observation: dict[str, int | list[int]]) -> int:
        """Choose the next action for the agent.

        Args:
            observation: Environment observation. Important keys include
                action_mask, frozen_ticks, team_bombs, and step.

        Returns:
            Integer action in {0..5}.
        """
        self.turn_counter += 1

        # Do nothing while frozen/dead.
        frozen_ticks = observation.get("frozen_ticks", 0)
        try:
            if int(frozen_ticks) > 0:
                return self.STAY
        except Exception:
            pass

        # Occasionally bomb if legal and available. This can open paths / damage
        # enemies, but doing it every turn wastes bombs and can self-sabotage.
        team_bombs = observation.get("team_bombs", 0)
        try:
            has_bomb = int(team_bombs) > 0
        except Exception:
            has_bomb = False
        if has_bomb and self.turn_counter % 20 == 0 and self._legal(observation, self.PLACE_BOMB):
            return self.PLACE_BOMB

        # Exploration cycle: mostly forward, then rotate right/left to avoid
        # getting stuck at walls. Always respects action_mask if available.
        if self.turn_counter % 7 == 0:
            preferred = [self.RIGHT, self.FORWARD, self.LEFT, self.BACKWARD, self.STAY]
        elif self.turn_counter % 11 == 0:
            preferred = [self.LEFT, self.FORWARD, self.RIGHT, self.BACKWARD, self.STAY]
        else:
            preferred = [self.FORWARD, self.RIGHT, self.LEFT, self.BACKWARD, self.STAY]

        return self._first_legal(observation, preferred)
