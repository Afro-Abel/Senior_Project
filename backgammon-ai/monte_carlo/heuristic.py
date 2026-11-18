"""Shared cutoff evaluation. Weights are initial design choices, not fitted data."""
from dataclasses import asdict, dataclass, fields
import json
from math import isfinite, tanh
from pathlib import Path

from backgammon.state import GameState, NUM_CHECKERS


@dataclass(frozen=True)
class HeuristicConfig:
    """Pip-count weight stays fixed at 1 to identify the relative scale."""
    borne_off: float = 4.0
    bar: float = 8.0
    home_points: float = 2.0
    prime_squared: float = 1.5
    threatened_blots: float = 3.0
    scale: float = 50.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"{field.name} must be a finite number")
            if value < 0 or (field.name == "scale" and value == 0):
                raise ValueError(f"{field.name} must be nonnegative; scale must be positive")


DEFAULT_HEURISTIC = HeuristicConfig()


def load_heuristic_config(path: str | Path | None) -> HeuristicConfig:
    """Allow partial JSON overrides; reject misspelled/unknown parameters."""
    if path is None:
        return DEFAULT_HEURISTIC
    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict):
        raise ValueError("Heuristic configuration must be a JSON object")
    unknown = set(values) - set(asdict(DEFAULT_HEURISTIC))
    if unknown:
        raise ValueError(f"Unknown heuristic parameters: {sorted(unknown)}")
    return HeuristicConfig(**values)


def pip_count(state: GameState, player: int) -> int:
    return 25 * state.bar[player] + sum(
        abs(count) * (point if player == 1 else 25 - point)
        for point, count in enumerate(state.board, 1) if count * player > 0
    )


def contact_features(state: GameState, player: int) -> tuple[int, int, int]:
    """Return home points made, longest prime, and directly threatened blots.

    Threats consider direct hits with a single die, including bar entry. This
    is a risk proxy, not an exact probability: combined-dice hits are omitted.
    """
    home = range(1, 7) if player == 1 else range(19, 25)
    made = sum(state.board[p - 1] * player >= 2 for p in home)
    longest = run = threatened = 0
    for point, count in enumerate(state.board, 1):
        run = run + 1 if count * player >= 2 else 0
        longest = max(longest, run)
        if count != player:
            continue
        risk = False
        if state.bar[-player]:
            risk = point <= 6 if player == 1 else point >= 19
        if not state.bar[-player] or state.rule_variant == "relaxed-bar":
            # With relaxed priority, opponent board checkers can attack even
            # while other opponent checkers remain on the bar.
            risk = risk or any(state.board[source - 1] * player < 0
                               for source in range(1, 25)
                               if 1 <= (point - source) * player <= 6)
        threatened += bool(risk)
    return made, longest, threatened


def evaluate_position(state: GameState, player: int,
                      config: HeuristicConfig = DEFAULT_HEURISTIC) -> float:
    """Score from player's perspective: terminal +/-1, otherwise (-.99, .99).

    Race progress dominates. Extra bar, blot, home-board, and prime terms apply
    only while contact remains possible, so irrelevant structures receive no
    bonus in a pure race. The result is a utility, not a win probability.
    """
    if state.borne_off[player] == NUM_CHECKERS:
        return 1.0
    if state.borne_off[-player] == NUM_CHECKERS:
        return -1.0
    raw = pip_count(state, -player) - pip_count(state, player)
    raw += config.borne_off * (state.borne_off[player] - state.borne_off[-player])
    p1 = [p for p, c in enumerate(state.board, 1) if c > 0]
    p2 = [p for p, c in enumerate(state.board, 1) if c < 0]
    contact = any(state.bar.values()) or (p1 and p2 and max(p1) > min(p2))
    if contact:
        own_home, own_prime, own_blots = contact_features(state, player)
        opp_home, opp_prime, opp_blots = contact_features(state, -player)
        raw += config.bar * (state.bar[-player] - state.bar[player])
        raw += config.home_points * (own_home - opp_home)
        raw += config.prime_squared * (own_prime ** 2 - opp_prime ** 2)
        raw += config.threatened_blots * (opp_blots - own_blots)
    return 0.99 * tanh(raw / config.scale)
