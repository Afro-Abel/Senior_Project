"""Shared cutoff evaluation. Weights are initial design choices, not fitted data."""
from math import tanh

from backgammon.state import GameState, NUM_CHECKERS


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
        if state.bar[-player]:
            # Opponent must enter before moving any board checker.
            risk = point <= 6 if player == 1 else point >= 19
        else:
            risk = any(state.board[source - 1] * player < 0
                       for source in range(1, 25)
                       if 1 <= (point - source) * player <= 6)
        threatened += bool(risk)
    return made, longest, threatened


def evaluate_position(state: GameState, player: int) -> float:
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
    raw += 4 * (state.borne_off[player] - state.borne_off[-player])
    p1 = [p for p, c in enumerate(state.board, 1) if c > 0]
    p2 = [p for p, c in enumerate(state.board, 1) if c < 0]
    contact = any(state.bar.values()) or (p1 and p2 and max(p1) > min(p2))
    if contact:
        own_home, own_prime, own_blots = contact_features(state, player)
        opp_home, opp_prime, opp_blots = contact_features(state, -player)
        raw += 8 * (state.bar[-player] - state.bar[player])
        raw += 2 * (own_home - opp_home)
        raw += 1.5 * (own_prime ** 2 - opp_prime ** 2)
        raw += 3 * (opp_blots - own_blots)
    return 0.99 * tanh(raw / 50)
