"""Optional gnubg-nn reference: maximize single-game win probability at 0 ply.

Uses the external evaluator, not its default gammon-weighted best_move ranking.
Our engine supplies legal actions. No cube, match scoring, or custom rules.
"""
import hashlib
from importlib import import_module, metadata
from pathlib import Path
import platform
import random

from backgammon.rules import apply_step, get_legal_moves
from backgammon.state import GameState


PINNED_VERSION = "1.1.0a11"


def to_gnubg_board(state: GameState, player: int | None = None):
    """Return [opponent, player] rows in each side's own distance-to-off order.

    Slots 0..23 mean distances 1..24 from bearing off; slot 24 is the bar.
    Borne-off checkers are implicit (15 minus on-board/bar). Row 1 is on roll
    for probability evaluation; for move-outcome comparisons keep player fixed.
    """
    player = state.current_player if player is None else player
    if player not in (1, -1) or len(state.board) != 24:
        raise ValueError("Invalid player or board size")
    if any(state.checker_count(p) != 15 or state.bar[p] < 0 or state.borne_off[p] < 0
           for p in (1, -1)):
        raise ValueError("Expected 15 conserved checkers per player")
    rows = []
    for side in (-player, player):
        row = [0] * 25
        for point, count in enumerate(state.board, 1):
            if count * side > 0:
                row[point - 1 if side == 1 else 24 - point] = abs(count)
        row[24] = state.bar[side]
        rows.append(row)
    return rows


def load_engine():
    try:
        engine = import_module("gnubg_nn")
        version = metadata.version("gnubg-nn")
    except ImportError as error:
        raise RuntimeError(
            "Optional benchmark missing. Use .venv-gnubg/bin/python or install "
            "requirements-gnubg.txt in a Python 3.13 virtual environment."
        ) from error
    if version != PINNED_VERSION:
        raise RuntimeError(f"Expected gnubg-nn {PINNED_VERSION}, found {version}")
    return engine


class GNUBGAgent:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng if rng is not None else random.Random()
        self.engine = load_engine()

    def benchmark_metadata(self):
        native = import_module("gnubg_nn._gnubg_nn")
        data_dir = Path(self.engine.__file__).parent / "data"
        return {"package": "gnubg-nn", "version": PINNED_VERSION,
                "python": platform.python_version(), "architecture": platform.machine(),
                "objective": "single-game win probability; no gammon/cube bonus",
                "evaluation_plies": 0, "policy": "rank own legal unique outcomes",
                "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(data_dir.iterdir()) if p.is_file()},
                "native_module_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()}

    def choose_move(self, state: GameState, dice: tuple[int, int]):
        if any(state.borne_off[p] == 15 for p in (1, -1)):
            raise ValueError("Cannot choose moves in a finished game")
        player = state.current_player
        candidates = []
        seen = set()
        for move in get_legal_moves(state, dice, player):
            after = state.copy()
            for step in move.steps:
                after = apply_step(after, step, player)
            board = to_gnubg_board(after, -player)  # Opponent rolls next.
            key = tuple(tuple(row) for row in board)
            if key in seen:
                continue
            seen.add(key)
            if after.borne_off[player] == 15:
                value = 1.0
            else:
                # probabilities()[0] is row 1's inclusive win probability.
                # The remaining entries describe gammons/backgammons, ignored.
                value = 1.0 - self.engine.probabilities(board, 0)[0]
            candidates.append((value, move))
        if not candidates:
            return None
        best = max(value for value, _ in candidates)
        return self.rng.choice([move for value, move in candidates if value == best])
