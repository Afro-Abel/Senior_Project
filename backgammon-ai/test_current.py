from backgammon.state import GameState, PLAYER_1, PLAYER_2
from backgammon.move import Step, Move, BAR, OFF
from backgammon.display import display_board
from backgammon.game import BackgammonGame
from backgammon.rules import (
    direction,
    owns_point,
    is_open,
    entry_point,
    apply_step,
    get_legal_steps,
    can_bear_off,
    can_bear_off_from,
    get_legal_moves,
    remove_duplicate_moves
)
# --------------------------------
# Test initial state
# --------------------------------

state = GameState.initial_state()

print("Player 1 checker count:", state.checker_count(PLAYER_1))
print("Player 2 checker count:", state.checker_count(PLAYER_2))

assert state.checker_count(PLAYER_1) == 15
assert state.checker_count(PLAYER_2) == 15


# --------------------------------
# Test direction
# --------------------------------

assert direction(PLAYER_1) == -1
assert direction(PLAYER_2) == 1


# --------------------------------
# Test owns_point
# --------------------------------

assert owns_point(state, 24, PLAYER_1)
assert owns_point(state, 1, PLAYER_2)

assert not owns_point(state, 24, PLAYER_2)


# --------------------------------
# Test entry point
# --------------------------------

assert entry_point(PLAYER_1, 5) == 20
assert entry_point(PLAYER_2, 5) == 5


# --------------------------------
# Test normal movement
# --------------------------------

new_state = apply_step(
    state,
    Step(13, 8),
    PLAYER_1
)

assert new_state.board[12] == 4
assert new_state.board[7] == 4

# Original state should NOT change.
assert state.board[12] == 5
assert state.board[7] == 3


# --------------------------------
# Test hitting
# --------------------------------

hit_state = GameState()

hit_state.board[12] = 1   # Player 1 on point 13
hit_state.board[7] = -1   # Player 2 blot on point 8

result = apply_step(
    hit_state,
    Step(13, 8),
    PLAYER_1
)

assert result.board[12] == 0
assert result.board[7] == 1
assert result.bar[PLAYER_2] == 1


# --------------------------------
# Test bar entry
# --------------------------------


bar_state = GameState()

bar_state.bar[PLAYER_1] = 1

result = apply_step(
    bar_state,
    Step(BAR, 20),
    PLAYER_1
)

assert result.bar[PLAYER_1] == 0
assert result.board[19] == 1

# --------------------------------
# Test Player 2 bearing off
# --------------------------------

state = GameState()

state.board[21] = -4   # Point 22
state.board[22] = -6   # Point 23
state.board[23] = -2   # Point 24
state.borne_off[PLAYER_2] = 3
state.current_player = PLAYER_2

# Exact roll: point 23 needs a 2.
assert can_bear_off_from(state, 23, 2, PLAYER_2)

# Point 22 needs a 3, so 2 is too small.
assert not can_bear_off_from(state, 22, 2, PLAYER_2)

# 6 is oversized and point 22 is the farthest occupied point.
assert can_bear_off_from(state, 22, 6, PLAYER_2)

# Cannot use oversized dice on point 24 because
# there are checkers farther away on point 22.
assert not can_bear_off_from(state, 24, 2, PLAYER_2)
assert not can_bear_off_from(state, 24, 6, PLAYER_2)

print("Player 2 bearing-off test passed!")
# --------------------------------
# Test bearing off
# --------------------------------

bear_state = GameState()

bear_state.board[2] = 1  # Point 3

result = apply_step(
    bear_state,
    Step(3, OFF),
    PLAYER_1
)

assert result.board[2] == 0
assert result.borne_off[PLAYER_1] == 1


# --------------------------------
# Test legal single steps
# --------------------------------

state = GameState.initial_state()

steps = get_legal_steps(
    state,
    5,
    PLAYER_1
)

print("Legal steps for Player 1 with die 5:")

# for step in steps:
#     print(step)

assert Step(13, 8) in steps
assert Step(8, 3) in steps
assert Step(6, 1) not in steps
assert Step(24, 19) not in steps

assert len(steps) == 2

# --------------------------------
# Test bar priority
# --------------------------------

bar_state = GameState()
bar_state.bar[PLAYER_1] = 1

steps = get_legal_steps(
    bar_state,
    5,
    PLAYER_1
)

assert steps == [Step(BAR, 20)]


# --------------------------------
# Test bearing off legality
# --------------------------------

bear_state = GameState()

bear_state.board[5] = 2   # Point 6
bear_state.board[4] = 3   # Point 5
bear_state.board[2] = 5   # Point 3
bear_state.board[0] = 5   # Point 1

assert can_bear_off(bear_state, PLAYER_1)

steps = get_legal_steps(
    bear_state,
    6,
    PLAYER_1
)

assert Step(6, OFF) in steps


# --------------------------------
# Cannot bear off too early
# --------------------------------

not_ready = GameState()

not_ready.board[7] = 1   # Point 8
not_ready.board[5] = 14  # Point 6

assert not can_bear_off(not_ready, PLAYER_1)


# --------------------------------
# Test higher die bearing off
# --------------------------------

higher_roll = GameState()

higher_roll.board[4] = 5   # Point 5
higher_roll.board[2] = 5   # Point 3
higher_roll.board[0] = 5   # Point 1

steps = get_legal_steps(
    higher_roll,
    6,
    PLAYER_1
)

assert Step(5, OFF) in steps


# --------------------------------
# Test complete move generation
# --------------------------------

state = GameState.initial_state()

moves = get_legal_moves(
    state,
    (5, 3),
    PLAYER_1
)

print("Legal moves for dice (5, 3):")

# for move in moves:
#     print(move)

assert len(moves) > 0

for move in moves:
    assert len(move.steps) <= 2

print("Complete move generation test passed!")

# --------------------------------
# Test duplicate move removal
# --------------------------------

duplicate_moves = [
    Move([
        Step(13, 8),
        Step(8, 5)
    ]),
    Move([
        Step(13, 8),
        Step(8, 5)
    ]),
    Move([
        Step(8, 3),
        Step(6, 3)
    ]),
]

unique_moves = remove_duplicate_moves(duplicate_moves)

assert len(unique_moves) == 2

assert Move([
    Step(13, 8),
    Step(8, 5)
]) in unique_moves

assert Move([
    Step(8, 3),
    Step(6, 3)
]) in unique_moves

print("Duplicate move test passed!")
# --------------------------------
# Test game initialization
# --------------------------------

game = BackgammonGame()

assert game.state.checker_count(PLAYER_1) == 15
assert game.state.checker_count(PLAYER_2) == 15
assert game.state.current_player == PLAYER_1

assert not game.is_game_over()
assert game.get_winner() is None


# --------------------------------
# Test player switching
# --------------------------------

game.switch_player()

assert game.state.current_player == PLAYER_2

game.switch_player()

assert game.state.current_player == PLAYER_1


# --------------------------------
# Test dice
# --------------------------------

for _ in range(100):
    die1, die2 = game.roll_dice()

    assert 1 <= die1 <= 6
    assert 1 <= die2 <= 6


# --------------------------------
# Test applying a complete move
# --------------------------------

game = BackgammonGame()

move = Move([
    Step(13, 8),
    Step(8, 5),
])

game.apply_move(move)

assert game.state.board[12] == 4   # Point 13
assert game.state.board[7] == 3    # Point 8
assert game.state.board[4] == 1    # Point 5


# --------------------------------
# Test winner
# --------------------------------

game = BackgammonGame()

game.state.board = [0] * 24
game.state.borne_off[PLAYER_1] = 15

assert game.is_game_over()
assert game.get_winner() == PLAYER_1


print("Game tests passed!")


# --------------------------------
# Test board display
# --------------------------------

state = GameState.initial_state()

display_board(state)