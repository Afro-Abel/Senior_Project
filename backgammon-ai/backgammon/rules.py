from backgammon.state import GameState, PLAYER_1
from backgammon.move import Move, Step, BAR, OFF

def direction(player: int) -> int:
    """
    Return the direction in which a player moves.

    Player 1 moves from point 24 toward point 1.
    Player 2 moves from point 1 toward point 24.
    """
    if player == PLAYER_1:
        return -1

    return 1


def owns_point(
    state: GameState,
    point: int,
    player: int
) -> bool:
    """
    Return True if the given player has at least one checker
    on the specified point.
    """
    value = state.board[point - 1]

    return value * player > 0


def is_open(
    state: GameState,
    point: int,
    player: int
) -> bool:
    """
    Return True if the player can land on the specified point.

    A point is open if it:
    - is empty,
    - contains the player's own checkers, or
    - contains exactly one opposing checker.
    """
    value = state.board[point - 1]

    if value == 0:
        return True

    if value * player > 0:
        return True

    return abs(value) == 1


def entry_point(player: int, die: int) -> int:
    """
    Return the point where a checker on the bar would enter
    for the given die value.

    Player 1:
        die 1 -> point 24
        die 6 -> point 19

    Player 2:
        die 1 -> point 1
        die 6 -> point 6
    """
    if player == PLAYER_1:
        return 25 - die

    return die


def apply_step(
    state: GameState,
    step: Step,
    player: int
) -> GameState:
    """
    Apply one checker movement and return the resulting state.

    This function assumes that the step is legal.
    """
    new_state = state.copy()

    # Remove checker from its starting location.
    if step.start == BAR:
        new_state.bar[player] -= 1
    else:
        start_index = step.start - 1
        new_state.board[start_index] -= player

    # Bear the checker off.
    if step.end == OFF:
        new_state.borne_off[player] += 1
        return new_state

    end_index = step.end - 1
    destination = new_state.board[end_index]

    # Hit an opposing blot.
    if destination == -player:
        opponent = -player

        new_state.board[end_index] = 0
        new_state.bar[opponent] += 1

    # Place the player's checker on the destination.
    new_state.board[end_index] += player

    return new_state



def get_legal_steps(
    state: GameState,
    die: int,
    player: int
) -> list[Step]:
    """
    Return all legal single-checker movements for one die.

    This handles:
    - entering from the bar
    - normal movement on the board
    - bearing off
    """
    steps = []

    # If the player has a checker on the bar,
    # they must enter it before moving any other checker.
    if state.bar[player] > 0:
        point = entry_point(player, die)

        if is_open(state, point, player):
            steps.append(Step(BAR, point))

        return steps

    # Check every point on the board.
    for start in range(1, 25):

        if not owns_point(state, start, player):
            continue

        end = start + direction(player) * die

        # Check whether this move bears the checker off.
        if end < 1 or end > 24:
            if can_bear_off_from(state, start, die, player):
                steps.append(Step(start, OFF))

            continue

        if is_open(state, end, player):
            steps.append(Step(start, end))

    return steps


def can_bear_off(state: GameState, player: int) -> bool:
    """
    Return True if all of the player's remaining checkers
    are in their home board.
    """
    if state.bar[player] > 0:
        return False

    if player == PLAYER_1:
        # Player 1's home board is points 1 through 6.
        for point in range(7, 25):
            if owns_point(state, point, player):
                return False
    else:
        # Player 2's home board is points 19 through 24.
        for point in range(1, 19):
            if owns_point(state, point, player):
                return False

    return True

def can_bear_off_from(
    state: GameState,
    point: int,
    die: int,
    player: int
) -> bool:
    """
    Return True if a checker can legally bear off from this point
    using the given die.
    """
    if not can_bear_off(state, player):
        return False

    if player == PLAYER_1:
        distance_from_off = point

        # Exact roll.
        if die == distance_from_off:
            return True

        # Die is too small.
        if die < distance_from_off:
            return False

        # Oversized die.
        # Player 1 moves toward point 1, so higher-numbered points
        # are farther from OFF.
        for farther_point in range(point + 1, 7):
            if owns_point(state, farther_point, player):
                return False

        return True

    else:
        distance_from_off = 25 - point

        # Exact roll.
        if die == distance_from_off:
            return True

        # Die is too small.
        if die < distance_from_off:
            return False

        # Oversized die.
        # Player 2 moves toward point 24, so lower-numbered points
        # are farther from OFF.
        for farther_point in range(19, point):
            if owns_point(state, farther_point, player):
                return False

        return True

def generate_move_sequences(
    state: GameState,
    dice: list[int],
    player: int
) -> list[Move]:
    """
    Generate legal move sequences using the dice in the given order.
    """
    if not dice:
        return [Move([])]

    die = dice[0]

    legal_steps = get_legal_steps(
        state,
        die,
        player
    )

    # This die cannot be played.
    if not legal_steps:
        return [Move([])]

    moves = []

    for step in legal_steps:
        new_state = apply_step(
            state,
            step,
            player
        )

        remaining_moves = generate_move_sequences(
            new_state,
            dice[1:],
            player
        )

        for remaining_move in remaining_moves:
            moves.append(
                Move([step] + remaining_move.steps)
            )

    return moves


def get_legal_moves(
    state: GameState,
    dice: tuple[int, int],
    player: int
) -> list[Move]:
    """
    Return all legal complete moves for a dice roll.

    The player must use as many dice as possible. If only one
    die can be used, the higher die must be used.
    """
    die1, die2 = dice

    # Doubles are played four times.
    if die1 == die2:
        moves = generate_move_sequences(
            state,
            [die1] * 4,
            player
        )

        moves = [
            move
            for move in moves
            if len(move.steps) > 0
        ]

        moves = keep_maximum_length_moves(moves)

        return remove_duplicate_moves(moves)

    first_order = generate_move_sequences(
        state,
        [die1, die2],
        player
    )

    second_order = generate_move_sequences(
        state,
        [die2, die1],
        player
    )

    moves = first_order + second_order

    moves = [
        move
        for move in moves
        if len(move.steps) > 0
    ]

    if not moves:
        return []

    moves = keep_maximum_length_moves(moves)
    moves = remove_duplicate_moves(moves)

    max_length = len(moves[0].steps)

    # If both dice can be used, we are done.
    if max_length == 2:
        return moves

    # If only one die can be used, the higher die must be used
    # whenever possible.
    higher_die = max(die1, die2)

    higher_die_moves = generate_move_sequences(
        state,
        [higher_die],
        player
    )

    higher_die_moves = [
        move
        for move in higher_die_moves
        if len(move.steps) > 0
    ]

    if higher_die_moves:
        return remove_duplicate_moves(higher_die_moves)

    return moves

def keep_maximum_length_moves(moves: list[Move]) -> list[Move]:
    """
    Keep only moves that use the maximum possible number of dice.
    """
    if not moves:
        return []

    max_length = max(len(move.steps) for move in moves)

    return [
        move
        for move in moves
        if len(move.steps) == max_length
    ]


def remove_duplicate_moves(moves: list[Move]) -> list[Move]:
    """
    Remove duplicate moves while preserving their original order.
    """
    unique_moves = []
    seen = set()

    for move in moves:
        key = tuple(move.steps)

        if key not in seen:
            seen.add(key)
            unique_moves.append(move)

    return unique_moves