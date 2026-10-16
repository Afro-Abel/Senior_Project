from backgammon.state import GameState, PLAYER_1


def format_point(value: int) -> str:
    """
    Convert a board value into a short printable representation.
    """
    if value == 0:
        return " . "

    return f"{value:+2d}"


def display_board(state: GameState) -> None:
    """
    Print an ASCII representation of the Backgammon board.
    """

    print()
    print(
        f"BAR: P1={state.bar[1]} P2={state.bar[-1]}"
        f"{' ' * 20}"
        f"OFF: P1={state.borne_off[1]} P2={state.borne_off[-1]}"
    )
    print()

    # Top half: points 13 -> 24
    top_points = list(range(13, 25))

    # Bottom half: points 12 -> 1
    bottom_points = list(range(12, 0, -1))

    print_points(top_points)
    print("-" * 53)
    print_values(state, top_points)

    print(" " * 24 + "BAR")

    print_values(state, bottom_points)
    print("-" * 53)
    print_points(bottom_points)

    print()

    if state.current_player == PLAYER_1:
        print("Current player: Player 1")
    else:
        print("Current player: Player 2")

    print()


def print_points(points: list[int]) -> None:
    """
    Print point numbers with a divider between the two board halves.
    """
    left = points[:6]
    right = points[6:]

    left_text = " ".join(f"{point:>3}" for point in left)
    right_text = " ".join(f"{point:>3}" for point in right)

    print(f"{left_text}  | {right_text}")


def print_values(
    state: GameState,
    points: list[int]
) -> None:
    """
    Print checker counts for the given points.
    """
    left = points[:6]
    right = points[6:]

    left_text = " ".join(
        f"{format_point(state.board[point - 1]):>3}"
        for point in left
    )

    right_text = " ".join(
        f"{format_point(state.board[point - 1]):>3}"
        for point in right
    )

    print(f"{left_text}  | {right_text}")