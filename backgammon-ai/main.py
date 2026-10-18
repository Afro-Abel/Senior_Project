from backgammon.game import BackgammonGame
from backgammon.rules import get_legal_moves
from backgammon.display import display_board
from backgammon.state import PLAYER_1


def main():
    game = BackgammonGame()
    input("Each player rolls one die to decide who starts. Press Enter...")
    opening_dice = game.roll_opening_dice()
    print(f"Opening roll: Player 1={opening_dice[0]}, Player 2={opening_dice[1]} "
          "(ties are rerolled).")

    while not game.is_game_over():
        display_board(game.state)

        player = game.state.current_player

        if player == PLAYER_1:
            player_name = "Player 1"
        else:
            player_name = "Player 2"

        if opening_dice is not None:
            dice = opening_dice
            opening_dice = None
            print(f"\n{player_name} starts using the opening dice: {dice}")
        else:
            input(f"{player_name}, press Enter to roll the dice...")
            dice = game.roll_dice()
            print(f"\n{player_name} rolled: {dice}")

        legal_moves = get_legal_moves(
            game.state,
            dice,
            player,
        )

        # Player cannot make a move.
        if not legal_moves:
            print("No legal moves. Turn skipped.")
            input("Press Enter to continue...")
            game.switch_player()
            continue

        print("\nLegal moves:")

        for i, move in enumerate(legal_moves):
            print(f"{i}: {move}")

        while True:
            try:
                choice = int(
                    input(f"\n{player_name}, choose a move: ")
                )

                if 0 <= choice < len(legal_moves):
                    break

                print("Invalid move number.")

            except ValueError:
                print("Please enter a number.")

        chosen_move = legal_moves[choice]

        print(f"\n{player_name} chose: {chosen_move}")

        game.apply_move(chosen_move)

        input("\nPress Enter to end your turn...")

        game.switch_player()

    display_board(game.state)

    winner = game.get_winner()

    if winner == PLAYER_1:
        print("Player 1 wins!")
    else:
        print("Player 2 wins!")


if __name__ == "__main__":
    main()
