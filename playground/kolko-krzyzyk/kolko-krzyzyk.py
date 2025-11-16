#!/usr/bin/env python3
import random
from collections import defaultdict
from typing import List, Tuple, Optional

# ----------------- Reprezentacja gry -----------------

# Plansza to krotka 9 pól: ('X', 'O', ' ')
EMPTY = ' '
PLAYER_X = 'X'  # nasz agent
PLAYER_O = 'O'  # przeciwnik (losowy / człowiek)

# Kolory ANSI
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

def colorize_symbol(symbol: str) -> str:
    if symbol == PLAYER_X:
        return f"{RED}{symbol}{RESET}"
    if symbol == PLAYER_O:
        return f"{GREEN}{symbol}{RESET}"
    return symbol


def empty_board() -> Tuple[str, ...]:
    return (EMPTY,) * 9


def available_actions(board: Tuple[str, ...]) -> List[int]:
    return [i for i, v in enumerate(board) if v == EMPTY]


def apply_move(board: Tuple[str, ...], action: int, player: str) -> Tuple[str, ...]:
    if board[action] != EMPTY:
        raise ValueError("Ruch w zajęte pole")
    board_list = list(board)
    board_list[action] = player
    return tuple(board_list)


def check_winner(board: Tuple[str, ...]) -> Optional[str]:
    winning_lines = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ]
    for a, b, c in winning_lines:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    if EMPTY not in board:
        return 'draw'
    return None


def render_board(board: Tuple[str, ...]):
    def sym(i):
        return colorize_symbol(board[i]) if board[i] != EMPTY else str(i + 1)

    row_sep = "---+---+---"
    print(f" {sym(0)} | {sym(1)} | {sym(2)} ")
    print(row_sep)
    print(f" {sym(3)} | {sym(4)} | {sym(5)} ")
    print(row_sep)
    print(f" {sym(6)} | {sym(7)} | {sym(8)} ")
    print()


# ----------------- Q-learning -----------------

class QAgent:
    """
    Q-learning dla Tic-Tac-Toe.
    Q jest słownikiem: (state, action) -> wartość.
    Uczymy się wyłącznie z perspektywy PLAYER_X.
    """

    def __init__(self, alpha: float = 0.1, epsilon: float = 0.2):
        self.Q = defaultdict(float)
        self.alpha = alpha
        self.epsilon = epsilon

    def get_q(self, state: Tuple[str, ...], action: int) -> float:
        return self.Q[(state, action)]

    def choose_action(self, state: Tuple[str, ...], actions: List[int], explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.choice(actions)
        # eksploatacja – wybierz najlepszy znany ruch
        qs = [(self.get_q(state, a), a) for a in actions]
        max_q = max(qs, key=lambda x: x[0])[0]
        best_actions = [a for q, a in qs if q == max_q]
        return random.choice(best_actions)

    def update_from_episode(self, episode: List[Tuple[Tuple[str, ...], int]], reward: float):
        """
        episode: lista (state, action) dla ruchów agenta X.
        reward: końcowa nagroda: +1 wygrana, -1 przegrana, 0 remis.
        Prosty MC: Q <- Q + alpha * (reward - Q)
        """
        for state, action in episode:
            old_q = self.get_q(state, action)
            new_q = old_q + self.alpha * (reward - old_q)
            self.Q[(state, action)] = new_q


def play_random_opponent(agent: QAgent, episodes: int = 50000) -> None:
    """
    Trening przez self-play:
    - Agent = X
    - Przeciwnik = losowy O
    """
    wins = 0
    draws = 0
    losses = 0
    report_every = max(1000, episodes // 10)

    for ep in range(1, episodes + 1):
        board = empty_board()
        current_player = PLAYER_X
        episode_trace: List[Tuple[Tuple[str, ...], int]] = []

        winner = None

        while True:
            if current_player == PLAYER_X:
                actions = available_actions(board)
                action = agent.choose_action(board, actions, explore=True)
                next_board = apply_move(board, action, PLAYER_X)
                episode_trace.append((board, action))
                board = next_board
            else:
                # przeciwnik losowy
                actions = available_actions(board)
                action = random.choice(actions)
                board = apply_move(board, action, PLAYER_O)

            winner = check_winner(board)
            if winner is not None:
                break

            current_player = PLAYER_O if current_player == PLAYER_X else PLAYER_X

        # nagroda z perspektywy agenta X
        if winner == PLAYER_X:
            reward = 1.0
            wins += 1
        elif winner == PLAYER_O:
            reward = -1.0
            losses += 1
        else:
            reward = 0.0
            draws += 1

        agent.update_from_episode(episode_trace, reward)

        if ep % report_every == 0:
            total = wins + draws + losses
            print(
                f"Episod {ep}/{episodes} | "
                f"X wygrane: {wins} ({wins/total:.2%}), "
                f"remisy: {draws} ({draws/total:.2%}), "
                f"przegrane: {losses} ({losses/total:.2%})"
            )


# ----------------- Gra z człowiekiem -----------------

def agent_move(agent: QAgent, board: Tuple[str, ...]) -> Tuple[str, ...]:
    actions = available_actions(board)
    action = agent.choose_action(board, actions, explore=False)
    return apply_move(board, action, PLAYER_X)


def human_move(board: Tuple[str, ...]) -> Tuple[str, ...]:
    actions = available_actions(board)
    while True:
        try:
            move_str = input("Twój ruch (1-9): ").strip()
            idx = int(move_str) - 1
            if idx not in actions:
                print("To pole jest zajęte lub nieprawidłowe, spróbuj jeszcze raz.")
                continue
            return apply_move(board, idx, PLAYER_O)
        except ValueError:
            print("Podaj liczbę 1–9, serio to nie jest trudne.")


def play_vs_human(agent: QAgent):
    """
    Człowiek gra jako O, agent jako X (zaczyna).
    """
    board = empty_board()
    current_player = PLAYER_X
    print("Zagrajmy. Ty jesteś O, ja X. Ja zaczynam.\n")
    render_board(board)

    while True:
        if current_player == PLAYER_X:
            print("Ruch AI (X):")
            board = agent_move(agent, board)
        else:
            print("Twój ruch (O):")
            board = human_move(board)

        render_board(board)
        winner = check_winner(board)
        if winner is not None:
            if winner == PLAYER_X:
                print("Wygrałem. Masz nauczone Q-learningiem kółko i krzyżyk.")
            elif winner == PLAYER_O:
                print("Wygrałeś. Gratulacje, pobiłeś tablicę Q.")
            else:
                print("Remis.")
            break

        current_player = PLAYER_O if current_player == PLAYER_X else PLAYER_X


# ----------------- main -----------------

def main():
    random.seed(42)
    agent = QAgent(alpha=0.2, epsilon=0.2)

    print("Trenuję agenta Tic-Tac-Toe przeciw losowemu przeciwnikowi...")
    play_random_opponent(agent, episodes=30000)

    print("\nTrening zakończony. Teraz możesz zagrać.")
    play_vs_human(agent)


if __name__ == "__main__":
    main()
