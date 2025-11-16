#!/usr/bin/env python3
import numpy as np
import random
import time
from collections import deque
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class MazeEnv:
    """
    Środowisko labiryntu ~35x90 (dokładnie 35x89) z nagrodą kształtowaną:
    - Perfect maze generowany DFS-em (spójny, bez cykli).
    - Potem dorzucane pętle (_add_loops), żeby były alternatywne ścieżki.
    - Start: (1,1)
    - Meta: (n_rows-2, n_cols-2)
    - Traps: kilka losowych pól z dodatkowymi karami.
    - Dodatkowo: bonus za ruch w stronę mety na podstawie dystansu BFS.
    """

    def __init__(
        self,
        cells_h: int = 17,   # 2*17+1 = 35
        cells_w: int = 44,   # 2*44+1 = 89  (~90)
        traps_count: int = 50,
        extra_loops: int = 80,
    ):
        """
        cells_h, cells_w – liczba "komórek" labiryntu.
        Z tego powstaje grid o rozmiarze (2*cells_h+1) x (2*cells_w+1).
        Dla 17x44 => 35x89.
        """

        # generujemy perfect maze
        self.grid = self._generate_perfect_maze(cells_h, cells_w)
        self.n_rows, self.n_cols = self.grid.shape

        # dorzucamy trochę dodatkowych przejść, żeby powstały pętle
        self._add_loops(extra_loops)

        # start/meta na wolnych polach
        self.start = (1, 1)
        self.goal = (self.n_rows - 2, self.n_cols - 2)

        # mapa odległości BFS do mety (tylko po wolnych polach)
        self.dist = self._compute_distances_to_goal()

        # pułapki
        self.traps = self._place_traps(traps_count)

        self.n_actions = 4  # góra, prawo, dół, lewo
        self.n_states = self.n_rows * self.n_cols

        self.agent_pos = None

    # ---------- generowanie labiryntu ----------

    def _generate_perfect_maze(self, cells_h: int, cells_w: int) -> np.ndarray:
        """
        Perfect maze na siatce cells_h x cells_w, mapowany na grid:
        1 = ściana, 0 = przejście.
        """
        grid_h = 2 * cells_h + 1
        grid_w = 2 * cells_w + 1

        grid = np.ones((grid_h, grid_w), dtype=np.int32)  # wszędzie ściany

        visited = [[False for _ in range(cells_w)] for _ in range(cells_h)]

        def cell_to_grid(r: int, c: int) -> Tuple[int, int]:
            return 2 * r + 1, 2 * c + 1

        stack = [(0, 0)]
        visited[0][0] = True
        sr, sc = cell_to_grid(0, 0)
        grid[sr, sc] = 0

        while stack:
            r, c = stack[-1]

            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < cells_h and 0 <= nc < cells_w and not visited[nr][nc]:
                    neighbors.append((nr, nc))

            if neighbors:
                nr, nc = random.choice(neighbors)
                gr, gc = cell_to_grid(r, c)
                ngr, ngc = cell_to_grid(nr, nc)
                wall_r = (gr + ngr) // 2
                wall_c = (gc + ngc) // 2

                grid[wall_r, wall_c] = 0
                grid[ngr, ngc] = 0

                visited[nr][nc] = True
                stack.append((nr, nc))
            else:
                stack.pop()

        return grid

    def _add_loops(self, loops_count: int = 80):
        """
        Dodaje kilka losowych przejść pomiędzy komórkami, rozwalając ściany.
        Dzięki temu powstają cykle, czyli wiele ścieżek do celu.
        Nie ruszamy zewnętrznej ramki.
        """
        if loops_count <= 0:
            return

        created = 0
        attempts = 0
        max_attempts = loops_count * 20  # żeby się nie zapętlić na amen

        # komórki są na współrzędnych nieparzystych
        while created < loops_count and attempts < max_attempts:
            attempts += 1

            # losowa komórka (wsp. nieparzyste, wewnątrz ramki)
            r = random.randrange(1, self.n_rows - 1, 2)
            c = random.randrange(1, self.n_cols - 1, 2)

            # losowy kierunek o 2 pola (do sąsiedniej komórki)
            dr, dc = random.choice([(2, 0), (-2, 0), (0, 2), (0, -2)])
            nr, nc = r + dr, c + dc   # sąsiednia komórka
            wr, wc = r + dr // 2, c + dc // 2  # ściana pomiędzy

            # sprawdzamy granice
            if not (1 <= nr < self.n_rows - 1 and 1 <= nc < self.n_cols - 1):
                continue

            # ściana musi być ścianą, a komórka po drugiej stronie – korytarzem
            if self.grid[wr, wc] == 1 and self.grid[nr, nc] == 0:
                self.grid[wr, wc] = 0
                created += 1

    def _compute_distances_to_goal(self) -> np.ndarray:
        """
        BFS od goal, liczy minimalną liczbę kroków do mety
        dla każdego wolnego pola (0 = meta, INF = nieosiągalne).
        """
        INF = 10**9
        dist = np.full((self.n_rows, self.n_cols), INF, dtype=np.int32)

        gr, gc = self.goal
        if self.grid[gr, gc] == 1:
            raise RuntimeError("Goal jest ścianą, coś poszło bardzo źle.")

        q = deque()
        dist[gr, gc] = 0
        q.append((gr, gc))

        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
                    continue
                if self.grid[nr, nc] == 1:
                    continue
                if dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))

        # sanity check: start musi być osiągalny
        sr, sc = self.start
        if dist[sr, sc] >= INF:
            raise RuntimeError("Start nie ma ścieżki do goal – coś jest nie tak z generacją.")

        return dist

    def _place_traps(self, traps_count: int):
        free_cells = [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if self.grid[r, c] == 0 and (r, c) not in (self.start, self.goal)
        ]
        traps_count = min(traps_count, len(free_cells))
        if traps_count <= 0:
            return set()
        return set(random.sample(free_cells, traps_count))

    # ---------- mapowanie stan <-> pozycja ----------

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        r, c = pos
        return r * self.n_cols + c

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        r = state // self.n_cols
        c = state % self.n_cols
        return r, c

    # ---------- API środowiska ----------

    def reset(self) -> int:
        self.agent_pos = self.start
        return self._pos_to_state(self.agent_pos)

    def step(self, action: int):
        """
        0 = góra, 1 = prawo, 2 = dół, 3 = lewo
        Zwraca: (next_state, reward, done, info)
        """
        r, c = self.agent_pos
        old_d = self.dist[r, c]

        if action == 0:
            nr, nc = r - 1, c
        elif action == 1:
            nr, nc = r, c + 1
        elif action == 2:
            nr, nc = r + 1, c
        elif action == 3:
            nr, nc = r, c - 1
        else:
            raise ValueError(f"Nieznana akcja: {action}")

        # poza planszę -> brak ruchu
        if nr < 0 or nr >= self.n_rows or nc < 0 or nc >= self.n_cols:
            nr, nc = r, c

        # ściana -> brak ruchu
        if self.grid[nr, nc] == 1:
            nr, nc = r, c

        self.agent_pos = (nr, nc)
        new_d = self.dist[nr, nc]

        # bazowa kara za krok
        reward = -0.1
        done = False
        is_trap = False

        # shaping: ruch w stronę mety = bonus, w przeciwną = kara
        shaping_coef = 0.5
        if old_d < 10**8 and new_d < 10**8:
            reward += shaping_coef * float(old_d - new_d)

        # pułapka
        if self.agent_pos in self.traps:
            reward += -2.0
            is_trap = True

        # meta
        if self.agent_pos == self.goal:
            reward += 10.0
            done = True

        next_state = self._pos_to_state(self.agent_pos)
        info = {"trap": is_trap, "dist": int(new_d)}

        return next_state, reward, done, info

    # ---------- wizualizacja: siatka jako tablica ----------

    def render_array(self) -> np.ndarray:
        """
        Zwraca 2D array z kodami:
        0 = ściana
        1 = puste pole
        2 = pułapka
        3 = start
        4 = meta
        5 = agent
        """
        img = np.ones((self.n_rows, self.n_cols), dtype=np.int32)  # domyślnie puste pola

        # ściany
        img[self.grid == 1] = 0

        # pułapki
        for (r, c) in self.traps:
            img[r, c] = 2

        # start / meta / agent nadpisują wcześniejsze
        sr, sc = self.start
        gr, gc = self.goal
        img[sr, sc] = 3
        img[gr, gc] = 4

        if self.agent_pos is not None:
            ar, ac = self.agent_pos
            img[ar, ac] = 5

        return img

    # stara wersja tekstowa – zostawiona na wszelki wypadek, ale nieużywana
    def render(self):
        r_agent, c_agent = self.agent_pos
        for r in range(self.n_rows):
            row_str = []
            for c in range(self.n_cols):
                if (r, c) == (r_agent, c_agent):
                    row_str.append("\x1b[1;32mA\x1b[0m")
                elif (r, c) == self.start:
                    row_str.append("S")
                elif (r, c) == self.goal:
                    row_str.append("G")
                elif self.grid[r, c] == 1:
                    row_str.append("\x1b[1;34mX\x1b[0m")
                elif (r, c) in self.traps:
                    row_str.append("\x1b[1;31m*\x1b[0m")
                else:
                    row_str.append(".")
            print(" ".join(row_str))
        print()


# ---------- wizualizacja polityki (matplotlib) ----------

def visualize_policy(env: MazeEnv, q_table, max_steps: int = 170, sleep: float = 0.01):
    """
    Podgląd aktualnej polityki (greedy) – z użyciem matplotlib zamiast tekstu.
    """
    state = env.reset()

    data = env.render_array()
    cmap = ListedColormap([
        "black",      # 0 = ściana
        "white",      # 1 = puste pole
        "red",        # 2 = pułapka
        "blue",       # 3 = start
        "gold",       # 4 = meta
        "lime",       # 5 = agent
    ])

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Podgląd polityki (greedy)")

    plt.ion()
    plt.show()

    total_reward = 0.0
    for step in range(max_steps):
        action = int(np.argmax(q_table[state]))
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state

        im.set_data(env.render_array())
        ax.set_title(
            f"Krok {step+1}, reward={reward:.2f}, dist={info['dist']}, trap={info['trap']}"
        )
        plt.pause(sleep)

        if done:
            print(f"  -> META w {step + 1} krokach, total_reward={total_reward:.2f}")
            break
    else:
        print(f"  -> Nie dotarł do celu, total_reward={total_reward:.2f}")

    plt.ioff()
    plt.show()
    print("=== Koniec podglądu ===\n")


# ---------- Q-learning ----------

def q_learning_train(env: MazeEnv,
                     episodes: int = 8000,
                     max_steps: int = 300,
                     alpha: float = 0.1,
                     gamma: float = 0.99,
                     epsilon_start: float = 1.0,
                     epsilon_min: float = 0.05,
                     epsilon_decay: float = 0.9995,
                     visualize_every: int = 200):
    """
    Q-learning z shapingiem + opcjonalna wizualizacja co N epizodów.
    Parametry podbite pod większy labirynt.
    """
    q_table = np.zeros((env.n_states, env.n_actions), dtype=np.float32)
    epsilon = epsilon_start

    success_in_window = 0
    window = 200

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            # eksploracja / eksploatacja
            if random.random() < epsilon:
                action = random.randint(0, env.n_actions - 1)
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, done, info = env.step(action)

            best_next_q = np.max(q_table[next_state])
            td_target = reward + gamma * best_next_q
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error

            state = next_state
            total_reward += reward

            if done:
                if reward > 0:
                    success_in_window += 1
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # log
        if (ep + 1) % window == 0:
            print(
                f"Epizod {ep + 1}/{episodes}, "
                f"epsilon={epsilon:.3f}, "
                f"last_total_reward={total_reward:.2f}, "
                f"sukcesy w ostatnich {window}: {success_in_window}"
            )
            success_in_window = 0

        # animacja co N epizodów
        if visualize_every is not None and (ep + 1) % visualize_every == 0:
            visualize_policy(env, q_table, max_steps=170, sleep=0.01)

    return q_table


def run_greedy_policy(env: MazeEnv, q_table, max_steps: int = 170, sleep: float = 0.01):
    """
    Test końcowy polityki greedy – wizualizacja w matplotlib.
    """
    state = env.reset()

    data = env.render_array()
    cmap = ListedColormap([
        "black", "white", "red", "blue", "gold", "lime"
    ])

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Polityka greedy – test końcowy")

    plt.ion()
    plt.show()

    total_reward = 0.0
    for step in range(max_steps):
        action = int(np.argmax(q_table[state]))
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state

        im.set_data(env.render_array())
        ax.set_title(
            f"Krok {step+1}, reward={reward:.2f}, dist={info['dist']}, trap={info['trap']}"
        )
        plt.pause(sleep)

        if done:
            print(f"Dotarł do celu w {step + 1} krokach, total_reward={total_reward:.2f}")
            break
    else:
        print(f"Nie dotarł do celu. total_reward={total_reward:.2f}")

    plt.ioff()
    plt.show()


def main():
    random.seed(42)
    np.random.seed(42)

    # wejście interaktywne: liczba epizodów i częstotliwość demo
    default_episodes = 8000
    default_visualize_every = 500  # 0 lub <=0 aby wyłączyć

    # liczba epizodów
    try:
        episodes_input = input(f"Podaj liczbę epizodów (Enter = {default_episodes}): ").strip()
        episodes = int(episodes_input) if episodes_input else default_episodes
        if episodes <= 0:
            print(f"Liczba epizodów musi być > 0 – ustawiam {default_episodes}.")
            episodes = default_episodes
    except Exception:
        print(f"Nieprawidłowa wartość – ustawiam {default_episodes}.")
        episodes = default_episodes

    # co ile epizodów pokazywać demo
    try:
        vis_input = input(f"Co ile epizodów pokazywać demo? 0=wyłącz (Enter = {default_visualize_every}): ").strip()
        visualize_every = int(vis_input) if vis_input else default_visualize_every
        if visualize_every <= 0:
            visualize_every = None
    except Exception:
        print("Nieprawidłowa wartość – wyłączam demo.")
        visualize_every = None

    # większy labirynt ~35x90 z pętlami
    env = MazeEnv(
        cells_h=17,
        cells_w=44,
        traps_count=50,
        extra_loops=80,
    )
    q_table = q_learning_train(env, episodes=episodes, visualize_every=visualize_every)

    # Pokaż, jak agent chodzi po labiryncie po treningu
    run_greedy_policy(env, q_table)


if __name__ == "__main__":
    main()
