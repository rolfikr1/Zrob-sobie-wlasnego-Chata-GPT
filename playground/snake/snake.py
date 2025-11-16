#!/usr/bin/env python3
import pygame
import random
from collections import deque, defaultdict

# --- Config ---
WIDTH, HEIGHT = 400, 400           # rozmiar LOGICZNEJ planszy
BLOCK_SIZE = 20
GRID_W = WIDTH // BLOCK_SIZE
GRID_H = HEIGHT // BLOCK_SIZE

HUD_HEIGHT = 40                    # pasek info nad planszą
WINDOW_WIDTH = WIDTH
WINDOW_HEIGHT = HEIGHT + HUD_HEIGHT

SNAKE_COLOR = (0, 200, 0)
FOOD_COLOR = (200, 0, 0)
BG_COLOR = (30, 30, 30)
HUD_BG_COLOR = (15, 15, 15)
HEAD_COLOR = (0, 255, 0)
TEXT_COLOR = (220, 220, 220)

# Actions: 0 = straight, 1 = right, 2 = left
ACTIONS = [0, 1, 2]


class SnakeGame:
    def __init__(self, render=False):
        self.render = render
        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("consolas", 18)
        self.reset()

    def reset(self):
        self.direction = (1, 0)  # start: w prawo
        self.head = (GRID_W // 2, GRID_H // 2)
        self.snake = deque([
            self.head,
            (self.head[0] - 1, self.head[1]),
            (self.head[0] - 2, self.head[1]),
        ])
        self.score = 0
        self.frame_iteration = 0
        self._place_food()
        return self._get_state()

    def _place_food(self):
        while True:
            x = random.randint(0, GRID_W - 1)
            y = random.randint(0, GRID_H - 1)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        x, y = pt
        # ściany
        if x < 0 or x >= GRID_W or y < 0 or y >= GRID_H:
            return True
        # ugryzienie siebie
        if pt in list(self.snake)[1:]:
            return True
        return False

    def _get_state(self):
        # Stan: zagrożenie (prosto/lewo/prawo), kierunek, położenie jedzenia
        dir_x, dir_y = self.direction

        left = (-dir_y, dir_x)
        right = (dir_y, -dir_x)

        head_x, head_y = self.head
        pt_straight = (head_x + dir_x, head_y + dir_y)
        pt_left = (head_x + left[0], head_y + left[1])
        pt_right = (head_x + right[0], head_y + right[1])

        danger_straight = int(self._is_collision(pt_straight))
        danger_left = int(self._is_collision(pt_left))
        danger_right = int(self._is_collision(pt_right))

        dir_right = int(self.direction == (1, 0))
        dir_left = int(self.direction == (-1, 0))
        dir_up = int(self.direction == (0, -1))
        dir_down = int(self.direction == (0, 1))

        food_left = int(self.food[0] < head_x)
        food_right = int(self.food[0] > head_x)
        food_up = int(self.food[1] < head_y)
        food_down = int(self.food[1] > head_y)

        state = (
            danger_straight,
            danger_left,
            danger_right,
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            food_left,
            food_right,
            food_up,
            food_down,
        )
        return state

    def step(self, action):
        self.frame_iteration += 1

        # dystans do jedzenia PRZED ruchem (Manhattan)
        old_dist = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])

        # 0 = prosto, 1 = w prawo, 2 = w lewo
        dir_x, dir_y = self.direction
        if action == 1:  # w prawo
            self.direction = (dir_y, -dir_x)
        elif action == 2:  # w lewo
            self.direction = (-dir_y, dir_x)

        dir_x, dir_y = self.direction
        new_head = (self.head[0] + dir_x, self.head[1] + dir_y)
        self.head = new_head

        reward = -0.05
        done = False

        # śmierć albo zacięcie (limit kroków zależny od długości)
        if self._is_collision(new_head) or self.frame_iteration > 150 * len(self.snake):
            done = True
            reward = -15
            return self._get_state(), reward, done, self.score

        self.snake.appendleft(new_head)

        if new_head == self.food:
            self.score += 1
            reward = 15
            self._place_food()
        else:
            self.snake.pop()

            # dystans do jedzenia PO ruchu
            new_dist = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])

            if new_dist < old_dist:
                reward += 0.2   # ruch w dobrą stronę
            elif new_dist > old_dist:
                reward -= 0.2   # ruch w złą stronę

        if self.render:
            self._update_ui()
            self.clock.tick(15)

        return self._get_state(), reward, done, self.score

    def _direction_str(self):
        if self.direction == (1, 0):
            return "RIGHT"
        if self.direction == (-1, 0):
            return "LEFT"
        if self.direction == (0, -1):
            return "UP"
        if self.direction == (0, 1):
            return "DOWN"
        return str(self.direction)

    def _update_ui(self):
        # tło
        self.display.fill(BG_COLOR)

        # HUD
        pygame.draw.rect(
            self.display,
            HUD_BG_COLOR,
            pygame.Rect(0, 0, WINDOW_WIDTH, HUD_HEIGHT)
        )

        info = f"Dir: {self._direction_str()}   Len: {len(self.snake)}   Score: {self.score}   Steps: {self.frame_iteration}"
        text_surf = self.font.render(info, True, TEXT_COLOR)
        self.display.blit(text_surf, (10, 10))

        # rysowanie planszy przesuniętej w dół o HUD_HEIGHT
        for i, pt in enumerate(self.snake):
            color = HEAD_COLOR if i == 0 else SNAKE_COLOR
            x, y = pt
            pygame.draw.rect(
                self.display,
                color,
                pygame.Rect(
                    x * BLOCK_SIZE,
                    y * BLOCK_SIZE + HUD_HEIGHT,
                    BLOCK_SIZE - 2,
                    BLOCK_SIZE - 2,
                ),
            )

        fx, fy = self.food
        pygame.draw.rect(
            self.display,
            FOOD_COLOR,
            pygame.Rect(
                fx * BLOCK_SIZE,
                fy * BLOCK_SIZE + HUD_HEIGHT,
                BLOCK_SIZE - 2,
                BLOCK_SIZE - 2,
            ),
        )

        pygame.display.flip()


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.Q = defaultdict(lambda: [0.0 for _ in ACTIONS])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        q_values = self.Q[state]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(ACTIONS, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        q_values = self.Q[state]
        a_idx = ACTIONS.index(action)
        q_old = q_values[a_idx]

        if done:
            target = reward
        else:
            next_q = max(self.Q[next_state])
            target = reward + self.gamma * next_q

        q_values[a_idx] = q_old + self.alpha * (target - q_old)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def run_demo_episode(agent, max_steps=5000):
    """Demo jednej gry w trakcie treningu (render, epsilon=0)."""
    game = SnakeGame(render=True)
    # zapamiętaj epsilon, żeby nie rozwalić eksploracji
    old_eps = agent.epsilon
    agent.epsilon = 0.0

    state = game.reset()
    done = False
    steps = 0
    score = 0

    while not done and steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        action = agent.select_action(state)
        next_state, reward, done, score = game.step(action)
        state = next_state
        steps += 1

    print(f"[DEMO] score: {score}, steps: {steps}")
    pygame.quit()
    agent.epsilon = old_eps


def train(num_episodes=500, demo_interval=100):
    game = SnakeGame(render=False)
    agent = QLearningAgent()
    scores = []
    best_score = 0

    for ep in range(1, num_episodes + 1):
        state = game.reset()
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, score = game.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        agent.decay_epsilon()
        scores.append(score)
        best_score = max(best_score, score)

        if ep % 50 == 0:
            print(f"Ep {ep}/{num_episodes} | score: {score} | best: {best_score} | eps: {agent.epsilon:.3f}")

        # DEMO co demo_interval epizodów (jeśli ustawione sensownie)
        if demo_interval and demo_interval > 0 and ep % demo_interval == 0:
            print(f"\n=== DEMO po epizodzie {ep} ===")
            run_demo_episode(agent)
            print("=== Koniec dema ===\n")

    return agent, scores


def demo(agent):
    """Końcowe demo po treningu."""
    game = SnakeGame(render=True)
    agent.epsilon = 0.0  # czysto zachłannie
    running = True
    state = game.reset()
    done = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                done = True

        if done:
            print("Final score:", game.score)
            state = game.reset()
            done = False

        action = agent.select_action(state)
        next_state, reward, done, score = game.step(action)
        state = next_state

    pygame.quit()


def get_int_input(prompt, default):
    """Bezpieczne wczytanie int z klawiatury z domyślną wartością."""
    try:
        txt = input(prompt).strip()
        if txt == "":
            return default
        value = int(txt)
        return value
    except Exception:
        print(f"Nieprawidłowa wartość, używam domyślnej: {default}")
        return default


if __name__ == "__main__":
    print("=== Snake RL – trening z interaktywnymi ustawieniami ===")
    num_episodes = get_int_input("Podaj liczbę epok (domyślnie 500): ", 500)
    demo_interval = get_int_input("Co ile epok odpalać demo? (0 = bez dema w trakcie, domyślnie 100): ", 100)

    if demo_interval < 0:
        print("Demo co ujemną liczbę epok nie istnieje, ustawiam 100.")
        demo_interval = 100

    print(f"\nStart treningu: epok = {num_episodes}, demo co {demo_interval if demo_interval > 0 else 'brak'} epok.")
    agent, scores = train(num_episodes=num_episodes, demo_interval=demo_interval)
    print("Trening zakończony. Odpalam finalne demo...")
    demo(agent)
