import argparse
import csv
import json
import math
import os
import random

try:
    import pygame
except Exception:
    pygame = None

# -----------------------------
# Config
# -----------------------------
WIDTH = 900
HEIGHT = 600
FPS = 60

PLAYER_SPEED = 3.0
ENEMY_SPEED = 2.6
BULLET_SPEED = 6.0
BULLET_RANGE = 260
BULLET_COOLDOWN = 18  # frames

MAX_HP = 3

TRAIN_EPISODES = 800
TRAIN_STEPS = 1200

ALPHA = 0.12
GAMMA = 0.95
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 0.995

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "qtable.json")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
TRAIN_LOG_CSV = os.path.join(LOG_DIR, "train_rewards.csv")
TRAIN_PLOT_PNG = os.path.join(LOG_DIR, "train_rewards.png")

# Visual theme
COLOR_BG_TOP = (12, 26, 46)
COLOR_BG_BOTTOM = (28, 52, 76)
COLOR_GRID = (45, 75, 105)
COLOR_PLAYER = (250, 210, 90)
COLOR_ENEMY = (255, 90, 90)
COLOR_BULLET = (235, 235, 255)
COLOR_UI = (230, 240, 255)
COLOR_ACCENT = (120, 210, 240)


# -----------------------------
# Helpers
# -----------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def draw_gradient(surface, top_color, bottom_color):
    h = surface.get_height()
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top_color[0] * (1 - t) + bottom_color[0] * t)
        g = int(top_color[1] * (1 - t) + bottom_color[1] * t)
        b = int(top_color[2] * (1 - t) + bottom_color[2] * t)
        pygame.draw.line(surface, (r, g, b), (0, y), (surface.get_width(), y))


def bar(surface, x, y, w, h, value, max_value, fg, bg):
    pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=6)
    fill = int(w * (value / max_value))
    pygame.draw.rect(surface, fg, (x, y, fill, h), border_radius=6)


def load_qtable(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # keys are stringified tuples
    return {tuple(map(int, k.split(","))): v for k, v in raw.items()}


def save_qtable(qtable, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw = {",".join(str(x) for x in k): v for k, v in qtable.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f)


def try_plot_rewards(csv_path, png_path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    episodes = []
    rewards = []
    moving = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            moving.append(float(row["moving_avg"]))

    if not episodes:
        return False

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, rewards, label="Reward", alpha=0.6)
    plt.plot(episodes, moving, label="Moving Avg", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    plt.close()
    return True


# -----------------------------
# Entities
# -----------------------------

class Bullet:
    def __init__(self, x, y, vx, vy, owner):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.owner = owner
        self.life = 0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life += 1

    def dead(self):
        return self.life * BULLET_SPEED >= BULLET_RANGE


class Agent:
    def __init__(self, x, y, speed, color):
        self.x = x
        self.y = y
        self.speed = speed
        self.color = color
        self.hp = MAX_HP
        self.cooldown = 0
        self.dir = (1, 0)

    def move(self, dx, dy):
        if dx == 0 and dy == 0:
            return False
        mag = math.hypot(dx, dy)
        if mag == 0:
            return False
        dx /= mag
        dy /= mag
        self.dir = (dx, dy)
        self.x += dx * self.speed
        self.y += dy * self.speed
        hit_wall = False
        if self.x < 18:
            self.x = 18
            hit_wall = True
        if self.x > WIDTH - 18:
            self.x = WIDTH - 18
            hit_wall = True
        if self.y < 18:
            self.y = 18
            hit_wall = True
        if self.y > HEIGHT - 18:
            self.y = HEIGHT - 18
            hit_wall = True
        return hit_wall

    def can_shoot(self):
        return self.cooldown <= 0

    def shoot(self):
        if not self.can_shoot():
            return None
        dx, dy = self.dir
        if dx == 0 and dy == 0:
            dx = 1
            dy = 0
        b = Bullet(self.x, self.y, dx * BULLET_SPEED, dy * BULLET_SPEED, self)
        self.cooldown = BULLET_COOLDOWN
        return b

    def tick(self):
        if self.cooldown > 0:
            self.cooldown -= 1


# -----------------------------
# Q-Learning
# -----------------------------

ACTIONS = [
    "UP", "DOWN", "LEFT", "RIGHT", "STAY", "SHOOT"
]


def discretize_state(agent, opponent, bullets):
    # Relative position bins
    dx = opponent.x - agent.x
    dy = opponent.y - agent.y
    rx = 0 if abs(dx) < 60 else (1 if dx > 0 else -1)
    ry = 0 if abs(dy) < 60 else (1 if dy > 0 else -1)

    # Distance bin
    d = dist((agent.x, agent.y), (opponent.x, opponent.y))
    if d < 120:
        db = 0
    elif d < 260:
        db = 1
    else:
        db = 2

    # Opponent direction (coarse)
    odx, ody = opponent.dir
    if abs(odx) > abs(ody):
        od = 0 if odx > 0 else 1
    else:
        od = 2 if ody > 0 else 3

    # Nearest bullet threat
    nearest = 9999.0
    toward = 0
    for b in bullets:
        if b.owner is opponent:
            bd = dist((agent.x, agent.y), (b.x, b.y))
            if bd < nearest:
                nearest = bd
                # Rough heading check
                vx = b.vx
                vy = b.vy
                ax = agent.x - b.x
                ay = agent.y - b.y
                dot = vx * ax + vy * ay
                toward = 1 if dot > 0 else 0
    if nearest < 80:
        bb = 0
    elif nearest < 160:
        bb = 1
    else:
        bb = 2

    ready = 1 if agent.can_shoot() else 0

    return (rx + 1, ry + 1, db, od, bb, toward, ready)


def best_action(qtable, state):
    if state not in qtable:
        qtable[state] = [0.0 for _ in ACTIONS]
    qvals = qtable[state]
    m = max(qvals)
    idxs = [i for i, v in enumerate(qvals) if v == m]
    return random.choice(idxs)


def choose_action(qtable, state, eps):
    if random.random() < eps:
        return random.randrange(len(ACTIONS))
    return best_action(qtable, state)


def apply_action(agent, action, bullets):
    hit_wall = False
    if action == "UP":
        hit_wall = agent.move(0, -1)
    elif action == "DOWN":
        hit_wall = agent.move(0, 1)
    elif action == "LEFT":
        hit_wall = agent.move(-1, 0)
    elif action == "RIGHT":
        hit_wall = agent.move(1, 0)
    elif action == "STAY":
        pass
    elif action == "SHOOT":
        b = agent.shoot()
        if b:
            bullets.append(b)
    return hit_wall


def opponent_policy(agent, opponent, bullets):
    # Simple chase + shoot heuristic
    dx = agent.x - opponent.x
    dy = agent.y - opponent.y
    move_dx = 0
    move_dy = 0
    if abs(dx) > 18:
        move_dx = 1 if dx > 0 else -1
    if abs(dy) > 18:
        move_dy = 1 if dy > 0 else -1
    opponent.move(move_dx, move_dy)

    aligned = abs(dx) < 26 or abs(dy) < 26
    if aligned and opponent.can_shoot():
        opponent.dir = (1, 0) if abs(dx) > abs(dy) and dx > 0 else (
            (-1, 0) if abs(dx) > abs(dy) else (0, 1 if dy > 0 else -1)
        )
        b = opponent.shoot()
        if b:
            bullets.append(b)


# -----------------------------
# Training (headless)
# -----------------------------


def train_qlearning(episodes=TRAIN_EPISODES, steps=TRAIN_STEPS, seed=1, log_every=20):
    random.seed(seed)
    qtable = {}
    eps = EPS_START
    os.makedirs(LOG_DIR, exist_ok=True)
    history = []

    with open(TRAIN_LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "moving_avg", "epsilon", "steps"])

    for ep in range(1, episodes + 1):
        agent = Agent(160, 160, ENEMY_SPEED, COLOR_ENEMY)
        opponent = Agent(WIDTH - 140, HEIGHT - 140, PLAYER_SPEED, COLOR_PLAYER)
        bullets = []
        total_reward = 0.0
        steps_taken = 0

        for _ in range(steps):
            state = discretize_state(agent, opponent, bullets)
            action_idx = choose_action(qtable, state, eps)
            action = ACTIONS[action_idx]

            reward = 0.01  # survival
            hit_wall = apply_action(agent, action, bullets)
            if hit_wall:
                reward -= 0.05

            opponent_policy(agent, opponent, bullets)

            # Update bullets and check collisions
            new_bullets = []
            for b in bullets:
                b.update()
                if b.dead():
                    continue
                # Collision
                if dist((b.x, b.y), (agent.x, agent.y)) < 16 and b.owner is opponent:
                    agent.hp -= 1
                    reward -= 1.0
                    continue
                if dist((b.x, b.y), (opponent.x, opponent.y)) < 16 and b.owner is agent:
                    opponent.hp -= 1
                    reward += 1.2
                    continue
                new_bullets.append(b)
            bullets = new_bullets

            if opponent.hp <= 0:
                reward += 2.0
            if agent.hp <= 0:
                reward -= 2.0

            next_state = discretize_state(agent, opponent, bullets)
            if next_state not in qtable:
                qtable[next_state] = [0.0 for _ in ACTIONS]
            if state not in qtable:
                qtable[state] = [0.0 for _ in ACTIONS]

            target = reward + GAMMA * max(qtable[next_state])
            qtable[state][action_idx] += ALPHA * (target - qtable[state][action_idx])

            total_reward += reward
            steps_taken += 1

            agent.tick()
            opponent.tick()

            if agent.hp <= 0 or opponent.hp <= 0:
                break

        eps = max(EPS_END, eps * EPS_DECAY)
        history.append(total_reward)
        window = history[-25:]
        moving_avg = sum(window) / len(window)

        with open(TRAIN_LOG_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([ep, f"{total_reward:.4f}", f"{moving_avg:.4f}", f"{eps:.4f}", steps_taken])

        if ep % log_every == 0:
            print(f"Episode {ep}/{episodes} reward={total_reward:.2f} avg={moving_avg:.2f} eps={eps:.2f}")

    save_qtable(qtable, MODEL_PATH)
    try_plot_rewards(TRAIN_LOG_CSV, TRAIN_PLOT_PNG)


# -----------------------------
# Game loop
# -----------------------------


def game_loop(mode, ai_side):
    if pygame is None:
        raise RuntimeError("pygame is required. Please install pygame.")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Q-Learning Gunfight")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Georgia", 18)
    big_font = pygame.font.SysFont("Georgia", 30, bold=True)

    qtable = load_qtable(MODEL_PATH)

    player = Agent(140, 140, PLAYER_SPEED, COLOR_PLAYER)
    enemy = Agent(WIDTH - 140, HEIGHT - 140, ENEMY_SPEED, COLOR_ENEMY)
    bullets = []
    running = True
    game_over = False
    last_reward = 0.0
    started = False
    winner = ""

    while running:
        dt = clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                player = Agent(140, 140, PLAYER_SPEED, COLOR_PLAYER)
                enemy = Agent(WIDTH - 140, HEIGHT - 140, ENEMY_SPEED, COLOR_ENEMY)
                bullets = []
                game_over = False
                last_reward = 0.0
                winner = ""
                started = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                started = True

        keys = pygame.key.get_pressed()

        if not started:
            draw_gradient(screen, COLOR_BG_TOP, COLOR_BG_BOTTOM)
            title = big_font.render("Q-Learning Gunfight", True, COLOR_UI)
            subtitle = font.render(f"Mode: {mode.upper()}   Press Enter to start", True, COLOR_ACCENT)
            tips = font.render("WASD move, Space shoot, R reset", True, COLOR_UI)
            screen.blit(title, (WIDTH // 2 - 165, HEIGHT // 2 - 60))
            screen.blit(subtitle, (WIDTH // 2 - 190, HEIGHT // 2 - 20))
            screen.blit(tips, (WIDTH // 2 - 140, HEIGHT // 2 + 18))
            pygame.display.flip()
            continue

        # Player controls
        if not game_over and mode == "human":
            dx = 0
            dy = 0
            if keys[pygame.K_w]:
                dy -= 1
            if keys[pygame.K_s]:
                dy += 1
            if keys[pygame.K_a]:
                dx -= 1
            if keys[pygame.K_d]:
                dx += 1
            player.move(dx, dy)
            if keys[pygame.K_SPACE]:
                b = player.shoot()
                if b:
                    bullets.append(b)

        # AI control in AI mode
        if not game_over and mode == "ai":
            # AI controls player, enemy is heuristic
            state = discretize_state(player, enemy, bullets)
            action_idx = best_action(qtable, state) if qtable else random.randrange(len(ACTIONS))
            action = ACTIONS[action_idx]
            last_reward = 0.0
            hit_wall = apply_action(player, action, bullets)
            if hit_wall:
                last_reward -= 0.05
        elif not game_over and mode == "human":
            # AI controls enemy
            state = discretize_state(enemy, player, bullets)
            action_idx = best_action(qtable, state) if qtable else random.randrange(len(ACTIONS))
            action = ACTIONS[action_idx]
            last_reward = 0.0
            hit_wall = apply_action(enemy, action, bullets)
            if hit_wall:
                last_reward -= 0.05

        if not game_over:
            # Heuristic opponent in AI mode
            if mode == "ai":
                opponent_policy(player, enemy, bullets)

            # Update bullets
            new_bullets = []
            for b in bullets:
                b.update()
                if b.dead():
                    continue
                if dist((b.x, b.y), (player.x, player.y)) < 16 and b.owner is enemy:
                    player.hp -= 1
                    last_reward -= 1.0
                    continue
                if dist((b.x, b.y), (enemy.x, enemy.y)) < 16 and b.owner is player:
                    enemy.hp -= 1
                    last_reward += 1.0
                    continue
                new_bullets.append(b)
            bullets = new_bullets

            if player.hp <= 0 or enemy.hp <= 0:
                game_over = True
                winner = "PLAYER" if enemy.hp <= 0 else "ENEMY"

        player.tick()
        enemy.tick()

        # Render
        draw_gradient(screen, COLOR_BG_TOP, COLOR_BG_BOTTOM)
        # subtle grid
        for x in range(0, WIDTH, 45):
            pygame.draw.line(screen, COLOR_GRID, (x, 0), (x, HEIGHT), 1)
        for y in range(0, HEIGHT, 45):
            pygame.draw.line(screen, COLOR_GRID, (0, y), (WIDTH, y), 1)

        # Entities
        pygame.draw.circle(screen, player.color, (int(player.x), int(player.y)), 14)
        pygame.draw.circle(screen, enemy.color, (int(enemy.x), int(enemy.y)), 14)

        for b in bullets:
            pygame.draw.circle(screen, COLOR_BULLET, (int(b.x), int(b.y)), 4)

        # UI
        title = big_font.render("Q-Learning Gunfight", True, COLOR_UI)
        screen.blit(title, (18, 14))

        bar(screen, 18, 58, 180, 16, player.hp, MAX_HP, COLOR_PLAYER, (40, 40, 40))
        bar(screen, 18, 80, 180, 16, enemy.hp, MAX_HP, COLOR_ENEMY, (40, 40, 40))

        mode_text = font.render(f"Mode: {mode.upper()}", True, COLOR_UI)
        screen.blit(mode_text, (220, 58))
        reward_text = font.render(f"Last reward: {last_reward:+.2f}", True, COLOR_UI)
        screen.blit(reward_text, (220, 80))

        # State/action HUD for AI
        if mode == "ai":
            ai_agent = player
            ai_opponent = enemy
        else:
            ai_agent = enemy
            ai_opponent = player
        ai_state = discretize_state(ai_agent, ai_opponent, bullets)
        ai_action = "N/A"
        if qtable:
            ai_action = ACTIONS[best_action(qtable, ai_state)]
        hud = font.render(f"AI state: {ai_state}  action: {ai_action}", True, COLOR_UI)
        screen.blit(hud, (220, 104))

        hint = font.render("WASD move, Space shoot, R reset", True, COLOR_ACCENT)
        screen.blit(hint, (18, HEIGHT - 32))

        if game_over:
            pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() * 0.01)
            color = (
                int(COLOR_UI[0] * (0.6 + 0.4 * pulse)),
                int(COLOR_UI[1] * (0.6 + 0.4 * pulse)),
                int(COLOR_UI[2] * (0.6 + 0.4 * pulse)),
            )
            msg = big_font.render(f"{winner} WINS", True, color)
            screen.blit(msg, (WIDTH // 2 - 90, HEIGHT // 2 - 16))

        pygame.display.flip()

    pygame.quit()


# -----------------------------
# CLI
# -----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["human", "ai"], default="human")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.train:
        print("Training Q-table...")
        train_qlearning(episodes=args.episodes, steps=args.steps, seed=args.seed, log_every=args.log_every)
        print(f"Saved model to {MODEL_PATH}")

    if args.train_only:
        return

    if pygame is None:
        print("pygame is required to run the game. Install with: pip install pygame")
        return

    game_loop(args.mode, ai_side="enemy")


if __name__ == "__main__":
    main()

