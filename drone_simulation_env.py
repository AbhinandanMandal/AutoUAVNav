import os
import random
import pygame
from dataclasses import dataclass


# Grid Env. configurations
START_CELL = (0, 0)
TARGET_CELL = (14, 9)
STEP_REWARD = 1  # Drone get 1 points for each non-collision movement
GRID_COLS = 15
GRID_ROWS = 10

# PyGame window values
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
GRID_COLS = 15
GRID_ROWS = 10
CELL_SIZE = 56
GRID_LEFT = 80
GRID_TOP = 88
FPS = 60


# Colors of PyGame window
BACKGROUND = (255, 255, 255)
PANEL = (30, 43, 51)
GRID_LINE = (64, 82, 92)
SAFE_CELL = (38, 58, 62)
START_COLOR = (60, 154, 120)
TARGET_COLOR = (232, 185, 73)
TEXT = (18, 27, 34)
MUTED_TEXT = (100, 110, 115)
FAIL_RED = (229, 88, 84)
SUCCESS_GREEN = (76, 190, 128)


# Graphical Assets
ASSET_DIR = "graphics"
ASSETS = {
    "drone": "drone.png",
    "target": "target.png",
    "building": "apartments.png",
    "tree": "tree.png",
}


"""
Obstacle class responsible for obstacles in the PyGame 2D simulation environment
"""


@dataclass(frozen=True)
class Obstacle:
    cell: tuple[int, int]
    kind: str


class DroneGridEnv:
    def __init__(self):
        # Places of obstacles in 2D environment
        # Obstacles are tunable for extensive study and analysis
        self.obstacles = [
            Obstacle((3, 0), "tree"),
            Obstacle((7, 0), "building"),
            Obstacle((11, 0), "tree"),
            # Obstacle((1, 2), "building"),

            Obstacle((4, 2), "tree"),
            Obstacle((8, 2), "building"),
            # Obstacle((12, 2), "tree"),
            Obstacle((5, 4), "building"),

            Obstacle((9, 4), "tree"),
            Obstacle((13, 4), "building"),
            Obstacle((2, 5), "tree"),
            # Obstacle((6, 6), "building"),

            Obstacle((10, 6), "tree"),
            Obstacle((4, 8), "building"),
            Obstacle((8, 8), "tree"),
            # Obstacle((12, 8), "building"),

            Obstacle((7, 9), "building"),
            Obstacle((8, 4), "tree"),
            # Obstacle((10, 4), "tree"),
            Obstacle((5, 5), "building"),

            Obstacle((14, 2), "tree"),
            Obstacle((9, 5), "building"),
            # Obstacle((13, 2), "tree"),
            Obstacle((0, 1), "tree"),

            Obstacle((2, 1), "building"),
            # Obstacle((6, 1), "tree"),
            Obstacle((9, 1), "building"),
            Obstacle((11, 1), "tree"),

            Obstacle((3, 3), "building"),
            Obstacle((6, 3), "tree"),
            # Obstacle((11, 3), "building"),
            Obstacle((14, 3), "tree"),

            Obstacle((0, 4), "building"),
            Obstacle((2, 4), "tree"),
            # Obstacle((1, 6), "building"),
            Obstacle((3, 6), "tree"),

            Obstacle((8, 6), "building"),
            Obstacle((13, 6), "tree"),
            # Obstacle((0, 7), "tree"),
            Obstacle((5, 7), "building"),

            # Obstacle((10, 7), "tree"),
            # Obstacle((14, 7), "building"),

            # Obstacle((1, 8), "tree"),
            # Obstacle((6, 8), "building"),

            # Obstacle((2, 9), "building"),
            # Obstacle((5, 9), "tree"),
            # Obstacle((11, 9), "building"),

        ]
        self.obstacle_cells = {obstacle.cell for obstacle in self.obstacles}
        self.reset()

    def reset(self):  # In case of reset 2D environment
        self.drone_cell = START_CELL
        self.score = 0
        self.steps = 0
        self.done = False
        self.failed = False
        self.won = False
        self.message = "use Arrow keys or W/S/A/D to reach the target state."
        return self._state()

    def step(self, action):
        if self.done:
            return self._state(), 0, self.done, self.info()

        # Choosing the next step in simulation
        dx, dy = action
        col, row = self.drone_cell
        next_cell = (
            max(0, min(GRID_COLS - 1, col+dx)),
            max(0, min(GRID_ROWS - 1, row+dy))
        )

        # Condition 1: boundary reached when next_cell == drone_cell
        if next_cell == self.drone_cell:
            self.message = "Boundary reached. Choose another direction."
            return self._state(), 0, self.done, self.info()

        self.steps += 1
        self.drone_cell = next_cell

        # Condition 2: If obstacle in next step then mission failed
        if next_cell in self.obstacle_cells:
            self.failed = True
            self.done = True
            self.message = "Collision detected. Mission failed."
            return self._state(), -100, self.done, self.info()

        self.score += STEP_REWARD

        # Condition 3: if next_cell is target cell then mission complete
        if next_cell == TARGET_CELL:
            self.won = True
            self.done = True
            self.message = "Target reached. Mission complete."
            return self._state(), 100, self.done, self.info()

        self.message = f"Safe step. +{STEP_REWARD} points."
        return self._state(), STEP_REWARD, self.done, self.info()

    # Info of each actions in simulation
    def info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "failed": self.failed,
            "won": self.won
        }

    # Current state of UAV in 2D environment
    def _state(self):
        return {
            "drone": self.drone_cell,
            "target": TARGET_CELL,
            "obstacles": tuple(self.obstacle_cells)
        }


# Loadding assest to make visualization better
def load_image(name, size):
    path = os.path.join(ASSET_DIR, ASSETS[name])
    if not os.path.exists(path):
        return None

    image = pygame.image.load(path).convert_alpha()
    return pygame.transform.smoothscale(image, size)


# Pygame.Rect is an an object used to store and manipulate rectangular coordinates
def cell_rect(cell):
    col, row = cell
    return pygame.Rect(
        GRID_LEFT + col * CELL_SIZE,
        GRID_TOP + row * CELL_SIZE,
        CELL_SIZE,
        CELL_SIZE,
    )


# Some functionality
def draw_centered_text(surface, font, text, color, center):
    rendered = font.render(text, True, color)
    rect = rendered.get_rect(center=center)
    surface.blit(rendered, rect)


def draw_asset_or_fallback(surface, image, rect, fallback_color, label=""):
    if image:
        surface.blit(image, image.get_rect(center=rect.center))
        return

    pygame.draw.rect(surface, fallback_color,
                     rect.inflate(-12, -12), border_radius=8)
    if label:
        font = pygame.font.SysFont("Arial", 13, bold=True)
        draw_centered_text(surface, font, label, TEXT, rect.center)


# Drawing 2D environment
# For much visual rich visualization LLM usage has been taken in this section
def draw_environment(screen, env, assets, fonts):
    screen.fill(BACKGROUND)

    grid_rect = pygame.Rect(
        GRID_LEFT,
        GRID_TOP,
        GRID_COLS * CELL_SIZE,
        GRID_ROWS * CELL_SIZE,
    )
    pygame.draw.rect(screen, PANEL, grid_rect.inflate(16, 16), border_radius=8)

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            rect = cell_rect((col, row))
            pygame.draw.rect(screen, SAFE_CELL, rect)
            pygame.draw.rect(screen, GRID_LINE, rect, 1)

    pygame.draw.rect(screen, START_COLOR, cell_rect(
        START_CELL).inflate(-8, -8), 3, border_radius=6)
    draw_asset_or_fallback(screen, assets["target"], cell_rect(
        TARGET_CELL), TARGET_COLOR, "GOAL")

    for obstacle in env.obstacles:
        image = assets[obstacle.kind]
        color = (101, 130, 83) if obstacle.kind == "tree" else (118, 111, 104)
        label = "TREE" if obstacle.kind == "tree" else "BLDG"
        draw_asset_or_fallback(
            screen, image, cell_rect(obstacle.cell), color, label)

    draw_asset_or_fallback(screen, assets["drone"], cell_rect(
        env.drone_cell), (80, 165, 220), "UAV")

    title = fonts["title"].render("2D UAV Grid Research Simulator", True, TEXT)
    screen.blit(title, (GRID_LEFT, 26))

    status_color = SUCCESS_GREEN if env.won else FAIL_RED if env.failed else MUTED_TEXT
    status = fonts["body"].render(env.message, True, status_color)
    screen.blit(status, (GRID_LEFT, 58))

    side_x = GRID_LEFT + GRID_COLS * CELL_SIZE + 40
    panel_rect = pygame.Rect(side_x - 20, GRID_TOP, 150, 236)
    pygame.draw.rect(screen, PANEL, panel_rect, border_radius=8)

    info_lines = [
        ("Score", str(env.score)),
        ("Steps", str(env.steps)),
        ("Position", f"{env.drone_cell}"),
        ("Reward", f"+{STEP_REWARD}/safe step"),
        ("Crash", "instant fail"),
    ]

    y = GRID_TOP + 18
    for label, value in info_lines:
        screen.blit(fonts["small"].render(
            label, True, MUTED_TEXT), (side_x, y))
        screen.blit(fonts["body"].render(value, True, TEXT), (side_x, y + 20))
        y += 42

    controls = [
        "Move: Arrow keys / WASD",
        "Reset: R",
        "Random map: M",
        "Quit: Esc",
    ]
    y = WINDOW_HEIGHT - 88
    for line in controls:
        screen.blit(fonts["small"].render(
            line, True, MUTED_TEXT), (GRID_LEFT, y))
        y += 22

    if env.done:
        overlay = pygame.Surface((GRID_COLS * CELL_SIZE, 118), pygame.SRCALPHA)
        overlay.fill((8, 13, 18, 218))
        overlay_rect = overlay.get_rect(
            center=(grid_rect.centerx, grid_rect.centery))
        screen.blit(overlay, overlay_rect)

        heading = "YOU WON" if env.won else "MISSION FAILED"
        color = SUCCESS_GREEN if env.won else FAIL_RED
        draw_centered_text(
            screen, fonts["title"], heading, color, (grid_rect.centerx, grid_rect.centery - 22))
        draw_centered_text(
            screen,
            fonts["body"],
            "Press R to run another episode",
            TEXT,
            (grid_rect.centerx, grid_rect.centery + 24),
        )


# Asset in 2D environment
def build_assets():
    icon_size = (CELL_SIZE - 10, CELL_SIZE - 10)
    return {
        "drone": load_image("drone", icon_size),
        "target": load_image("target", icon_size),
        "building": load_image("building", icon_size),
        "tree": load_image("tree", icon_size),
    }


# What if at every time, we have a new 2D environment for UAV simulation
# Randomized obstacles for random 2D environment
def randomize_obstacles(env):
    reserved = {START_CELL, TARGET_CELL, (1, 0), (0, 1), (13, 9), (14, 8)}
    cells = [
        (col, row)
        for row in range(GRID_ROWS)
        for col in range(GRID_COLS)
        if (col, row) not in reserved
    ]
    random.shuffle(cells)
    env.obstacles = [
        Obstacle(cell, "tree" if index % 2 else "building")
        # obstacles right now is 16, it can tunable like 20, 30 etc, Total obstacle right now 50
        for index, cell in enumerate(cells[:30])
    ]
    env.obstacle_cells = {obstacle.cell for obstacle in env.obstacles}
    env.reset()
    env.message = "Random 2D Environment Loaded."
