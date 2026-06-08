

import pygame
from drone_simulation_env import DroneGridEnv, build_assets, randomize_obstacles, draw_environment
pygame.init()  # Initialize PyGame window

# PyGame window values
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 720
FPS = 60


def main():
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("UAV 2D Environment Simulator")
    clock = pygame.time.Clock()
    env = DroneGridEnv()
    assets = build_assets()
    fonts = {
        "title": pygame.font.SysFont("Arial", 28, bold=True),
        "body": pygame.font.SysFont("Arial", 20),
        "small": pygame.font.SysFont("Arial", 15),
    }

    actions = {
        pygame.K_UP: (0, -1),
        pygame.K_w: (0, -1),
        pygame.K_DOWN: (0, 1),
        pygame.K_s: (0, 1),
        pygame.K_LEFT: (-1, 0),
        pygame.K_a: (-1, 0),
        pygame.K_RIGHT: (1, 0),
        pygame.K_d: (1, 0),
    }

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                elif event.key == pygame.K_m:
                    randomize_obstacles(env)
                elif event.key in actions:
                    env.step(actions[event.key])

        draw_environment(screen, env, assets, fonts)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
