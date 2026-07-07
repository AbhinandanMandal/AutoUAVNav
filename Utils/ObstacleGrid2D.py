
"""
2D obstacle grid map for UAV navigation.
Idea taken from https://ieeexplore.ieee.org/document/10475692  
"""
from Utils.HyperparametersConfig import Config
cfg = Config()

# Obstacle for the 2D grid


def add_rect(cells, x0, x1, y0, y1):
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            cells.add((x, y))


def build_obstacle_map():
    cells = set()
    add_rect(cells, 6, 8, 27, 29)
    add_rect(cells, 6, 11, 23, 24)
    add_rect(cells, 6, 8, 20, 22)
    add_rect(cells, 0, 6, 16, 17)
    add_rect(cells, 13, 16, 23, 25)
    add_rect(cells, 11, 15, 15, 17)
    add_rect(cells, 10, 12, 6, 8)
    add_rect(cells, 15, 17, 2, 7)
    add_rect(cells, 19, 26, 24, 29)
    add_rect(cells, 20, 24, 8, 9)
    add_rect(cells, 24, 29, 12, 14)
    for i in range(7):
        cells.add((18 + i, 21 - i))
        cells.add((24 + i, 15 + i))
    cells.discard(cfg.start)
    cells.discard(cfg.target)
    return cells
