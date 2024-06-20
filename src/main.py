import pygame
import os
from game import run

pygame.init()
pygame.display.set_mode((600, 800))  # Set the proper display mode for the game

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'flappy_bird.ini')
    run(config_path)
