import pygame
import sys
import os
from game import Game
from settings import *

### TASK ###
"""
1. pygame 화면 크기 설정
2. Loop 설정

game.py
hand_tracking.py
settings.py
dot.py
image.py
target.py
"""

##### Default Set Up #####
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 0)
pygame.init()
pygame.display.set_caption(TITLE)
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# screen.fill([255, 255, 255]

mainClock = pygame.time.Clock()


##### State #####
state = "game"
game = Game(SCREEN)

##### Default Functions #####
def user_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

def update():
    global state
    if state == "game":
        game.update()
    
    pygame.display.update()
    mainClock.tick(FPS)


##### Main Loop #####
while True:
    user_events()
    update()
    SCREEN.fill([255, 255, 255])
