import pygame
import random
import image
from settings import *

### TASK ###
"""
1. 탄도학
"""
class Target:
    def __init__(self):
        random_size_value = random.uniform(TARGET_SIZE_RANDOM[0], TARGET_SIZE_RANDOM[1])
        self.size = (int(TARGET_SIZE[0] * random_size_value), int(TARGET_SIZE[1] * random_size_value))
        self.mid = (TARGET_SIZE[0] * 2 - TARGET_SIZE[1] * 1) / 10 * 4
        # self.mid = TARGET_SIZE[1] + (TARGET_SIZE[0] * 2 - TARGET_SIZE[1] * 1) / 10 * 4

        ### Random postion ###
        self.rect = pygame.Rect(random.randint(0, SCREEN_WIDTH - self.size[0]-20), random.randint(0, SCREEN_HEIGHT - self.size[1]-90), TARGET_HIT_SIZE[0] * self.size[0] // 100, TARGET_HIT_SIZE[1] * self.size[1] // 100)
        self.img = image.load("./assets/target.png", size = self.size)
        self.hit_area = self.rect.center[1] - (self.size[1] - self.mid)
        print(self.size, self.mid)
        print(self.hit_area)

    def draw_hit_rect(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), self.rect)

    def draw(self, screen):
        myfont = pygame.font.SysFont("monospace", 11)
        # image.draw(screen, self.img, (self.rect.center[0], self.rect.center[1] - 2 * (self.size[1] - self.mid)), pos_mode='center')
        image.draw(screen, self.img, (self.rect.center[0], self.hit_area), pos_mode='center')
        self.draw_hit_rect(screen)
        text = myfont.render("{0} M".format(250 - self.size[1]*3), 1, (0, 0 ,0))
        screen.blit(text, (self.rect[0]+15, self.rect.center[1]+self.size[1]/2))
        
    def kill(self, targets):
        print("kill target")
        targets.remove(self)
        return 1