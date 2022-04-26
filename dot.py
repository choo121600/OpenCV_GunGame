import pygame
import image
from settings import *
from hands_tracking import HandTracking
import cv2

class Dot:
    def __init__(self):
        self.original_img = image.load("./assets/dot.jpeg", size = (DOT_SIZE, DOT_SIZE))
        self.img = self.original_img.copy()
        self.rect = pygame.Rect(SCREEN_HEIGHT//2, SCREEN_WIDTH//2, DOT_HIT_SIZE[0], DOT_HIT_SIZE[1])
        # self.img_smaller
        self.shoot = False
        self.refill = False
        # self.game = Game()

    def draw_hit_rect(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), self.rect)

    def draw(self, screen, bullet):
        self.draw_hit_rect(screen)
        image.draw(screen, self.img, self.rect.center, pos_mode='center')
        if bullet == 0:
            myfont = pygame.font.SysFont("monospace", 11)
            text = myfont.render("Need To Reload Bullet", 1, (0, 0, 0))
            screen.blit(text, (self.rect[0] + 15, self.rect.center[1] + 15))


    def on_target(self, targets):
        return [target for target in targets if self.rect.colliderect(target.rect)]

    def bull_left(self, bullet):
        if bullet > 0:
            if self.shoot:
                bullet -= 1
        else:
            if self.refill:
                bullet += 20
        return bullet

    def kill_targets(self, targets, score, bullet):
        if bullet > 0:
            if self.shoot:
                for target in self.on_target(targets):
                    target_score = target.kill(targets)
                    score += target_score
            else:
                self.shoot = False
        else:
            self.shoot = False
        return score