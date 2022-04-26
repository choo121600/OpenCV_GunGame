import time
import random
from settings import *
from hands_tracking import HandTracking
from dot import Dot
from target import Target
import cv2
import ui


class Game:
    def __init__(self, screen):
        self.screen = screen
        
        self.cap = cv2.VideoCapture(0)
        self.hand_tracking = HandTracking()
        self.dot = Dot()
        self.bullet = BULLET
        # self.target = Target()
        self.targets = []
        self.target_spawn_timer = 0
        self.score = 0
        self.flag = 0

    # def reset(self):
    #     # self.hand = 

    def show_hand_location(self):
        self.img = self.hand_tracking.hand_location(self.img)

    def spawn_target(self):
        t = time.time()
        if t > self.target_spawn_timer:
            self.target_spawn_timer = t + TARGET_SPAWN_TIME
            self.targets.append(Target())

    def draw(self):
        myfont = pygame.font.SysFont("monospace", 16)
        self.screen.fill([255, 255, 255])
        for target in self.targets:
            target.draw(self.screen)
        self.dot.draw(self.screen, self.bullet)
        text = myfont.render("SCORE {0}".format(self.score), 1, (0, 0 ,0))
        self.screen.blit(text, (5, 10))
        bul_text = myfont.render("BULLET {0}".format(self.bullet), 1, (0, 0 ,0))
        self.screen.blit(bul_text, (400, 10))

    def update(self):
        ret, self.img = self.cap.read()
        self.show_hand_location()
        self.draw()
        self.spawn_target()

        (x, y) = self.hand_tracking.get_hand_center()
        self.dot.rect.center = (x, y)
        self.dot.shoot = self.hand_tracking.shot
        self.dot.refill = self.hand_tracking.refill
        self.score = self.dot.kill_targets(self.targets, self.score, self.bullet)
        if self.bullet > 0:
            if self.dot.shoot and self.flag == 0:
                self.flag = 1
                print(self.flag)
                self.bullet = self.dot.bull_left(self.bullet)
            elif self.dot.shoot == False:
                self.flag = 0
                print(self.flag)
        else:
            if self.dot.refill:
                self.bullet = self.dot.bull_left(self.bullet)

        cv2.resizeWindow('frame', width = SCREEN_WIDTH, height = SCREEN_HEIGHT)
        cv2.imshow("Controller", self.img)
        cv2.waitKey(1)