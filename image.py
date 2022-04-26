import pygame

def load(img_path, size="default", convert="alpha", flip=False):
    if convert == "alpha":
        img = pygame.image.load(img_path).convert_alpha()
    else:
        img = pygame.image.load(img_path).convert()
    if size != "default":
        img = pygame.transform.scale(img, size)
    if flip:
        img = pygame.transform.flip(img, True, False)
    return img

def scale(img, size):
    return pygame.transform.scale(img, size)

def draw(screen, img, pos, pos_mode="top_left"):
    if pos_mode == "center":
        pos = list(pos)
        pos[0] -= img.get_width() // 2
        pos[1] -= img.get_height() // 2
    screen.blit(img, pos)