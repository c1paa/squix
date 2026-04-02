# Snake Game
import pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Snake')

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Snake settings
BLOCK_SIZE = 20
SPEED = 15

# Initial snake position
snake = [[100, 50]]
direction = 'RIGHT'

# Food
food = [300, 200]

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT and direction != 'RIGHT':
                direction = 'LEFT'
            elif event.key == pygame.K_RIGHT and direction != 'LEFT':
                direction = 'RIGHT'
            elif event.key == pygame.K_UP and direction != 'DOWN':
                direction = 'UP'
            elif event.key == pygame.K_DOWN and direction != 'UP':
                direction = 'DOWN'

    # Move snake
    head = snake[0].copy()
    if direction == 'LEFT':
        head[0] -= BLOCK_SIZE
    elif direction == 'RIGHT':
        head[0] += BLOCK_SIZE
    elif direction == 'UP':
        head[1] -= BLOCK_SIZE
    elif direction == 'DOWN':
        head[1] += BLOCK_SIZE
    snake.insert(0, head)

    # Check food collision
    if head == food:
        food = [pygame.randint(0, WIDTH-BLOCK_SIZE), pygame.randint(0, HEIGHT-BLOCK_SIZE)]
    else:
        snake.pop()

    # Draw
    screen.fill(BLACK)
    for segment in snake:
        pygame.draw.rect(screen, GREEN, (*segment, BLOCK_SIZE, BLOCK_SIZE))
    pygame.draw.rect(screen, RED, (*food, BLOCK_SIZE, BLOCK_SIZE))
    pygame.display.flip()

    # Speed control
    pygame.time.delay(SPEED)

pygame.quit()