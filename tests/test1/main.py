import pygame
import random
import sys

pygame.init()

# Screen dimensions
screen_width = 600
screen_height = 400
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Snake Game')

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
red = (213, 50, 80)
green = (0, 255, 0)

clock = pygame.time.Clock()
BLOCK_SIZE = 10

font = pygame.font.SysFont("arial", 25)

class Snake:
    def __init__(self):
        self.body = [[screen_width // 2, screen_height // 2]]
        self.direction = [1, 0]  # Moving right
        self.growing = False

    def move(self):
        new_head = [
            self.body[0][0] + self.direction[0] * BLOCK_SIZE,
            self.body[0][1] + self.direction[1] * BLOCK_SIZE
        ]
        self.body.insert(0, new_head)
        if not self.growing:
            self.body.pop()
        else:
            self.growing = False

    def grow(self):
        self.growing = True

    def change_direction(self, new_dir):
        if [new_dir[0] + self.direction[0], new_dir[1] + self.direction[1]] != [0, 0]:
            self.direction = new_dir

    def get_head(self):
        return self.body[0]

    def check_collision(self):
        if self.body[0] in self.body[1:]:
            return True
        if (self.body[0][0] < 0 or self.body[0][0] >= screen_width or
            self.body[0][1] < 0 or self.body[0][1] >= screen_height):
            return True
        return False

class Food:
    def __init__(self, snake_body):
        self.position = self.generate_position(snake_body)

    def generate_position(self, snake_body):
        while True:
            pos = [
                random.randint(0, (screen_width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE,
                random.randint(0, (screen_height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            ]
            if pos not in snake_body:
                return pos

    def respawn(self, snake_body):
        self.position = self.generate_position(snake_body)

def draw_snake(snake):
    for segment in snake.body:
        pygame.draw.rect(screen, white, pygame.Rect(segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE))

def draw_food(food):
    pygame.draw.rect(screen, green, pygame.Rect(food.position[0], food.position[1], BLOCK_SIZE, BLOCK_SIZE))

def display_score(score):
    text = font.render(f"Score: {score}", True, white)
    screen.blit(text, [0, 0])

def game_loop():
    game_over = False
    game_close = False

    snake = Snake()
    food = Food(snake.body)
    score = 0

    while not game_over:

        while game_close:
            screen.fill(black)
            message = font.render(f"Game Over! Score: {score} | Q-Quit", True, red)
            screen.blit(message, [screen_width // 6, screen_height // 3])
            display_score(score)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                    game_close = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    snake.change_direction([-1, 0])
                elif event.key == pygame.K_RIGHT:
                    snake.change_direction([1, 0])
                elif event.key == pygame.K_UP:
                    snake.change_direction([0, -1])
                elif event.key == pygame.K_DOWN:
                    snake.change_direction([0, 1])

        snake.move()

        if snake.get_head() == food.position:
            score += 1
            snake.grow()
            food.respawn(snake.body)

        if snake.check_collision():
            game_close = True

        screen.fill(black)
        draw_snake(snake)
        draw_food(food)
        display_score(score)
        pygame.display.update()

        clock.tick(15)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    game_loop()