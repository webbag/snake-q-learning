import numpy as np
import random
import pygame
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Stałe
BOARD_SIZE = 30
PIXEL_SIZE = 30
WINDOW_SIZE = BOARD_SIZE * PIXEL_SIZE
FPS = 1000
EPIZODE = 1000

INITIAL_SNAKE_POSITION = [(15, 15), (15, 16), (15, 27), (15, 18), (15, 19), (15, 20)]  # Początkowa pozycja węża
INITIAL_SNAKE_DIRECTION = (0, 1)  # Początkowy kierunek węża (prawo)

# Kolory
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
LIGHT_GREEN = (144, 238, 144)
DARK_RED = (139, 0, 0)

Q_TABLE_FILENAME = "q_table_v9.pkl"
Q_TABLE_FILENAME_VISION = "q_table_v9.png"


# Inicjalizacja Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Snake Game with Q-learning")
clock = pygame.time.Clock()


def load_q_table(filename=Q_TABLE_FILENAME):
    with open(filename, "rb") as f:
        q_table = pickle.load(f)
    return q_table

def visualize_q_table(q_table):
    # Przekształcenie tablicy Q na odpowiedni format
    q_table_max = np.max(q_table, axis=2)  # Maksymalna wartość Q dla każdej pozycji

    plt.imshow(q_table_max, cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(label='Wartość Q')
    plt.title('Tablica Q')
    plt.xlabel('Kolumna')
    plt.ylabel('Rząd')
    plt.savefig(Q_TABLE_FILENAME_VISION)  # Zapisz obraz do pliku zamiast wyświetlać go
    print(f"Wizualizacja tablicy Q została zapisana jako '{Q_TABLE_FILENAME_VISION}'")



# Funkcja do wyświetlania wiadomości
def display_message(screen, message, position=(10, 10)):
    font = pygame.font.Font(None, 24)
    text = font.render(message, True, BLACK)
    screen.blit(text, position)

# Funkcja do wykonania akcji przez węża
def perform_action(board, snake_position, snake_direction, action, apple_position):
    reward_message = ""

    # Zmiana kierunku węża na podstawie akcji
    if action == 0:  # góra
        snake_direction = (-1, 0)
    elif action == 1:  # dół
        snake_direction = (1, 0)
    elif action == 2:  # lewo
        snake_direction = (0, -1)
    elif action == 3:  # prawo
        snake_direction = (0, 1)

    # Nowa pozycja głowy węża
    new_head = (snake_position[0][0] + snake_direction[0], snake_position[0][1] + snake_direction[1])

    # Sprawdzenie, czy nowa głowa jest w granicach planszy
    if not (0 <= new_head[0] < BOARD_SIZE and 0 <= new_head[1] < BOARD_SIZE):
        reward = -10
        reward_message = "Kara: Wąż wyszedł poza planszę!"
        done = True
        return board, snake_position, snake_direction, reward, done, reward_message, apple_position

    # Sprawdzenie czy wąż zjadł jabłko
    if new_head == apple_position:
        reward = 10
        reward_message = "Nagroda: Wąż zjadł jabłko!"
        done = False
        apple_position = generate_new_apple(board, snake_position)
    else:
        # Przesunięcie ciała węża
        snake_position.insert(0, new_head)
        if board[new_head] == 1:  # kolizja z samym sobą
            reward = -10
            reward_message = "Kara: Wąż uderzył w siebie!"
            done = True
        else:
            board[new_head] = 1
            tail = snake_position.pop()
            board[tail] = 0
            reward = -1
            done = False

    return board, snake_position, snake_direction, reward, done, reward_message, apple_position


# Funkcja do generowania nowego jabłka
def generate_new_apple(board, snake_position):
    empty_spaces = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r, c] == 0]
    return random.choice(empty_spaces)

# Parametry uczenia
alpha = 0.1  # współczynnik uczenia
gamma = 0.9  # współczynnik dyskontowania
epsilon = 0.1  # współczynnik eksploracji

# Spróbuj załadować Q-table z pliku, jeśli istnieje, w przeciwnym razie zainicjalizuj nową
try:
    with open(Q_TABLE_FILENAME, "rb") as f:
        q_table = pickle.load(f)
except FileNotFoundError:
    q_table = np.zeros((BOARD_SIZE, BOARD_SIZE, 4))  # 4 możliwe akcje: góra, dół, lewo, prawo

max_moves = 0

def initialize_board_and_snake():
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    snake_position = INITIAL_SNAKE_POSITION.copy()
    snake_direction = INITIAL_SNAKE_DIRECTION
    for segment in snake_position:
        board[segment] = 1
    apple_position = generate_new_apple(board, snake_position)
    board[apple_position] = 2
    return board, snake_position, snake_direction, apple_position

def choose_action(current_state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # eksploracja
    else:
        return np.argmax(q_table[current_state])  # eksploatacja

def update_q_table(current_state, action, reward, new_state, alpha, gamma):
    q_table[current_state][action] = (1 - alpha) * q_table[current_state][action] + alpha * (
            reward + gamma * np.max(q_table[new_state]))


def visualize(screen, board, snake_position, apple_position, reward_message, episode, move_count, max_moves):

    screen.fill(WHITE)
    pygame.draw.rect(screen, RED,
                     pygame.Rect(apple_position[1] * PIXEL_SIZE, apple_position[0] * PIXEL_SIZE, PIXEL_SIZE,
                                 PIXEL_SIZE))
    for segment in snake_position:
        pygame.draw.rect(screen, GREEN,
                         pygame.Rect(segment[1] * PIXEL_SIZE, segment[0] * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

    # Kolorowanie planszy w zależności od decyzji
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == -10:  # Błędna decyzja
                color = DARK_RED
            elif board[r, c] == 10:  # Dobra decyzja
                color = LIGHT_GREEN
            else:
                continue
            pygame.draw.rect(screen, color, pygame.Rect(c * PIXEL_SIZE, r * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE))

    display_message(screen, reward_message)
    display_message(screen, f"Epoka: {episode + 1}", position=(10, 40))
    display_message(screen, f"Liczba ruchów: {move_count}", position=(10, 70))
    display_message(screen, f"Największa liczba ruchów: {max_moves}", position=(10, 100))
    pygame.display.flip()


def handle_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

for episode in range(EPIZODE):
    board, snake_position, snake_direction, apple_position = initialize_board_and_snake()
    done = False
    move_count = 0  # Licznik ruchów
    while not done:
        move_count += 1
        current_state = snake_position[0]  # aktualny stan to pozycja głowy węża
        action = choose_action(current_state, epsilon)
        new_board, new_snake_position, new_snake_direction, reward, done, reward_message, apple_position = perform_action(
            board, snake_position, snake_direction, action, apple_position)
        new_state = new_snake_position[0]  # nowy stan to nowa pozycja głowy węża
        update_q_table(current_state, action, reward, new_state, alpha, gamma)
        board = new_board
        snake_position = new_snake_position
        snake_direction = new_snake_direction
        visualize(screen, board, snake_position, apple_position, reward_message, episode, move_count, max_moves)
        clock.tick(FPS)
        handle_pygame_events()
    print(f"Epizod {episode + 1} zakończony. Liczba ruchów: {move_count}")
    if move_count > max_moves:
        max_moves = move_count

def save_q_table(q_table, filename=Q_TABLE_FILENAME):
    with open(filename, "wb") as f:
        pickle.dump(q_table, f)

# Po zakończeniu wszystkich epizodów
display_message(screen, f"Największa liczba ruchów: {move_count}", position=(10, 70))

save_q_table(q_table)

pd.set_option('display.max_rows', None)
q_table_df = pd.DataFrame(q_table.reshape(-1, 4), columns=['Góra', 'Dół', 'Lewo', 'Prawo'])

print("Wyuczona Q-table:")
print(q_table_df)

print(f"Największa liczba ruchów w epizodzie: {max_moves}")

# Załaduj tablicę Q i wyświetl ją
q_table = load_q_table()
visualize_q_table(q_table)

# Zamknięcie Pygame po zakończeniu treningu
pygame.quit()
