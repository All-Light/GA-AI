import pygame
import numpy as np
# initialize pygame
pygame.init()

# This file starts a window that you can draw in. 
# When you release the mouse button a prediction will be outputted to the terminal
# To clear the screen press "Q"
# It is not known if it is entierly accurate of the model.
# Either the model is not perfect or this file is somehow converting something wrong.

# set the window size
screen_width, screen_height = 540, 540
window_size = (540, 540)

# create the window
screen = pygame.display.set_mode(window_size)
font = pygame.font.SysFont('Calibri', 124)

# Set the timer duration and start time
timer_duration = 0  # in milliseconds
start_time = pygame.time.get_ticks()

global text_surface
global display_text
text_surface = font.render('', True, (255, 0, 0))
display_text = True
# fill the screen with a solid color
screen.fill((0, 0, 0))

# create the grid
grid_size = 28
cell_size = window_size[0] // grid_size
for i in range(grid_size):
    for j in range(grid_size):
        pygame.draw.rect(screen, (255, 255, 255), (i * cell_size, j * cell_size, cell_size, cell_size), 1)

# update the screen
pygame.display.update()

# create a 2D array to store the pixel values
grid = [[0 for j in range(grid_size)] for i in range(grid_size)]

# track whether the mouse button is being pressed
mouse_down = False


def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    """Compute softmax values for each sets of scores in x."""
    exp = np.exp(Z - np.max(Z)) 
    return exp / exp.sum(axis=0)

def forward_prop(W1, b1, W2, b2, X):
    # this function will run the neural network with the current weights and biases (with X as input image)
    # A2 is the output layer
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def output(grid):
    W1 = np.loadtxt('./output/npW1_10000_0.4.csv', delimiter = ',')
    b1 = np.random.rand(10, 1) - 0.5 # so b1 exsist
    b1x = np.array(np.loadtxt('./output/npb1_10000_0.4.csv'))
    for i in range(10):
        b1[i] = b1x[i]
    W2 = np.loadtxt('./output/npW2_10000_0.4.csv', delimiter= ',') 
    b2 = np.random.rand(10, 1) - 0.5 # so b2 exsist
    b2x = np.array(np.loadtxt('./output/npb2_10000_0.4.csv'))
    for i in range(10):
        b2[i] = b2x[i]

    new_grid = np.zeros((784, 1))

    new_grid = np.array(grid).flatten().astype('float64')

    # correct format
    vect_X = new_grid[:,None]
    _, _, _, prediction = forward_prop(W1,b1,W2,b2,vect_X)
    pred =  np.argmax(prediction,0)
    
    print("Prediction: ", pred[0])
    global text_surface
    text_surface = font.render(str(pred[0]), True, (255, 0, 0))

# run the main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False
            x, y = pygame.mouse.get_pos()
            grid[y // cell_size][x // cell_size] = 1
            output(grid)
            start_time = pygame.time.get_ticks()
            timer_duration = 3000 
            display_text = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                grid = [[0 for j in range(grid_size)] for i in range(grid_size)]
                for x in range(grid_size):
                    for y in range(grid_size):
                        pygame.draw.rect(screen, (0, 0, 0), (x * cell_size, y * cell_size, cell_size, cell_size), 0)
                        pygame.draw.rect(screen, (255, 255, 255), (x * cell_size, y * cell_size, cell_size, cell_size), 1)

                display_text = False
                pygame.display.update()

    time_elapsed = pygame.time.get_ticks() - start_time
    if time_elapsed < timer_duration:
        # Blit the text surface to the screen
        if display_text:
            screen.blit(text_surface, (screen_width // 2 - text_surface.get_width() // 2, screen_height // 2 - text_surface.get_height() // 2))

    if mouse_down:
        x, y = pygame.mouse.get_pos()
        clicked_cell_x, clicked_cell_y = x // cell_size, y // cell_size
        for i in range(-1, 2):
            for j in range(-1, 2):
                adjacent_cell_x, adjacent_cell_y = clicked_cell_x + i, clicked_cell_y + j
                if 0 <= adjacent_cell_x < grid_size and 0 <= adjacent_cell_y < grid_size:
                    # semi-accurate colors
                    adjacent_cell_center_x = adjacent_cell_x * cell_size
                    adjacent_cell_center_y = adjacent_cell_y * cell_size
                    distance = 1 - (((x - adjacent_cell_center_x) ** 2 + (y - adjacent_cell_center_y) ** 2) ** 0.5) / (((2*cell_size) ** 2 + (2*cell_size) ** 2) ** 0.5)
                    grid[(y // cell_size) + i][(x // cell_size) + j] = distance
                    col = int(distance*255)
                    # white only
                    #grid[(y // cell_size) + i][(x // cell_size) + j] = 1
                    #col = 255
                    #
                    pygame.draw.rect(screen, (col, col, col), ((x // cell_size + i) * cell_size, (y // cell_size + j) * cell_size, cell_size, cell_size))

    pygame.display.flip()

# quit pygame
pygame.quit()