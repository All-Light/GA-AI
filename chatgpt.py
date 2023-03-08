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
window_size = (540, 540)

# create the window
screen = pygame.display.set_mode(window_size)

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
    #exp = np.exp(Z - np.max(Z)) 
    return 1 #exp / exp.sum(axis=0)

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

    new_grid = np.array(grid).flatten()
    #new_grid.reshape((28,28))
    #for yindex,y in enumerate(grid):
    #    for xindex,x in enumerate(y):
    #        new_grid[xindex*28+yindex] = x
    print(new_grid)
    
    _, _, _, prediction = forward_prop(W1,b1,W2,b2,new_grid)
    indexes =  np.argmax(prediction,0)
    #print(prediction)
    #print(indexes)
    max_val = -1
    max_index = -1
    sums = 0
    for i in range(len(indexes)):
        calc_val = prediction[i][indexes[i]]
        print(calc_val)
        sums+=calc_val
        if(calc_val> max_val): 
            max_val = calc_val
            max_index = i
            

    #print("Prediction: ", np.argmax(prediction,0))
    print("Prediction: ", max_index)
    print("Confidence: ", round((max_val/sums)*100,2), "%")

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
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                grid = [[0 for j in range(grid_size)] for i in range(grid_size)]
                for x in range(grid_size):
                    for y in range(grid_size):
                        pygame.draw.rect(screen, (0, 0, 0), (x * cell_size, y * cell_size, cell_size, cell_size), 0)
                        pygame.draw.rect(screen, (255, 255, 255), (x * cell_size, y * cell_size, cell_size, cell_size), 1)

                pygame.display.update()

    if mouse_down:
        x, y = pygame.mouse.get_pos()
        for i in range(-1, 1):
            for j in range(-1, 1):
                grid[(y // cell_size) + i][(x // cell_size) + j] = 255
                pygame.draw.rect(screen, (255, 255, 255), ((x // cell_size + i) * cell_size, (y // cell_size + j) * cell_size, cell_size, cell_size))
        pygame.display.update()

# quit pygame
pygame.quit()