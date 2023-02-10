import pygame
import numpy as np
# initialize pygame
pygame.init()

# set the window size
window_size = (540, 540)

# create the window
screen = pygame.display.set_mode(window_size)

# fill the screen with a solid color
screen.fill((255, 255, 255))

# create the grid
grid_size = 27
cell_size = window_size[0] // grid_size
for i in range(grid_size):
    for j in range(grid_size):
        pygame.draw.rect(screen, (0, 0, 0), (i * cell_size, j * cell_size, cell_size, cell_size), 1)

# update the screen
pygame.display.update()

# create a 2D array to store the pixel values
grid = [[0 for j in range(grid_size)] for i in range(grid_size)]

# track whether the mouse button is being pressed
mouse_down = False

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
            grid[x // cell_size][y // cell_size] = 1

    if mouse_down:
        x, y = pygame.mouse.get_pos()
        grid[x // cell_size][y // cell_size] = 1
        pygame.draw.rect(screen, (255, 0, 0), (x // cell_size * cell_size, y // cell_size * cell_size, cell_size, cell_size))
        pygame.display.update()
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
    W1 = np.loadtxt('npW1.csv', delimiter = ',')
    b1 = np.random.rand(10, 1) - 0.5 # so b1 exsist
    b1x = np.array(np.loadtxt('npb1.csv'))
    for i  in  range (10):
        b1[i] = b1x[i]
    W2 = np.loadtxt('npW2.csv', delimiter= ',') 
    b2 = np.random.rand(10, 1) - 0.5 # so b2 exsist
    b2x = np.array(np.loadtxt('npb2.csv'))
    for i  in  range (10):
        b2[i] = b2x[i]

    new_grid = [0 for x in range(0,784)]

    for xindex,x in enumerate(grid):
        for yindex,y in enumerate(grid[xindex]):
            
            new_grid[xindex*27+yindex] = y

    
    _, _, _, prediction = forward_prop(W1,b1,W2,b2,new_grid)
    print(prediction)
    print("Prediction: ", np.argmax(prediction,0))

output(grid)


# quit pygame
pygame.quit()