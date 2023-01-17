#
# Verify Reading Dataset via MnistDataloader class
#
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from datareader import MnistDataloader

#
# Set file paths based on added MNIST Datasets
#
training_images_filepath = './dataset/training/train-images-idx3-ubyte'
training_labels_filepath = './dataset/training/train-labels-idx1-ubyte'
test_images_filepath = './dataset/testing/t10k-images-idx3-ubyte'
test_labels_filepath = './dataset/testing/t10k-labels-idx1-ubyte'

#
# TODO: Saveable rotated images in custom datafile
#

def show_images(images, title_texts):
    cols = 3
    rows = int(len(images)/cols) + 2
    plt.figure(figsize=(15,7))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 12);        
        index += 1
    plt.show()

def rotate_image(image):
    # create an array of only zeroes with the correct size
    rotated_image = np.zeros((len(image), len(image[0]))) 

    # get a random angle to rotate by, [- PI/4, PI/4]
    # random.random() returns a decimal digit between [0, 1]
    angle = (random.random()-0.5) * math.pi/2

    for y,y_val in enumerate(image):
        for x, x_val in enumerate(image[y]):
            # We get a specific pixel (x,y) if this has the value 0 there is no point in moving it.
            if (x_val == 0): continue
            
            rotation_matrix = [[math.cos(angle),-math.sin(angle)], [math.sin(angle), math.cos(angle)]]
            
            # we want to rotate it relative to center point
            dx = x - round(len(image[0])/2)
            dy = y - round(len(image)/2)

            relative_x = rotation_matrix[0][0]*dx + rotation_matrix[0][1]*dy
            relative_y = rotation_matrix[1][0]*dx + rotation_matrix[1][1]*dy
            rounded_relative_x = round(relative_x)
            rounded_relative_y = round(relative_y)

            new_x = rounded_relative_x + round(len(image[0])/2)-1
            new_y = rounded_relative_y + round(len(image)/2)-1
            rotated_image[new_y][new_x] = x_val

    # Preventing random black pixels in the middle of the digit.
    # if a pixel has at least 3 neighbors that are above 0 in light we can also set this pixel to the average.
    for y,y_val in enumerate(rotated_image):
        for x, x_val in enumerate(rotated_image[y]):
            if x == 0 or x == 27 or y == 0 or y == 27: continue
            if rotated_image[y][x] == 0:
                above = rotated_image[y-1][x]
                below = rotated_image[y+1][x]
                left = rotated_image[y][x-1]
                right = rotated_image[y][x+1]
                if sum([above > 0, below > 0, left > 0, right > 0]) >= 3:
                    rotated_image[y][x] = round((above+below+left+right)/4)

    return rotated_image

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

# image data, label data.
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

#images (x_train or x_test) is an array of all image data.
# the image data is in the form of 
# [
# y1 {x1,x2,x3...x28}
# y2 {x1,x2,x3...x28}
# y3 {x1,x2,x3...x28}
# ...
# y28 {x1,x2,x3...x28}
# ]
# where "[]" is one large array, and "{}" is a subarray. x1-x28 is of type int 0-255.



images_2_show = []
titles_2_show = []
'''
for i in range(0, 10):
    r = random.randint(1, 60000)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_2_show.append(x_test[r])        
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    
'''
for i in range(0, 5):
    #r = 10277#random.randint(1, 60000)
    r = random.randint(1,60000)
    # rotate the image of index r.
    rotated_example = rotate_image(x_train[r])

    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

    images_2_show.append(rotated_example)
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

show_images(images_2_show, titles_2_show)