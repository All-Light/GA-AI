import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from PIL import Image

# this file will override test_image.png and draw to it with the mouse cursor
# hopefully this png image can then be used in the model and get a prediction the users drawn number

class Painter(object):
    def __init__(self, ax, img):
        self.showverts = True
        self.figure = plt.figure(1)
        self.button_pressed = False
        self.img = img
        self.brush_size = 10
        self.ax = ax
        self.color = 255

        canvas = self.figure.canvas
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.on_move)

    def button_press_callback(self, event):
        if(event.button == 1):
            self.button_pressed = True
            x = int(math.floor(event.xdata))
            y = int(math.floor(event.ydata))
            cv2.circle(self.img, (x, y), int(self.brush_size / 2), (self.color, self.color, self.color), -1)
            #update the image

    def button_release_callback(self, event):
        self.button_pressed = False
        self.ax.images.pop()
        self.ax.imshow(self.img, interpolation='nearest', alpha=0.6)
        plt.draw()
        cv2.imwrite('test_image.png', self.img)

    def on_move(self, event):
        if(self.button_pressed):
            x = int(math.floor(event.xdata))
            y = int(math.floor(event.ydata))
            cv2.circle(self.img, (x, y), int(self.brush_size / 2), (self.color, self.color, self.color), -1)
            #update the image

def draw_demo():
    global imgMain
    imgOver = np.zeros((400,600,3), np.uint8)
    img = Image.new('RGB', (400, 600), (255, 255, 255))
    img.save("test_image.png", "PNG")
    imgMain = mpimg.imread("test_image.png")


    ax = plt.subplot(111)
    ax.imshow(imgMain, interpolation='nearest', alpha=1)
    ax.imshow(imgOver, interpolation='nearest', alpha=0.6)

    pntr = Painter(ax, imgOver)
    plt.title('Click on the image to draw')
    plt.show()

if __name__ == '__main__':
    draw_demo()