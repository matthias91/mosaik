import numpy as np
import cv2
from matplotlib import pyplot as plt

def load_image_to_rgb(filename):
    img = cv2.imread(filename, 3)
    b, g, r = cv2.split(img)  # get b, g, r
    return cv2.merge([r, g, b])  # switch it to r, g, b

def save_image(img, path):
    #cvtColor(image, gray_image, CV_BGR2GRAY);
    cv2.imwrite(path, img);


def show_image(img):
    plt.imshow(img)
    plt.show()