import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def load_image_to_rgb(filename):
    img = cv2.imread(filename, 3)
    b, g, r = cv2.split(img)  # get b, g, r
    return cv2.merge([r, g, b])  # switch it to r, g, b

def save_image(path, img):
    r, g, b = cv2.split(img)
    cv2.imwrite(path, cv2.merge([b, g, r]));


def show_image(img):
    plt.imshow(img)
    plt.show()


def plot_3d_data(data_red, data_blue):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for d in data_red:
        ax.scatter(d[0], d[1], d[2], c='r', marker='o')
    for d in data_blue:
        ax.scatter(d[0], d[1], d[2], c='b', marker='^')

    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')

    plt.show()

def calc_image_color_mean(img):
    average = img.mean(axis=0).mean(axis=0)
    return average


def calc_image_dominant_color(img):
    # print(img.shape)
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    return dominant
