import os
from functools import partial

import numpy as np
import cv2
from matplotlib import pyplot as plt
from multiprocessing import Pool
import argparse
from util import *

def create_mosaic(args):


    final_img = load_image_to_rgb(args.big_image_path)
    height, width = final_img.shape[:2]
    rows = int(height / args.mosaic_size)
    columns = int(width / args.mosaic_size)

    # Crop image TODO center crop
    final_img = final_img[0: rows * args.mosaic_size, 0:columns * args.mosaic_size]

    # create small mosaics
    mosaics = []
    # TODO multiprocessing
    #with Pool(4) as p:
        #mosaics.append(p.map(partial(image_to_mosaic, size=100), get_all_iamge_paths("/home/test/Desktop/")))

    for path in get_all_iamge_paths(args.image_pool):
        img = image_to_mosaic(path, args.mosaic_size)
        average_color = calc_image_color_mean(img)
        # show_image(img)
        mosaics.append((img, average_color))


    # Create mosaic
    for h in range(rows):
        for w in range(columns):
            #print("area", h * args.mosaic_size, h * args.mosaic_size + args.mosaic_size,
                   # w * args.mosaic_size, w * args.mosaic_size + args.mosaic_size)
            current_area = final_img[h * args.mosaic_size: h * args.mosaic_size + args.mosaic_size,
                    w * args.mosaic_size: w * args.mosaic_size + args.mosaic_size]
            color = calc_image_color_mean(current_area)
            found_mosaik = find_nearest_mosaic(mosaics, color)

            final_img[h * args.mosaic_size: h * args.mosaic_size + args.mosaic_size,
            w * args.mosaic_size: w * args.mosaic_size + args.mosaic_size] = found_mosaik

    #save_image("/home/test/Desktop/r.jpg", final_img)
    show_image(final_img)

def find_nearest_mosaic(mosaics, color):
    color_mean = np.mean(color)

    # TODO
    #result = mosaics[0][0]
    result = None
    min_diff = 255
    for m in mosaics:
        mosaics_color_mean = np.mean(m[1])
        diff = abs(mosaics_color_mean - color_mean)
        if diff < min_diff:
            min_diff = diff
            result = m[0]
    return result


def get_all_iamge_paths(directory):
    if not os.path.isdir(directory):
        raise AttributeError("Directory does not exist")

    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                yield os.path.join(root, name)


def image_to_mosaic(image_path, size):
    img = load_image_to_rgb(image_path)
    height, width = img.shape[:2]
    if width >= height:
        crop_img = img[0:height, int(((width - height) / 2)):int(((width - height) / 2)) + height]
    else:
        crop_img = img[int(((height- width) / 2)):int(((height - width) / 2)) + width, 0 :width]
    resized_image = cv2.resize(crop_img, (size, size))
    # show_image(resized_image)
    return resized_image


def calc_image_color_mean(img):
    return calc_image_color_mean_impl2(img)


def calc_image_color_mean_impl1(img):
    average = img.mean(axis=0).mean(axis=0)
    return average


def calc_image_color_mean_impl2(img):
    #print(img.shape)
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    return dominant


def main():
    parser = argparse.ArgumentParser(description='Create a mosaic from given images')
    parser.add_argument('big_image_path', help='the image path which should be drawn with mosaics')
    parser.add_argument('image_pool', help='directory with images used as single mosaics')
    parser.add_argument('mosaic_size', type=int,  help='the size in pixel of a single mosaic')

    args = parser.parse_args()

    create_mosaic(args)

    print("Finished")


if __name__ == '__main__':
    main()