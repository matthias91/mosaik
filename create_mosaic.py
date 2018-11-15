import os
import sys
from functools import partial

import numpy as np
import cv2
from matplotlib import pyplot as plt
from multiprocessing import Pool
import argparse
from util import *


def create_mosaic(args):

    final_img = load_image_to_rgb(args.input_image_path)
    height, width = final_img.shape[:2]
    rows = int(height / args.mosaic_size)
    columns = int(width / args.mosaic_size)

    # Crop image TODO center crop
    final_img = final_img[0: rows * args.mosaic_size, 0:columns * args.mosaic_size]

    print("Start loading the image pool")

    # create small mosaics
    mosaics = []
    # TODO multiprocessing
    # for path in get_all_iamge_paths(args.image_pool):
    #   img = image_to_mosaic(path, args.mosaic_size)
    with Pool(args.threads) as p:
        for img in p.imap(partial(image_to_mosaic, size=args.mosaic_size), get_all_iamge_paths(args.image_pool)):
            c = calc_image_dominant_color(img)
            mosaics.append((img, c))

    print("Loaded image pool")

    # Create mosaic
    for h in range(rows):
        sys.stdout.write("\r%i/%i rows processed" % (h, rows))
        for w in range(columns):
            current_area = final_img[h * args.mosaic_size: h * args.mosaic_size + args.mosaic_size,
                           w * args.mosaic_size: w * args.mosaic_size + args.mosaic_size]
            found_mosaik = find_best_matching_mosaic(mosaics, current_area)

            final_img[h * args.mosaic_size: h * args.mosaic_size + args.mosaic_size,
            w * args.mosaic_size: w * args.mosaic_size + args.mosaic_size] = found_mosaik

    #plot_3d_data(data_pool, data_img)
    show_image(final_img)
    save_image(args.output_image_path, final_img)


def find_best_matching_mosaic(mosaics, current_area):
    result = None
    min = 3*256
    c = calc_image_dominant_color(current_area)
    for m in mosaics:
        x = np.sum(abs(m[1] - c))
        if x < min:
            min = x
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



def main():
    parser = argparse.ArgumentParser(description='Create a mosaic from given images')
    parser.add_argument('input_image_path', help='the image path which should be drawn with mosaics')
    parser.add_argument('output_image_path', help='the path of the output image')
    parser.add_argument('image_pool', help='directory with images used as single mosaics')
    parser.add_argument('mosaic_size', type=int,  help='the size in pixel of a single mosaic')
    parser.add_argument('--threads', '-t', default=4, type=int, help='Number of threads')

    args = parser.parse_args()

    create_mosaic(args)

    print("Finished")


if __name__ == '__main__':
    main()