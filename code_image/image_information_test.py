from os import listdir
from os.path import isfile, join
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def get_object_height(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 0)
    edged = cv.Canny(blur, 50, 100)
    edged = cv.dilate(edged, None, iterations=1)
    edged = cv.erode(edged, None, iterations=1)

    s, e = 0, 0
    for y in range(50, 150):
        c = edged[y, 70]
        if c == 255 and s == 0:
            s = y
        elif c == 255 and e == 0:
            e = y

    h = abs(s - e)
    return h


def get_average_rgb(img, x1, y1, x2, y2):
    t_px = np.zeros([1, 3])
    i = 0
    for x in range(x1, x2):
        for y in range(y1, y2):
            px = img[y, x]
            # print(px)
            t_px += px
            i += 1

    t_px = t_px / i
    t_px = np.round(t_px)
    return t_px


def get_edge_image(img):
    # Converting the image to grayscale.
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edge_img= cv.Canny(gray, 10, 50)
    return edge_img


def get_image_info(img_path):
    img_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    x, y = 1, 1
    plt.figure(0)

    num = len(img_files)

    for file in img_files:
        img_file = join(img_path, file)
        img = cv.imread(img_file)
        print(f'====== Image: {file} =======')
        # print(img.shape)
        # print(img.size)
        # print(img.dtype)

        # 2. Get the Average RGB
        a_px = get_average_rgb(img, 20, 160, 110, 240)
        print(a_px)

        # 3. Get a target object height
        height = get_object_height(img)
        cv.rectangle(img, (20, 160), (110, 240), (0, 255, 0), 1)

        # Show Images
        if x > 6:
            x = 1
            y += 1

        sub = plt.subplot2grid((int(num / 4), 7), (y, x))

        sub.imshow(img)
        sub.set_title(f'{os.path.splitext(file)[0]} \n {a_px}:{height}')
        x += 1

    plt.subplots_adjust(bottom=.1, top=.99, hspace=1)
    plt.show()


def get_edge_image_info(img_path):
    img_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    x, y = 1, 1
    plt.figure(0)

    num = len(img_files)

    for file in img_files:
        img_file = join(img_path, file)
        img = cv.imread(img_file)
        print(f'====== Image: {file} =======')
        # print(img.shape)
        # print(img.size)
        # print(img.dtype)

        # 1. Get the edge image
        edge_img = get_edge_image(img)

        # cv.imshow("image_" , edge_img)

        # 2. Get the Average RGB
        a_px = 0
        # a_px = get_average_rgb(img, 20, 160, 110, 240)
        # print(a_px)

        # 3. Get a target object height
        height = 0
        # height = get_object_height(img)
        # cv.rectangle(img, (20, 160), (110, 240), (0, 255, 0), 1)

        # Show Images
        if x > 6:
            x = 1
            y += 1

        sub = plt.subplot2grid((int(num / 4), 7), (y, x))

        sub.imshow(edge_img)
        sub.set_title(f'{os.path.splitext(file)[0]} \n {a_px}:{height}')
        x += 1

    plt.subplots_adjust(bottom=.1, top=.99, hspace=1)
    plt.show()


get_image_info('../image_data/Triton X-100/')
# get_edge_image_info('../image_data/Triton X-100/')
