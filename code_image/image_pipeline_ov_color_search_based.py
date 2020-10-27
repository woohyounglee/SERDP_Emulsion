from os import listdir
from os.path import isfile, join
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


class ImagePipeline():
    def __init__(self):
        self.db = []

    def show_my_image(self, img, winname='Title'):
        cv.imshow(winname, img)
        cv.moveWindow(winname, 1000, 500)
        cv.waitKey(0)
        cv.destroyWindow(winname)

    def get_colors(self, img):
        h, w, c = img.shape

        c_px = np.zeros([1, 3])

        i = 0
        for x in range(0, w):
            for y in range(0, h):
                px = img[y, x]
                if px[0] == 0 and px[1] == 0 and px[2] == 0:
                    pass
                else:
                    # print(px)
                    c_px += px
                    i += 1

        c_px = c_px / i
        c_px = np.round(c_px)
        c_px = c_px.tolist()[0]
        return c_px

    def show_db_img(self):
        # Find width and height
        max_x, max_y = 0, 0
        max_w, max_h = 0, 0
        for img_info in self.db:
            x = img_info['X']
            y = img_info['Y']
            img = img_info['img']
            h, w, c = img.shape
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_w = max(max_w, w)
            max_h = max(max_h, h)

        # print(max_x, max_y)

        # Make a big canvas
        height = max_y*max_h+100
        width = max_x*max_w+100
        img_canvas = np.zeros((height, width, 3), np.uint8)
        h, w, c = img_canvas.shape
        img_canvas = cv.resize(img_canvas, (w, h))

        gab_x, gab_y = 5, 5
        for img_info in self.db:
            x = (img_info['X']-1)*max_w + gab_x
            y = (img_info['Y']-1)*max_h + gab_y
            img = img_info['img']
            edges = img_info['edges']
            title = img_info['title']
            oil_value = img_info['oil_value']
            oil_color = img_info['oil_color']
            h, w, c = img.shape
            w += x
            h += y
            img_canvas[y:h, x:w] = img
            cv.rectangle(img_canvas, (x, y), (w, h), (125, 125, 125), 1)

            # Draw edge lines
            if edges is not None:
                for edge in edges:
                    h_e = int(edge + y)
                    cv.line(img_canvas, (x, h_e), (w, h_e), (255, 255, 255), 1)

            # Draw title
            str = f'{title}'
            cv.putText(img_canvas, str, (x+3, y+20), cv.FONT_HERSHEY_TRIPLEX, 0.33, (255, 255, 255), 1)
            str = '{:.4f}'.format(oil_value)
            cv.putText(img_canvas, str, (x + 3, y + 35), cv.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)
            # str = f'{oil_color}'
            # cv.putText(img_canvas, str, (x + 3, y + 55), cv.FONT_HERSHEY_TRIPLEX, 0.33, (255, 255, 255), 1)

        cv.imshow('', img_canvas)
        cv.waitKey()

    def store_image(self, src, x, y, title='title', edges=None, oil_value=0, oil_color=None):
        img_0 = np.copy(src)
        img_info = {'X': x, 'Y': y, 'title': title, 'img': img_0, 'edges': edges, 'oil_value': oil_value, 'oil_color': oil_color}
        self.db.append(img_info)

    def run_image_pipeline(self, img_path):
        # 1. Extract image files from an image path
        img_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]

        x, y = 1, 1
        plt.figure(0)
        num_imgs = len(img_files)

        # 2. Perform the pipeline for each image
        for file in img_files:
            # print(f'====== Image: {file} =======')
            if x > 12:
                x = 1
                y += 1

            # 2.1 Read an image
            img_file = join(img_path, file)
            img = cv.imread(img_file)

            # 2.2 Get colors
            color = self.get_colors(img)

            # 2.3 Calculate OV
            ov = (255-(color[0]+color[1]+color[2])/3)/255

            # self.show_my_image(horizontal)

            print(f'{file}\t{ov}')

            self.store_image(img, x, y, os.path.splitext(file)[0], None, ov, None)

            x += 1

        self.show_db_img()


# img_paths = ['../image_data/Power Green/', '../image_data/PRC/', '../image_data/Blast-off/']
# img_paths = ['../image_data/Power Green/']
# img_paths = ['../image_data/SDS/']
# img_paths = ['../image_data/PRC/']
# img_paths = ['../image_data/Blast-off/']

# img_paths = ['../image_data/Calla/']
# img_paths = ['../image_data/BB/']
# img_paths = ['../image_data/AFFF/']
# img_paths = ['../image_data/Type 1/']
img_paths = ['../image_data/Test/']
# img_paths = ['../image_data/Solid surge/']


for path in img_paths:
    ip = ImagePipeline()
    ip.run_image_pipeline(path)
