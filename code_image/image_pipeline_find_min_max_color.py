from os import listdir
from os.path import isfile, join
import os
import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt
import winsound


class ImagePipeline():
    def __init__(self):
        self.df_excel = pd.read_excel('../data/20200203_UCF_Env_Data_DD_updated_for_paper_9-1-2020.xlsx', 'Image analysis')
        self.df_output = pd.DataFrame(columns=['name', 'num', 'ns1', 'ns2', 'temperature', 'ov'])

        self.img_excel_names = {
            '6%': 'AFFF',
            'B&B': 'B&B',
            'BO': 'Blast',
            'Calla': 'Calla',
            'PG': 'Powergreen',
            'PRC': 'PRC',
            'SDS': 'SDS',
            'SS': 'Surge',
            'TX': 'Triton-X-100',
            'T1': 'Type 1'
        }

    def reset(self):
        self.db = []

    def save_output(self):
        self.df_output.to_excel('../output/[OV_Results]image_processing.xlsx')

    def show_my_image(self, img, winname='Title'):
        cv.imshow(winname, img)
        cv.moveWindow(winname, 1000, 500)
        cv.waitKey(0)
        cv.destroyWindow(winname)

    def get_start_and_end_colors(self, name, temperature):
        colors = self.df_excel[self.df_excel['Surfactant name'] == name]
        colors = colors[colors['Temperature (°C)'] == float(temperature)]
        s = (colors['R1'].iloc[0] + colors['G1'].iloc[0] + colors['B1'].iloc[0])/3
        e = (colors['R2'].iloc[0] + colors['G2'].iloc[0] + colors['B2'].iloc[0])/3
        return s, e

    def get_image_color(self, img):
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
        return (c_px[0] + c_px[1] + c_px[2])/3

    def get_min_max_color(self, img):
        h, w, c = img.shape

        min_avg_px, max_avg_px = 1000, -1000
        min_px, max_px = None, None

        for x in range(0, w):
            for y in range(0, h):
                px = img[y, x]
                if px[0] == 0 and px[1] == 0 and px[2] == 0:
                    pass
                else:
                    # print(px)
                    avg_px = (float(px[0]) + float(px[1]) + float(px[2])) / 3
                    if avg_px < min_avg_px:
                        min_avg_px = avg_px
                        min_px = px
                    if min_avg_px == 0:
                        print('1')

                    if avg_px > max_avg_px:
                        max_avg_px = avg_px
                        max_px = px

        return min_px, max_px

    def show_db_img(self, name):
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

        # cv.imshow('', img_canvas)
        # cv.waitKey()

        # Saving the image
        cv.imwrite(f'../output_image/{name}.png', img_canvas)

    def store_image(self, src, x, y, title='title', edges=None, oil_value=0, oil_color=None):
        img_0 = np.copy(src)
        img_info = {'X': x, 'Y': y, 'title': title, 'img': img_0, 'edges': edges, 'oil_value': oil_value, 'oil_color': oil_color}
        self.db.append(img_info)

    def run_image_pipeline(self, img_path):
        # 1. Reset all information
        self.reset()

        # 2. Extract image files from an image path
        img_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]

        x, y = 1, 1
        plt.figure(0)

        # 3. Perform the pipeline for each image
        for file in img_files:
            # print(f'====== Image: {file} =======')
            if x > 12:
                x = 1
                y += 1

            # 3.1 Get file information
            temperature, name, num, ns1, ns2 = '', '', '', '', ''
            file_info = file
            file_info = file_info.replace('.png', '')
            file_info = file_info.split('_')
            name = self.img_excel_names[file_info[0]]
            num = file_info[1]
            if len(file_info) == 3:
                temperature = file_info[2].replace('C', '')
            elif len(file_info) == 4:
                ns1 = file_info[2]
                temperature = file_info[3].replace('C', '')
            elif len(file_info) == 5:
                ns1 = file_info[2]
                ns2 = file_info[3]
                temperature = file_info[4].replace('C', '')

            # 3.2 Get start and end color information
            # s_color, e_color = self.get_start_and_end_colors(name, temperature)

            # 3.3 Read an image
            img_file = join(img_path, file)
            img = cv.imread(img_file)

            # 3.4 Get colors
            c_min, c_max = self.get_min_max_color(img)

            # 3.5 Calculate OV
            ov = 1

            if ov < 0:
                ov = 0

            if ov > 1:
                ov = 1

            # self.show_my_image(horizontal)

            print(f'{file}\t{ov}')

            # 3.6 Store image data
            self.store_image(img, x, y, os.path.splitext(file)[0], None, ov, None)

            x += 1

            # 3.7 Store image info with ov
            self.df_output = self.df_output.append({'name': name, 'num': num, 'ns1': ns1, 'ns2': ns2, 'temperature': temperature, 'ov': ov}, ignore_index=True)

        self.show_db_img(name)


img_paths = ['../image_data/Type 1/']
# img_paths = ['../image_data/AFFF/',
#              '../image_data/BB/',
#              '../image_data/Blast-off/',
#              '../image_data/Calla/',
#              '../image_data/Power Green/',
#              '../image_data/PRC/',
#              '../image_data/SDS/',
#              '../image_data/Solid surge/',
#              '../image_data/Triton X-100/',
#              '../image_data/Type 1/']


ip = ImagePipeline()
for path in img_paths:
    ip.run_image_pipeline(path)

ip.save_output()
winsound.Beep(1000, 440)