from sklearn.mixture import GaussianMixture
import pandas as pd
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

    def get_horizontal(self, img):
        horizontal = np.copy(img)

        # Specify size on horizontal axis
        cols = horizontal.shape[1]
        horizontal_size = cols // 10 #2 3 4 5
        # horizontal_size = cols // 15 #1

        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))

        # Apply morphology operations
        horizontal = cv.erode(horizontal, horizontalStructure)
        horizontal = cv.dilate(horizontal, horizontalStructure)
        return horizontal

    def get_high_ranked_edges(self, img, num_edge):
        # data = [0 for x in range(img.shape[0])]
        data = {}
        for x in range(img.shape[1]):
            for y in range(img.shape[0]):
                if img[y][x] == 255:
                    if y not in data:
                        data[y] = 0
                    data[y] += 1

        data = {k: v for k, v in sorted(data.items(), key=lambda item: item[1], reverse=True)}

        re_list = []
        i = 0
        for k, v in data.items():
            if i == num_edge:
                break
            re_list.append(k)
            i += 1

        return re_list

    def get_edges(self, img, num_edge):
        data = []
        for x in range(img.shape[1]):
            for y in range(img.shape[0]):
                if img[y][x] == 255:
                    data.append(y)

        # print(data)
        df = pd.DataFrame(data)

        # Fit GMM
        gmm = GaussianMixture(n_components=num_edge, random_state=42)
        gmm = gmm.fit(df)

        means = gmm.means_.tolist()
        means = [x[0] for x in means]
        means = sorted(means)
        return means

    def remove_out_objects(self, src, edges):
        img = np.copy(src)
        h, w = img.shape

        img_cleaned = np.zeros((h, w), np.uint8)
        img_cleaned = cv.resize(img_cleaned, (w, h))

        a = int(edges[0])
        b = int(edges[1])
        top = min(a, b) - 4
        bot = max(a, b) + 7
        gg = img[top:bot, 0:w]
        img_cleaned[top:bot, 0:w] = gg

        # self.show_my_image(img_cleaned)
        return img_cleaned

    def get_average_rgb(self, img, x1, y1, x2, y2):
        """
        This returns average rbg values from rectangle points
        """
        c_px = np.zeros([1, 3])
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # If there is no space between two edges
        if abs(y1-y2) <= 1:
            return None

        # cv.line(img, (0, y1), (130, y1), (0, 255, 0), 1)
        # cv.rectangle(img, (x1, y1), (x2, y2), (125, 125, 125), 1)

        i = 0
        for x in range(x1, x2):
            for y in range(y1, y2):
                px = img[y, x]
                # print(px)
                c_px += px
                i += 1

        c_px = c_px / i
        c_px = np.round(c_px)
        c_px = c_px.tolist()[0]
        return c_px

    def get_colors(self, img, edges):
        h, w, c = img.shape
        x_gab = 10
        y_gab = 1

        ee = edges[0] # Ending edge (EE)
        me = edges[1] # Mid edge (ME)
        se = edges[2] # Starting edge (SE)

        oc = self.get_average_rgb(img, x_gab, ee + y_gab, w - x_gab, me - y_gab) # Oil Color
        cc = self.get_average_rgb(img, x_gab, me + y_gab, w - x_gab, se - y_gab) # Creaming Color
        wc = self.get_average_rgb(img, x_gab, se + y_gab, w - x_gab, se + 20) # Water Color
        # self.show_my_image(img)
        return oc, cc, wc

    def list_distance(self, l1, l2):
        sum = 0
        for i in range(len(l1)):
            sum += abs(l1[i]-l2[i])

        return sum

    def get_oil_value_using_edges_and_colors(self, img, edges, oc, cc, wc):
        #     색을 통한 오일 분포 분류 규칙:
        # O가 오일 색[120, 161, 179]에 가까우면 오일로 규정
        # O가 크림 색[165, 190, 204]에 가까우면 크림으로 규정
        # C는 색에 상관없이 크림으로 규정

        sum_oil = self.list_distance([120, 161, 179], oc)
        sum_cream = self.list_distance([165, 190, 204], oc)

        ee = edges[0]  # Ending edge (EE)
        me = edges[1]  # Mid edge (ME)
        se = edges[2]  # Starting edge (SE)

        if sum_oil < sum_cream:
            return (me-ee)/(se-ee)
        else:
            return 0



    def get_monochrome_image(self, img):
        """
        This converts an input image to a grayscale image.
        """
        img_0 = np.copy(img)
        gray = cv.cvtColor(img_0, cv.COLOR_BGR2GRAY)
        monochrome_img = cv.Canny(gray, 10, 50)

        return monochrome_img

    def show_db_img(self):
        # Find width and height
        max_x, max_y = 0, 0
        max_w, max_h = 0, 0
        for img_info in self.db:
            x = img_info['X']
            y = img_info['Y']
            img = img_info['img']
            h, w = img.shape
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
            h, w = img.shape
            w += x
            h += y
            img_canvas[y:h, x:w, 1] = img
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
            str = f'{oil_color}'
            cv.putText(img_canvas, str, (x + 3, y + 55), cv.FONT_HERSHEY_TRIPLEX, 0.33, (255, 255, 255), 1)

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
            if x > 6:
                x = 1
                y += 1

            # if x == 1 and y == 2:
            # 2.1 Read an image
            img_file = join(img_path, file)
            img = cv.imread(img_file)

            # 2.2 Get the monochrome image
            img_mono = self.get_monochrome_image(img)

            # self.show_my_image(edge_img)

            # 2.3 Get the horizontal image from the mono image
            img_hori = self.get_horizontal(img_mono)

            # 2.4 Get edges from the horizontal image
            two_edges = self.get_high_ranked_edges(img_hori, 2)

            # 2.5 Get a cleaned image
            img_cleaned = self.remove_out_objects(img_hori, two_edges)

            # 2.6 Get edges from the horizontal image
            edges = self.get_edges(img_cleaned, 3)

            # 2.7 Get colors
            oc, cc, wc = self.get_colors(img, edges)
            # print(file, oc)

            # 2.7 Get colors
            ov = self.get_oil_value_using_edges_and_colors(img, edges, oc, cc, wc)
            # Show extracted horizontal lines
            # self.show_my_image(horizontal)

            # all_img = self.combine_image(horizontal, all_img)
            self.store_image(img_cleaned, x, y, os.path.splitext(file)[0], edges, ov, oc)

            cv.rectangle(img_cleaned, (20, 160), (110, 240), (0, 255, 0), 1)

            x += 1

        self.show_db_img()

            # 2.3 Get edges from the edge image
        #     a_px = 0
        #     # a_px = get_average_rgb(img, 20, 160, 110, 240)
        #     # print(a_px)
        #
        #
        #     # Show Images
        #     if x > 6:
        #         x = 1
        #         y += 1
        #
        #     sub = plt.subplot2grid((int(num_imgs / 4), 7), (y, x))
        #
        #     sub.imshow(edge_img)
        #     sub.set_title(f'{os.path.splitext(file)[0]} \n {a_px}:{1}')
        #     x += 1
        #
        # plt.subplots_adjust(bottom=.1, top=.99, hspace=1)
        # plt.show()


img_path = '../image_data/Triton X-100/'
ip = ImagePipeline()
ip.run_image_pipeline(img_path)
