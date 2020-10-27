from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2


# Function to show array of images (intermediate results)
def show_images(images):
	for i, img in enumerate(images):
		cv2.imshow("image_" + str(i), img)
		# cv2.setWindowTitle("image_" + str(i), img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def object_height(img, img_saved_path):
	# Converting the image to grayscale.
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# blur = cv2.GaussianBlur(gray, (9, 9), 0)

	# Using the Canny filter to get contours
	monochrome_img = cv2.Canny(gray, 10, 50)
	images = []
	# images.append(cv2.Canny(gray, 10, 30))
	images.append(monochrome_img)  # Now best parameters
	# images.append(cv2.Canny(gray, 60, 120))

	# Saving the image
	cv2.imwrite(img_saved_path, monochrome_img)

	show_images(images)


def get_average_rgb(img, x1, y1, x2, y2):
	c_px = np.zeros([1, 3])
	x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

	# If there is no space between two edges
	if abs(y1 - y2) <= 1:
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
	return c_px


# img_path = "../image_data/Triton X-100/TX_0_NaCl_4C.png"
# img_path = "../image_data/Triton X-100/TX_4_4C.png"
# img_path = "../image_data/Triton X-100/TX_4_NaCl_4C_MOD.png"
# img_path = "../image_data/Triton X-100/TX_10_NaCl_4C.png"
# img_path = "../image_data/Triton X-100/TX_0_NaCl_SS_4C_MOD.png"
# img_path = "../image_data/Triton X-100/TX_10_NaCl_SS_4C_MOD.png"
img_path = "../image_data/Triton X-100/TX_0_SS_4C_MOD.png"
img_saved_path = "../image_saved_data/cream_color.png" # [[192. 215. 230.]]
# img_saved_path = "../image_saved_data/oil_color.png" # [[143. 181. 199.]]

# Read image and preprocess
# image = cv2.imread(img_path)

# h, edged = object_height(image, img_saved_path)

c = get_average_rgb(cv2.imread(img_saved_path), 10, 10, 120, 120)

print(c)
