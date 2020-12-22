from document_scanner.transform import four_point_transform
from imutils import contours
import numpy as np
import imutils
import cv2
from utils import show_image

filename = 'image.png'

# define the answer key which maps the question number
# to the correct answer
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
image = cv2.imread(filename)
show_image(image, "Test")
