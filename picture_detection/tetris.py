# import the necessary packages
import argparse
import imutils
import cv2

from utils import show_image

filename = 'tetris_blocks.png'

image = cv2.imread(filename)
show_image(image, "Image")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show_image(gray, "Gray")

edged = cv2.Canny(gray, 30, 150)
# show_image("Edged", edged)

# threshold the image by setting all pixel values less than 225
# to 255 (white; foreground) and all pixel values >= 225 to 255
# (black; background), thereby segmenting the image
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
show_image(thresh, "Thresh")

# find contours (i.e., outlines) of the foreground objects in the
# thresholded image
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
output = image.copy()
# loop over the contours
for c in contours:
    # draw each contour on the output image with a 3px thick purple
    # outline, then display the output contours one at a time
    cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
    show_image(output, "Contours")

text = "I found {} objects!".format(len(contours))
cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (240, 0, 159), 2)
show_image(output, "Contours")

# we apply erosions to reduce the size of foreground objects
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
show_image(mask, "Eroded")

# similarly, dilations can increase the size of the ground objects
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
show_image(mask, "Dilated")

# a typical operation we may want to apply is to take our mask and
# apply a bitwise AND to our input image, keeping only the masked
# regions
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
show_image(output, "Output")
