import cv2


def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
