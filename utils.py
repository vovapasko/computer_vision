import cv2


def show_image(image, title='Image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)


def find_edges(image):
    # convert image to grayscale, blur it
    # slightly, then find edges
    gray = gray_image(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    return edged


def gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
