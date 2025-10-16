import cv2
from cv2 import threshold
import numpy as np
from matplotlib import pyplot as plt


def sobel_edge_detection(image):
    img_blur = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)

    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)  # Combined X and Y Sobel Edge Detection

    cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    cv2.waitKey(0)


def canny_edge_detection(image, threshold_1, threshold_2):
    img_blur = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)

    edges = cv2.Canny(image=img_blur, threshold1=threshold_1, threshold2=threshold_2)

    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)


def template_match(image, template):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

    cv2.imwrite('outputs/res.png', image)
    cv2.imshow('res.png', image)
    cv2.waitKey(0)


def resize(image, scale_factor, up_or_down):


    resized_image = image.copy()

    for _ in range(scale_factor):
        if up_or_down == "up":
            resized_image = cv2.pyrUp(resized_image)
        elif up_or_down == "down":
            resized_image = cv2.pyrDown(resized_image)

    cv2.imshow("Resized (Pyramid)", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def main ():
    image = cv2.imread("lambo.png")
    template = cv2.imread('shapes_template.jpg', 0)


    threshold_1 = 50
    threshold_2 = 50

    # sobel_edge_detection(image)

    # canny_edge_detection(image, threshold_1, threshold_2)

    #template_match(image, template)

    resize(image, 2, "up")






if __name__ == "__main__":
    main()



