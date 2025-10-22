import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt


def padding(image, border_width):
    reflect = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    plt.plot(1), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
    plt.show()


def crop (image, x_0, x_1, y_0, y_1):
    cropped_image = image[y_0:y_1, x_0:x_1]
    cv2.imshow("cropped", cropped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize (image, width, height):
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    plt.imshow(resized_image)
    plt.show()


def copy (image, emptyPictureArray):
    emptyPictureArray[:] = image[:]

    cv2.imshow('Source Image', image)
    cv2.imshow('Copied Image (on empty canvas)', emptyPictureArray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def grayscale (image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def hsv (image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV", hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def hue_shifted(image, emptyPictureArray, hue):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    shift = int(round(hue / 2.0))
    h = ((h.astype(np.int16) + shift) % 180).astype(np.uint8)

    hsv_shifted = cv2.merge([h, s, v])
    hue_image = cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)

    emptyPictureArray[:] = hue_image

    cv2.imshow("Hue changed", hue_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def smoothing (image):
    blur = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)

    cv2.imshow("Gaussian Smoothing", numpy.hstack((image, blur)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotation (image, rotation_angle):

    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)

    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main ():
    image = cv2.imread("lena-2.png")
    height, width, channels = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # invert colors
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)


    # padding(img_rgb, 100)
    # crop(image, 80, 382, 80, 382)
    # resize(img_rgb, 200, 200)
    # copy(image, emptyPictureArray)
    # grayscale(image)
    # hsv(image)
    hue_shifted(image,emptyPictureArray, 50)
    # smoothing(image)
    # rotation(image, 90)


if __name__ == "__main__":
    main()