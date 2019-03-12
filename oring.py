import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
import queue
import time


# histogram of image
def hist(img):

    y = np.zeros(256)
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            y[img[i, j]] += 1

    return y


# number of pixels in image
def pixelCount(h):
    count = 0
    for i in range(1, len(h)):
        if h[i] > 0:
            count += h[i]
    return count


def weight(x, y):
    w = 0
    for i in range(x, y):
        w += h[i]
    return w


def mean(x, y):
    m = 0
    w = weight(x, y)
    for i in range(x, y):
        m += h[i] * i

    return m / w


def variance(x, y):
    v = 0
    m = mean(x, y)
    w = weight(x, y)
    for i in range(x, y):
        v += ((i - m) ** 2) * h[i]
    v /= w
    return v


def threshold(h):
    count = pixelCount(h)
    for i in range(1, len(h)):
        # background variables
        vb = variance(0, i)
        wb = weight(0, i)/float(count)
        mb = mean(0, i)

        # foreground variables
        vf = variance(i, len(h))
        wf = weight(i, len(h))/float(count)
        mf = mean(i, len(h))

        #within class variance
        wcv = (wb * vb) + (wf * vf)

        if not math.isnan(wcv):
            threshold_values[i] = wcv


def getThreshold():
    minWcv = min(threshold_values.values())
    optimal_threshold = [k for k, v in threshold_values.items() if v == minWcv]
    print('optimal threshold', optimal_threshold[0])
    return optimal_threshold[0]


def thresholdedImage(img, threshold):
    row, col = img.shape
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            if img[i, j] > threshold:
                img[i, j] = 0
            else:
                img[i, j] = 255
    return img


def dilation(img):
    row, col = img.shape
    imgCopy = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            if img[i, j] == 255:
                imgCopy[i][j] = 255
                if i > 0:
                    imgCopy[i-1][j] = 255
                if j > 0:
                    imgCopy[i][j-1] = 255
                if i + 1 < len(img):
                    imgCopy[i+1][j] = 255
                if j + 1 < len(img[i]):
                    imgCopy[i][j+1] = 255

    return imgCopy


def erosion(img):
    row, col = img.shape
    imgCopy = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            if img[i, j] == 0:
                imgCopy[i][j] = 0
                if i > 0:
                    imgCopy[i-1][j] = 0
                if j > 0:
                    imgCopy[i][j-1] = 0
                if i + 1 < len(img):
                    imgCopy[i+1][j] = 0
                if j + 1 < len(img[i]):
                    imgCopy[i][j+1] = 0

    return imgCopy


def ccl(img):

    label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    currLabel = 1
    q = queue.Queue()
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            if img[i, j] == 255 and label[i, j] == 0:
                q.put((i, j))
                currLabel += 1
                while not q.empty():
                    pixel = q.get()
                    label[pixel[0], pixel[1]] = currLabel
                    for x in range(-1, 2):
                        for y in range(-1, 2):
                            if img[pixel[0] + x, pixel[1] + y] == 255 and label[pixel[0] + x, pixel[1] + y] == 0:
                                label[pixel[0] + x, pixel[1] + y] = currLabel
                                q.put((pixel[0] + x, pixel[1] + y))

    for x in range(1, img.shape[0]):
        for y in range(1, img.shape[1]):
            if label[x, y] == 2:
                label[x, y] = 255
            else:
                label[x, y] = 0

    return label


def perimeter(img):

    perimeter = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)

    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            if img[i, j] == 255:
                edge = False
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        if img[i + x, j + y] == 0:
                                edge = True
                if edge:
                    perimeter[i, j] = 255
    return perimeter


def test(img):
    label = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
    ringNo = 1
    q = queue.Queue()
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            if img[i, j] == 255 and label[i, j] == 0:
                label[i, j] = ringNo
                q.put((i, j))

                while not q.empty():
                    pixel = q.get()
                    for x in range(-1, 2):
                        for y in range(-1, 2):
                            if img[pixel[0] + x, pixel[1] + y] == 255 and label[pixel[0] + x, pixel[1] + y] == 0:
                                label[pixel[0] + x, pixel[1] + y] = ringNo
                                q.put((pixel[0] + x, pixel[1] + y))
                ringNo += 1
    return ringNo == 3


for i in range(15):
    img = cv.imread("C:/Users/User/Desktop/ITB2019/ComputerVision/assignment/Orings/Oring" + str(i+1) + ".jpg", 0)
    img = np.asarray(img, dtype=np.uint8)
    copy = img.copy()
    threshold_values = {}

    start = time.time()
    # generate histogram of image
    h = hist(img)
    # get threshold value
    threshold(h)
    thres = getThreshold()
    # threshold image
    img = thresholdedImage(img, thres)
    # show and save thresholded image
    cv.imshow("thresholded", img)
    # closing holes, dilation then erosion
    # dilation
    img = dilation(img)
    # erosion
    # img = erosion(img)
    # dilation and erosion image
    cv.imshow("dilated image", img)
    # connected component labelling
    img = ccl(img)
    cv.imshow("labelled image", img)
    # calculate perimeter and count ring
    img = perimeter(img)
    if test(img):
        result = "true"
    else:
        result = "fail"
    texts = result
    end = time.time()
    duration = end - start
    cv.putText(img, str(texts), (70, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(img, str(duration), (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.imshow("test image", img)
    # show original image
    cv.imshow("image", copy)
    cv.waitKey(0)
    cv.destroyAllWindows()

