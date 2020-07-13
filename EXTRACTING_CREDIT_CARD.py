import cv2
import numpy as np
import os
import utils

images_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(images_dir, 'images')

images = os.listdir(images_dir)
image = images[0]
image_path = images_dir + '\\' + image

CREDIT_CARD_1 = cv2.imread(image_path)
def EXTRACT_DIGITS(IMAGE):
    CREDIT_CARD_1_GRAY = cv2.cvtColor(CREDIT_CARD_1, cv2.COLOR_BGR2GRAY)
    ret, TRESH = cv2.threshold(CREDIT_CARD_1_GRAY, 100,255,0)
    IMG = cv2.Canny(TRESH, 0, 1)
    COUNTOURS, hierarchy = cv2.findContours(IMG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    w_arr = []
    h_arr = []
    for coordinate in COUNTOURS[0]:
        coordinate = coordinate[0]
        w, h = coordinate
        w_arr.append(w)
        h_arr.append(h)



    X0 = np.min(w_arr)
    Y0 = np.min(h_arr)
    X1 = np.max(w_arr)
    Y1 = np.max(h_arr)

    IMG2 = CREDIT_CARD_1[Y0:Y1, X0:X1]
    region = [(50, 150), (380, 290)]

    top_left_y = region[0][1]
    bottom_right_y = region[1][1]
    top_left_x = region[0][0]
    bottom_right_x = region[1][0]

    # Extracting the area were the credit numbers are located
    roi = IMG2[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    return roi