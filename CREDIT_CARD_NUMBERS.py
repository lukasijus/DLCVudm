import os
import cv2
import numpy as np
import utils

CARD_1 = cv2.imread('26. Credit Card\\creditcard_digits1.jpg', 0)
CARD_2 = cv2.imread('26. Credit Card\\creditcard_digits2.jpg')

HEIGHT = np.min([CARD_1.shape[0], CARD_2.shape[0]])
WIDHT  = np.min([CARD_1.shape[1], CARD_2.shape[1]])

CARD_1 = cv2.resize(CARD_1,(WIDHT, HEIGHT))
CARD_2 = cv2.resize(CARD_2,(WIDHT, HEIGHT))
CARD_2 = cv2.cvtColor(CARD_2, cv2.COLOR_BGR2GRAY)
_, th2 = cv2.threshold(CARD_2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
top_left_x = 0
# Create 2000 Images for train
for i in range(0, 11):
    # We jump the next digit each time we loop
    if i > 0:
        cv2.imshow('digit', CARD_2[5:50,top_left_x:top_left_x + 35])
        top_left_x = top_left_x + 35
        cv2.waitKey()
    roi = CARD_2[5:50,top_left_x:top_left_x + 35]
    # We create 200 versions of each image for our dataset
    for j in range(1001, 2000):
        roi2 = utils.DigitAugmentation(roi)
        roi_otsu = utils.pre_process(roi2, inv=True)
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path,  'valid ation')
        path = os.path.join(path, str(i))
        cv2.imwrite(path + '\\' + str(j) + '.jpg', roi_otsu)



