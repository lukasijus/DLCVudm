from EXTRACTING_CREDIT_CARD import EXTRACT_DIGITS
import os
import cv2
from tensorflow import keras
from tensorflow.keras.models import load_model
from utils import pre_process
import numpy as np

root_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(root_dir, 'images')
model_dir = os.path.join(root_dir, 'model')

models = os.listdir(model_dir)
model_name = models[0]
model_path = model_dir + '\\' + model_name

images = os.listdir(images_dir)
image_name = images[0]
image_path = images_dir + '\\' + image_name

classifier = load_model(model_path)


CREDIT_CARD_1 = cv2.imread(image_path)
IMG = EXTRACT_DIGITS(CREDIT_CARD_1)
IMG_GRAY = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)


def x_cord_contour(contours):
    # Returns the X cordinate for the contour centroid
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return (int(M['m10'] / M['m00']))
    else:
        pass

# Blur image then find edges using Canny
blurred = cv2.GaussianBlur(IMG_GRAY, (5, 5), 0)
# cv2.imshow("blurred", blurred)
# cv2.waitKey(0)

edged = cv2.Canny(blurred, 30, 150)
# cv2.imshow("edged", edged)
# cv2.waitKey(0)

# Find Contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort out contours left to right by using their x cordinates
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:13]  # Change this to 16 to get all digits
contours = sorted(contours, key=x_cord_contour, reverse=False)

# Create empty array to store entire number
full_number = []

# loop over the contours
for c in contours:
    # compute the bounding box for the rectangle
    (x, y, w, h) = cv2.boundingRect(c)
    if w >= 5 and h >= 25 and cv2.contourArea(c) < 1000:
        roi = blurred[y:y + h, x:x + w]
        # ret, roi = cv2.threshold(roi, 20, 255,cv2.THRESH_BINARY_INV)
        cv2.imshow("ROI1", roi)
        roi_otsu = pre_process(roi, True)
        cv2.imshow("ROI2", roi_otsu)
        roi_otsu = cv2.cvtColor(roi_otsu, cv2.COLOR_GRAY2RGB)
        roi_otsu = keras.preprocessing.image.img_to_array(roi_otsu)
        roi_otsu = roi_otsu * 1. / 255
        roi_otsu = np.expand_dims(roi_otsu, axis=0)
        image = np.vstack([roi_otsu])
        label = str(classifier.predict_classes(image, batch_size=10))[1]
        print(label)
        (x, y, w, h) = (x + 0, y + 0, w, h)
        cv2.rectangle(IMG, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(IMG, label, (x, y + 90), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("image", IMG)
        cv2.waitKey(0)

cv2.destroyAllWindows()