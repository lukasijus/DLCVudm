import os
import cv2
import numpy as np
import random
import cv2
from scipy.ndimage import convolve

def makedir(directory):
    """Create a new directory if it does not exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        return print('Directory. ', directory, ' created successfully')
    else:
        return print('Directory already exist!')


def DigitAugmentation(frame, dim=32):
    """Randomly alters the image using noise, pixelation and streching image functions"""
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    random_num = np.random.randint(0, 9)

    if (random_num % 2 == 0):
        frame = add_noise(frame)
    if (random_num % 3 == 0):
        frame = pixelate(frame)
    if (random_num % 2 == 0):
        frame = stretch(frame)
    frame = cv2.resize(frame, (dim, dim), interpolation=cv2.INTER_AREA)

    return frame


def add_noise(image):
    """Addings noise to image"""
    prob = random.uniform(0.01, 0.05)
    rnd = np.random.rand(image.shape[0], image.shape[1])
    noisy = image.copy()
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 1
    return noisy


def pixelate(image):
    "Pixelates an image by reducing the resolution then upscaling it"
    dim = np.random.randint(8, 12)
    image = cv2.resize(image, (dim, dim), interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
    return image


def stretch(image):
    "Randomly applies different degrees of stretch to image"
    ran = np.random.randint(0, 3) * 2
    if np.random.randint(0, 2) == 0:
        frame = cv2.resize(image, (32, ran + 32), interpolation=cv2.INTER_AREA)
        return frame[int(ran / 2):int(ran + 32) - int(ran / 2), 0:32]
    else:
        frame = cv2.resize(image, (ran + 32, 32), interpolation=cv2.INTER_AREA)
        return frame[0:32, int(ran / 2):int(ran + 32) - int(ran / 2)]


def pre_process(image, inv=False):
    """Uses OTSU binarization on an image"""
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        gray_image = image
        pass

    if inv == False:
        _, th2 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, th2 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(th2, (32, 32), interpolation=cv2.INTER_AREA)
    return resized
