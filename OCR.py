from PIL import Image
import pytesseract
import re
import cv2
import numpy as np

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated


# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

def extract_text(img):
    # Smooth section
    # Medianını alıp sonrada kernel kadar filtre yapmayı deneyelim blurlamak için sonrada gaussian aldık bakalım
    kernel = np.ones((5, 5), np.float32) / 25
    img = cv2.blur(img, (5, 5))
    img = cv2.medianBlur(img, 9)
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Blurtlama kısmı tamam

    # adaptive threshold
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    img = cv2.erode(img, kernel, iterations=1)
    img = opening(img)

    text = pytesseract.image_to_string(img, config='--psm 6')

    wordPattern = r'^.*?(?=\s*\d+\s*[\.\,]?\s*\d+\s*$)'
    rakamPattern = r'\d+\s*[\.\,]?\s*\d+\s*$'

    for s in text.splitlines():
        s = re.sub(r'[^\w\s,]|_', '', s)
        s = s.strip().replace('x', '')
        ra = re.findall(rakamPattern, s)
        w = re.findall(wordPattern, s)

        if(w):
            w = w[0].strip()

        if(w and ra):

            number_str = ra[0].replace(',', '.')  # Virgülü noktaya dönüştürme
            number_str = re.sub(r'\s+', '', number_str)
            number_float = float(number_str)
            print('text is: ', w, end= ' \t')
            print('rakam is: ', number_float)


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread('test.jpg')

# aslında 4 yerine 2 de yetiyor.
height, width, channels = img.shape
new_height = height // 3

for i in range(3):
    y1 = i * new_height
    y2 = (i + 1) * new_height

    tempIm = img[y1:y2,:]

    extract_text(tempIm)


