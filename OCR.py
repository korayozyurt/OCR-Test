from PIL import Image
import pytesseract
import re
import cv2
from matplotlib import pyplot as plt
from pytesseract import Output
import numpy as np

def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread('test.jpg')

#Smooth section
#Medianını alıp sonrada kernel kadar filtre yapmayı deneyelim blurlamak için sonrada gaussian aldık bakalım
kernel = np.ones((5,5),np.float32)/25
img = cv2.medianBlur(img,5)
img = cv2.filter2D(img,-1,kernel)
img = cv2.GaussianBlur(img,(5,5),0)
#Blurtlama kısmı tamam

#adaptive threshold
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

#erosion section
erosionKernel = np.ones((5, 5), np.uint8)
img = cv2.erode(img, kernel, iterations=1)

img = opening(img)

cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
cv2.imshow('output', img)
cv2.waitKey()

# closing all open windows
cv2.destroyAllWindows()


d = pytesseract.image_to_data(img, output_type=Output.DICT,config='--oem 3 --psm 6')
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (125, 255, 125), 2)

cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
cv2.imshow('output', img)
cv2.waitKey()

# closing all open windows
cv2.destroyAllWindows()

# text = pytesseract.image_to_string(img,config='--oem 3 --psm 6')
# print(text)
# data = []
# t = []
# prices = []
#
# word = r'([A-Z\s]+)'
# rakam = r'(\d+,\d+)'
#
# for s in text.splitlines():
#     s =  re.sub(r'(?<=\d),\s+', ',', s)
#     w = re.findall(word, s)
#     ra = re.findall(rakam, s)
#     #data.append(eslesmeler)
#     if(ra) :
#
#         print('word is', w, end= '   ')
#         print(ra)

#print(data)

