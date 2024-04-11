from PIL import Image
import pytesseract
import re
import cv2
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

#img = Image.open('test.jpeg')
img = cv2.imread('test.jpeg')

h, w, c = img.shape

d = pytesseract.image_to_data(img, output_type=Output.DICT)

n_boxes = len(d['text'])
print(d)
for i in range(n_boxes):

    if int(float(d['conf'][i])) > 30:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255,0),2)


cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
cv2.imshow('output', img,)
cv2.waitKey()
cv2.destroyAllWindows()