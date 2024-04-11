from PIL import Image
import pytesseract
import re
import cv2
from pytesseract import Output
import numpy as np


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

#img = Image.open('test.jpeg')
img_gray_mode = cv2.imread('test.jpg')

lab= cv2.cvtColor(img_gray_mode, cv2.COLOR_BGR2LAB)
l_channel, a, b = cv2.split(lab)

# Applying CLAHE to L-channel
# feel free to try different values for the limit and grid size:
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(l_channel)

# merge the CLAHE enhanced L-channel with the a and b channel
limg = cv2.merge((cl,a,b))

# Converting image from LAB Color model to BGR color spcae
enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Stacking the original image with the enhanced image
result = np.hstack((img_gray_mode, enhanced_img))

ret,thresh1 = cv2.threshold(img_gray_mode,115,255,cv2.THRESH_BINARY)

#cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
cv2.imshow('output', thresh1)
cv2.waitKey()
#cv2.destroyAllWindows()

# closing all open windows
cv2.destroyAllWindows()
text = pytesseract.image_to_string(img_gray_mode,config='--psm 6')
print(text)
data = []
t = []
prices = []

word = r'([A-Z\s]+)'
rakam = r'(\d+,\d+)'

for s in text.splitlines():
    s =  re.sub(r'(?<=\d),\s+', ',', s)
    w = re.findall(word, s)
    ra = re.findall(rakam, s)
    #data.append(eslesmeler)
    if(ra) :

        print('word is', w, end= '   ')
        print(ra)

#print(data)

