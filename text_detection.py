import cv2
import os
import pytesseract 

img=cv2.imread("./inp_images/inp_image_0.jpg")
# pytesseract.pytesseract.tesseract_cmd=''
# cv2.imshow('rando',img)
# cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
# cv2.imshow("rand",thresh1)
# cv2.waitKey(0)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
# cv2.imshow("rand",rect_kernel)
# cv2.waitKey(0)
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
# cv2.imshow("rand",dilation)
# cv2.waitKey(0)
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
# cv2.imshow("rand",dilation)
# cv2.waitKey(0)
im2 = img.copy()

for cnt in contours: 
    x, y, w, h = cv2.boundingRect(cnt)  
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    cv2.imshow("rand",rect)
    cv2.waitKey(0)