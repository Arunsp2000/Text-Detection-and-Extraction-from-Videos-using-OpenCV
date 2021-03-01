import cv2
import os
import sys


inp=cv2.VideoCapture("text.mp4")
inp.set(1,400)

try:
    if not os.path.exists('inp_images'):
        os.makedirs('inp_images')
except OSError:
    print('Error creating input directory')

curr_frame=0
while(1):
    ret,frame=inp.read()
    if(ret):
        name = './inp_images/inp_image_' + str(curr_frame) + '.jpg'
        print('Creating...'+ name)
        cv2.imwrite(name,frame)
        curr_frame+=1
    else:
        break

inp.release()
cv2.destroyAllWindows()
