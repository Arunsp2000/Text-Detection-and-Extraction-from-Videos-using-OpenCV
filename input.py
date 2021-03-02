# Required Imports
import cv2
import os
import sys

# Reading input and taking setting the frames
inp=cv2.VideoCapture("text.mp4")
inp.set(1,400)

# Checking if there is a problem creating the inp_images directory
try:
    if not os.path.exists('inp_images'):
        os.makedirs('inp_images')
except OSError:
    print('Error creating input directory')

# Current Frame number
curr_frame=0

# Reads the frame and stores it in the directory
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
