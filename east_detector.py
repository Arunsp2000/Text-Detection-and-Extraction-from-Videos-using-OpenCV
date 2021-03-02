# Required Imports
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import os
import pytesseract 

# Function which detects and extracts the text 
def captch_ex(folder):
    
    # Reading each image and extracting the text and storing it in a text file (lines 12-19)
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        orig = image.copy()
        extract_name='./text_extraction/text_extract_' + filename[:-4] + ".txt"
        file = open(extract_name, "w+")
        text = pytesseract.image_to_string(orig)  
        file.write(text) 
        file.close()
        
        # Resizing the images to make them have a multiple of 32(EAST model requires this)
        (H, W) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        (newW, newH) = (320, 320)
        rW = W / float(newW)
        rH = H / float(newH)
        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]
        
        # Setting the layer Names
        layerNames = ["feature_fusion/Conv_7/Sigmoid",
                      "feature_fusion/concat_3"]

        # Reading the model file
        net = cv2.dnn.readNet("frozen_east_text_detection.pb")

        # construct a blob from the image and then perform a forward pass of
        # Uses a method called mean subtraction which combats illumnination changes
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            
            # Tries to find the probable bouding boxes using the given scores
            Data_Scores = scores[0, 0, y]
            x0 = geometry[0, 0, y]
            x1 = geometry[0, 1, y]
            x2 = geometry[0, 2, y]
            x3 = geometry[0, 3, y]
            anglesD = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if Data_Scores[x] < 0.5:
                    continue
                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                # extract the rotation angle f
                # compute the sin and cosine
                angle = anglesD[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                # use the geometry volume to derive the width and height of the bounding box
                h = x0[x] + x2[x]
                w = x1[x] + x3[x]
                # compute both the starting and ending (x, y)-coordinates for
                endX = int(offsetX + (cos * x1[x]) + (sin * x2[x]))
                endY = int(offsetY - (sin * x1[x]) + (cos * x2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(Data_Scores[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            # draw the bounding box on the image
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

        #stores the image in the given directory
        name = './text_detection/text_detect_' + filename
        cv2.imwrite(name, orig)

# checks to see if the directory has a problem in being creted
try:
    if not os.path.exists('text_detection'):
        os.makedirs('text_detection')
except OSError:
    print('Error creating input directory')

try:
    if not os.path.exists('text_extraction'):
        os.makedirs('text_extraction')
except OSError:
    print('Error creating input directory')

# Setting the inp_folder and running the function
folder = "./inp_images"
captch_ex(folder)
