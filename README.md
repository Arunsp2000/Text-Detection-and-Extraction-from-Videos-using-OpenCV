# Text Detection and Extraction from Videos using OpenCV
This is a program implemented to detect and extract text from images using East Detection Model and Pytesseract respectively.

# Requirements
- ```imutils==0.5.4```
- ```pytesseract==0.3.7```
- ```numpy==1.16.2```
- ```opencv_python==4.2.0.34``` 

# Usage
- Clone the repository.
- Delete the existing folders ```inp_images```, ```text_detection``` and ```text_extraction```.(Do this process everytime after running the codes)
- Change the path to the input file in the ```input.py``` file.
- Run the input file using
    ```bash
    python3 input.py 
    ```
- An ```inp_images``` directory is formed which contains the input frames of the input video. The number of frames can be set by changing the ```inp.set(1,"enter_frames_here")``` command in the above file.
- Run the East Detection file next using
    ```bash
    python3 east_detector.py 
    ```
# Folders
- ```inp_images/inp_images_(frame_number).jpg``` : Contains the input images of the video.
- ```text_detection/text_detect_inp_image_(frame_number).jpg``` : Contains the detected portions of the inp_images which contain text.
- ```text_extraction/text_extract_inp_image_(frame_number).txt``` : Contains the extracted text portion of the inp_images.

# Sample Outputs:
<table>
    <tr>
        <td colspan = "2" align = "center" >Bounding Boxes using the East Detection Method </td>
    </tr>
    <tr>
        <td><img src="text_detection/text_detect_inp_image_98.jpg" height="400" width = "600"></td>
        <td><img src="text_detection/text_detect_inp_image_253.jpg" height="400" width = "600"></td>
    </tr>
 </table>

The text extraction results can be found in the text_extraction folder.
- [text_extract_inp_image_98.txt](text_extraction/text_extract_inp_image_98.txt)
- [text_extract_inp_image_253.txt](text_extraction/text_extract_inp_image_253.txt)



# References
 - Pyimagesearch : [EAST Text Detection](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/)
 - .pb file :[EAST DETECTION MODEL FILE](https://github.com/oyyd/frozen_east_text_detection.pb)
 - Extraction : [Pytesseract](https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/)
