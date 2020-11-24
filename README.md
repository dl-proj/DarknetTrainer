# DarknetTrainer

## Overview

This project is to detect the license plate of the cars and recognize the plate number.

The first part is to detect the license plate of the cars, which can be solved with Darknet by using the Yolo v3 model. The training dataset with the detected license plate is made manually from the images and video frames.

The second part is to recognize the digits and letters in the license plate, which can be trained with the train dataset using the Keras framework.

## Configuration

- darknet
    
    The source code to train the Yolo model with the train dataset.
    
    There is no custom_data directory,  file in initial framework.
    
    We should make custom_data directory and download weights and conv file.
    
- training_dataset
    
    Annotated dataset with images and xmls
    
- utils
    
    Several functions to prepare for training dataset

## Installation

- Environment

    Ubuntu 18.04, Python 3.6, GPU
    
- Dependency Installation

    ```
        pip3 install -r requirements.txt
    ```
- Download models

    * For the model to detect the license plate, download from and copy them in /utils/model/models/lp_detection_model 
    directory.
    
## Execution

- Please set the several path(DETECT_FRAME_PATH, HANDWRITTEN_DIGITS_TRAINING_PATH, HANDWRITTEN_DIGITS_TESTING_PATH,
HANDWRITTEN_LETTERS_TRAINING_PATH, HANDWRITTEN_LETTERSS_TESTING_PATH, etc) in settings file.

- Please run the following command in terminal in this project directory.

    ```
        python3 main.py
    ```

## Appendix

If you want to train the Yolo model and the handwritten recognition model, please read the following instructions.

- Training Yolo model
    
    * Preparation for training Yolo model
    
        Please download the darknet framework from https://github.com/pjreddie/darknet, make custom_data directory in it, 
        make the files whose name are custom.names and detector.data referencing 
        https://blog.francium.tech/custom-object-training-and-detection-with-yolov3-darknet-and-opencv-41542f2ff44e.
        Also, in the custom_data directory, the test and train text file is needed. The process to make these files is 
        referred in the execution step. And insert labels directory and images directory which contains the xml and jpg
        files for training in the custom_data folder. 
        
        Then download yolov3.weights from https://pjreddie.com/darknet/yolo/ and 
        darkenet53.conv.74 from https://pjreddie.com/media/files/darknet53.conv.74 and copy them in darknet directory.
    
    * Train the model to detect license plate of the cars
                    
        Then plesae run the following command in the terminal. 
        ```
            ./darknet detector train custom_data/detector.data custom_data/cfg/yolov3-custom.cfg darknet53.conv.74
        ```
        
        After finishing train, copy the trained model(.weights) to utils/model/models/lp_detection_model directory.
      
- Training the model to recognize the handwritten digits and letters
    
    * Preparation for training the handwritten letters and digits
    
        Please download the handwritten digits and letters dataset from Kaggle and copy them /data directory. 
        Since the image sizes of the handwritten digits(28 * 28) are different from the one of handwritten letters
        (32 * 32), the adjustment between their sizes is needed.
        
## Main reference site

    https://blog.francium.tech/custom-object-training-and-detection-with-yolov3-darknet-and-opencv-41542f2ff44e
