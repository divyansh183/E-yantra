'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 2B of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''
############################## FILL THE MANDATORY INFORMATION BELOW ###############################

# Team ID:			[ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:			task_2b.py
# Functions:	    [`classify_event(image)` ]
###################################################################################################

# IMPORTS (DO NOT CHANGE/REMOVE THESE IMPORTS)
from sys import platform
import numpy as np
import subprocess
import shutil
import ast
import sys
import os

# Additional Imports
'''
You can import your required libraries here
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Import PyTorch
import torch
from torch import nn

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Import torchvision 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

# Import matplotlib for visualization
import matplotlib.pyplot as plt



# DECLARING VARIABLES (DO NOT CHANGE/REMOVE THESE VARIABLES)
detected_list = []
numbering_list = []
img_name_list = []

# Declaring Variables
'''
You can delare the necessary variables here
'''
event_categories = {
    0: "Combat",
    1: "DestroyedBuildings",
    2: "Fire",
    3: "Humanitarian Aid and Rehabilitation",
    4: "Military Vehicles and Weapons"
}

data_dir = "Train"

# EVENT NAMES
'''
We have already specified the event names that you should train your model with.
DO NOT CHANGE THE BELOW EVENT NAMES IN ANY CASE
If you have accidently created a different name for the event, you can create another 
function to use the below shared event names wherever your event names are used.
(Remember, the 'classify_event()' should always return the predefined event names)  
'''
combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"
###################################################################################################
###################################################################################################
''' 
	Purpose:
	---
	This function will load your trained model and classify the event from an image which is 
    sent as an input.
	
	Input Arguments:
	---
	`image`: Image path sent by input file 	
	
	Returns:
	---
	`event` : [ String ]
						  Detected event is returned in the form of a string

	Example call:
	---
	event = classify_event(image_path)
	'''
def classify_event(image):
    '''
    ADD YOUR CODE HERE
    '''
    image_size = (224, 224)
    batch_size = 32

    # Load the pre-trained model 
    model = tf.keras.models.load_model("event_classification_model.h5")


    # train_datagen = ImageDataGenerator(
    # rescale=1.0/255.0,  # Normalize pixel values
    # rotation_range=20,  # Data augmentation: random rotation
    # width_shift_range=0.2,  # Data augmentation: random horizontal shift
    # height_shift_range=0.2,  # Data augmentation: random vertical shift
    # horizontal_flip=True,  # Data augmentation: horizontal flip
    # validation_split=0.3  # Split a portion for validation
    # )

    # train_generator = train_datagen.flow_from_directory(
    # data_dir,
    # target_size=image_size,
    # batch_size=batch_size,
    # class_mode='categorical',
    # subset='training'
    # )

    # validation_generator = train_datagen.flow_from_directory(
    # data_dir,
    # target_size=image_size,
    # batch_size=batch_size,
    # class_mode='categorical',
    # subset='validation'
    # )

    # Load and preprocess the input image
    img = tf.keras.preprocessing.image.load_img(image, target_size=image_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize pixel values
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    event = event_categories[predicted_class_index]
    
    return event

# ADDITIONAL FUNCTIONS
'''
Although not required but if there are any additonal functions that you're using, you shall add them here. 
'''


# Create data generators for training and validation


###################################################################################################
########################### DO NOT MAKE ANY CHANGES IN THE SCRIPT BELOW ###########################
def classification(img_name_list):
    for img_index in range(len(img_name_list)):
        img = "events/" + str(img_name_list[img_index]) + ".jpeg"
        detected_event = classify_event(img)
        print((img_index + 1), detected_event)
        if detected_event == combat:
            detected_list.append("combat")
        if detected_event == rehab:
            detected_list.append("rehab")
        if detected_event == military_vehicles:
            detected_list.append("militaryvehicles")
        if detected_event == fire:
            detected_list.append("fire")
        if detected_event == destroyed_building:
            detected_list.append("destroyedbuilding")
    shutil.rmtree('events')
    return detected_list

def detected_list_processing(detected_list):
    try:
        detected_events = open("detected_events.txt", "w")
        detected_events.writelines(str(detected_list))
    except Exception as e:
        print("Error: ", e)

def input_function():
    if platform == "win32":
        try:
            subprocess.run("input.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./input")
        except Exception as e:
            print("Error: ", e)
    img_names = open("image_names.txt", "r")
    img_name_str = img_names.read()

    img_name_list = ast.literal_eval(img_name_str)
    return img_name_list
    
def output_function():
    if platform == "win32":
        try:
            subprocess.run("output.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./output")
        except Exception as e:
            print("Error: ", e)

###################################################################################################
def main():
    ##### Input #####
    img_name_list = input_function()
    #################

    ##### Process #####
    detected_list = classification(img_name_list)
    detected_list_processing(detected_list)
    ###################

    ##### Output #####
    output_function()
    ##################

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        if os.path.exists('events'):
            shutil.rmtree('events')
        sys.exit()
###################################################################################################
###################################################################################################
