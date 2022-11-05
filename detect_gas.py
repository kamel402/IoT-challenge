# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util


# Define and parse input arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')

parser.add_argument('--video', help='Name of the video file',
                    default='gas_leak_test.mp4')


args = parser.parse_args()

MODEL_NAME = args.model
VIDEO_NAME = args.video

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter


# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH, VIDEO_NAME)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME)


# Load the label map
labels = ['No Leak',
          '1_leak',
          '2_leak',
          '3_leak',
          '4_leak',
          '5_leak',
          '6_leak',
          '7_leak']


# Load the Tensorflow Lite model.

interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

backSub = cv2.createBackgroundSubtractorKNN()

counter = 0
predections = []

while (video.isOpened()):

    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
        print('Reached the end of the video!')
        break

    fgMask = backSub.apply(frame)

    frame_resized = cv2.resize(fgMask, (width, height))
    input_data = np.expand_dims(frame_resized, axis=2)
    input_data = np.expand_dims(input_data, axis=0)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    classes = interpreter.get_tensor(output_details[0]['index'])[0]  # Class index of detected objects

    if counter % 15 == 0:
        try:
            # Look up object name from "labels" array using class index
            leak_type = labels[max(set(predections), key=predections.count)]
        except:
            leak_type = labels[0]
        predections = []
    else:
        predections.append(np.argmax(classes))
    counter += 1

    frame = cv2.putText(frame, leak_type,
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255),
                        2, cv2.LINE_AA)

    print(leak_type)
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Gas Leak detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()
