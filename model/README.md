# IoT-challenge
This model extends the capabilities of IR camera-based leak monitoring system from leak detection only to
automated leak classification, and shows high accuracy for binary and eight-class classification.

### Data
We trained the model on the GasVid dataset.

### Model
We constructed 2D CNN model using TensorFlow-Keras library.

### Run the detection
```
python detect_gas.py --model detect.tflite --video gas_leak_test.mp4
``` 
