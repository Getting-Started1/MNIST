# Create a simple tensor flow model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1)


# Convert to TFLite without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the full-precision model
with open('model_float32.tflite', 'wb') as f:
    f.write(tflite_model)

# Convert with quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()

# Save the quantized model
with open('model_quantized.tflite', 'wb') as f:
    f.write(quantized_tflite_model)
# Compare Model Sizes
import os

float_model_size = os.path.getsize('model_float32.tflite')
quantized_model_size = os.path.getsize('model_quantized.tflite')

print(f"Full-Precision Model Size: {float_model_size / 1024:.2f} KB")
print(f"Quantized Model Size: {quantized_model_size / 1024:.2f} KB")


#Test Full-Precision Model Accuracy
import numpy as np

# Load the full-precision TFLite model
interpreter_fp = tf.lite.Interpreter(model_path='model_float32.tflite')
interpreter_fp.allocate_tensors()

# Get input and output details
input_details_fp = interpreter_fp.get_input_details()
output_details_fp = interpreter_fp.get_output_details()

# Test accuracy of full-precision model
correct_fp = 0

for i in range(100):  # Evaluate on the first 100 test samples
    input_data = x_test[i:i+1].astype('float32')
    interpreter_fp.set_tensor(input_details_fp[0]['index'], input_data)
    interpreter_fp.invoke()
    output_data = interpreter_fp.get_tensor(output_details_fp[0]['index'])
    if np.argmax(output_data) == y_test[i]:
        correct_fp += 1

accuracy_fp = correct_fp / 100 * 100
print(f"Full-Precision Model Accuracy: {accuracy_fp:.2f}%")


# Test Quantized Model Accuracy
correct_quantized = 0

for i in range(100):  # Evaluate on the first 100 test samples
    input_data = x_test[i:i+1].astype('float32')
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if np.argmax(output_data) == y_test[i]:
        correct_quantized += 1

accuracy_quantized = correct_quantized / 100 * 100
print(f"Quantized Model Accuracy: {accuracy_quantized:.2f}%")