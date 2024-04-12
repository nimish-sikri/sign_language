#include <TensorFlowLite.h>
#include <SoftwareSerial.h> // Include the SoftwareSerial library for Bluetooth communication

// Include the converted TFLite model file
#include "ml_model.h"

// Define the analog pins connected to the flex sensors
const int flexSensorPins[] = {A0, A1, A2, A3};
const int numFlexSensors = 4;

// Create a TensorFlow Lite interpreter
TfLiteTensor *input;
TfLiteTensor *output;
TfLiteInterpreter *interpreter;

// Define min and max values for normalization
const int minSensorValue = 0;
const int maxSensorValue = 1023; // Assuming 10-bit ADC resolution

// Define the RX and TX pins for Bluetooth communication
const int bluetoothRxPin = 2; // RX pin of the Bluetooth module
const int bluetoothTxPin = 3; // TX pin of the Bluetooth module

// Initialize a SoftwareSerial object for Bluetooth communication
SoftwareSerial bluetoothSerial(bluetoothRxPin, bluetoothTxPin);

void setup()
{
  // Initialize serial communication for debugging
  Serial.begin(9600);

  // Initialize Bluetooth communication
  bluetoothSerial.begin(9600);

  // Initialize TensorFlow Lite interpreter
  interpreter = tflite::GetModel(ml_model);
  interpreter->AllocateTensors();

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop()
{
  // Read analog values from all flex sensors
  float flexValues[numFlexSensors];
  for (int i = 0; i < numFlexSensors; i++)
  {
    int flexValue = analogRead(flexSensorPins[i]);
    // Preprocess sensor data: normalize to range [0, 1]
    float normalizedValue = map(flexValue, minSensorValue, maxSensorValue, 0, 1);
    flexValues[i] = normalizedValue;
  }

  // Set input tensor values
  for (int i = 0; i < numFlexSensors; i++)
  {
    input->data.f[i] = flexValues[i];
  }

  // Run inference
  interpreter->Invoke();

  // Get inference result
  float outputValue = output->data.f[0];

  // Interpret inference result
  int predictedSign = round(outputValue);

  // Send predicted sign via Bluetooth
  bluetoothSerial.println(predictedSign);

  // Print predicted sign for debugging
  Serial.print("Predicted sign: ");
  Serial.println(predictedSign);

  // Delay before next inference
  delay(1000);
}
