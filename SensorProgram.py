import serial
import time
import csv
import numpy as np
from textwrap import wrap
from pynput.keyboard import Key, Controller
import onnxruntime as rt

port = 'COM3'   # If incorrect COM port, replace with correct COM port.

path = "Replace this with the path to the folder with the .onnx files"

currentTime = 0
previousTime = 0

# Initialises a pynput controller for the keyboard that allows the software to press keys digitally.
keyboard = Controller()

try:
    # Initialises connection to serial monitor
    ser = serial.Serial(port, 9600, timeout=1)
    
    i = 10

    while True:
        # Read a line from the serial port
        
            line = ser.readline().decode('utf-8').rstrip()  # Reads in line from serial monitor, decodes it and strips it.
            line = line.lstrip('a$')
            splitLine = wrap(line, 5)   # Splits up the string every 5 characters, i.e. every channel
            
            # Occasionally, the sensor will fall out of sync and only supply two or three
            # channels of data. To ensure this does not interfere with the process, the 
            # program will only act if all four channels are present.
            if(len(splitLine) == 4):
                
                # This loop exists to trim down the amount of data being processed.
                if(i%10 == 0):
                    # Print each channel for visual inspection
                    print(splitLine[0]+"    "+splitLine[1]+"    "+splitLine[2]+"    "+splitLine[3])

                    # Load ONNX model into session. Currently loading decision tree model.
                    # If you want to load a different model type, replace 'dtree.onnx' with the appropriate
                    # ONNX file name.
                    # This code was based on the ONNX documentation (ONNX, n.d.)
                    sess = rt.InferenceSession(path+"dtree.onnx", providers=["CPUExecutionProvider"])
                    input_name = sess.get_inputs()[0].name
                    label_name = sess.get_outputs()[0].name
                    x = np.array(np.int64(splitLine[0])).reshape(1,1)
                    
                    # Runs the input channel through the ML model to classify it.
                    # Prints the prediction for visual inspection.
                    try:
                        r = sess.run([label_name], {input_name: x})[0]
                        print("Predicted label = "+str(r))
                    except (RuntimeError) as e:
                        print("Error with model prediction. Skipping line.")
                    
                    # Checks the current time.
                    currentTime = time.time()
                    
                    # Only presses the key if 2 seconds have passed since the last activation.
                    # This aims to prevent the key being pressed multiple times per muscle
                    # activation.
                    # Code based on response provided by user "Standard" (2019).
                    if((currentTime-previousTime)>=2 and str(r)=="[1]"):
                        Controller.press(Key.space)
                        Controller.release(Key.space)
                    previousTime = currentTime
                    
                    # Writes the value from the data channel being assessed and the assigned
                    # label to a .csv file for later analysis.
                    with open('C:/Users/maxbo/OneDrive/Desktop/FYP_Stuff/storedData.csv', 'a', newline = '') as file:
                        write = csv.writer(file)
                        write.writerow([splitLine[0],r])
                        
                i = i+1
            else:
                print("Buffering")
except KeyboardInterrupt:
    print('Exiting program.')
finally:
    if ser.is_open:
        ser.close()

'''
Bibliography:

ONNX, n.d., "sklearn-onnx: Convert your scikit-learn model into ONNX", 
sklearn-onnx, viewed 28 October 2024, https://onnx.ai/sklearn-onnx/index.html

'Standard', 2019, "Execute a method only when two seconds have elapsed from last execution time", 
StackOverflow, viewed 28 October 2024, https://stackoverflow.com/questions/58098177/execute-a-method-only-when-two-seconds-have-elapsed-from-last-execution-time

'''
