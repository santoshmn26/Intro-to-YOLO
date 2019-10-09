# -*- coding: utf-8 -*-
"""
Simple GUI setup for the Hard Hat Application - VA
"""

import tkinter as Tk
from darkflow.net.build import TFNet

def initGUI ():
    top = Tk.Tk()
    top.geometry('940x460') #Change here to alter the Application size
    top.title('Video Analytics - Hard Hat')

	
    global saved_option
    saved_option = 0

    Canvas = Tk.Canvas(top, width = 640, height = 480)
    B_detect = Tk.Button(top, text = 'DETECT', font = ("arial", 12))
    B_reset = Tk.Button(top, text = 'RESET', font = ("arial", 12))
    B_capture = Tk.Button(top, text = 'SHOOT', font = ("arial", 12))
    
    Canvas.place(x = 15, y = 15, width = 640, height = 480)
    B_detect.place(x = 15 + 640 + 30, y = 30, width = 150, height = 60)
    B_capture.place(x = 15 + 640 + 30, y = 150, width = 150, height = 60)
    B_reset.place(x = 15 + 640 + 30, y = 270, width = 150, height = 60)
   
    buttons = (B_reset, B_detect, B_capture)
    
    buttons[0].configure(command = reset)
    buttons[1].configure(command = setOption)
    #buttons[3].configure(command = functionName) #If you would like to configure more buttons
    option = {
        'model': 'cfg/tiny-yolo-voc-3c.cfg',  # Model to load
        'load': -1,                           # Weights to load
        'threshold': 0.3,                     # Threshold value
        'gpu': 0                              # GPU value (0 - 1)         
    }
    video_source='camera'
    global tfnet
    tfnet = TFNet(option)
    
    return top, Canvas, saved_option, tfnet, buttons[2]


#Function to set the option to start detection
def setOption ():
    global saved_option
    print ('Detection Started...!!')
    saved_option = 1

#Function to reset the option
def reset ():
    global saved_option
    print ('Terminating Detection..! ')
    saved_option = 0
    