# -*- coding: utf-8 -*-
"""
"""

import sys
import numpy as np 
import cv2
import tkinter as Tk
from PIL import Image
from PIL import ImageTk
import GUI
from darkflow.net.build import TFNet
import queue
import math
import sys
import threading 

def capture(end_img):
    import time
    localtime = time.localtime()[0:5]
    timestr = str(localtime[0])[2:4]
    for i in range(1,5):
        timestr += str(localtime[i])
        if i == 2:
            timestr += '_'
    cv2.imwrite('photoshot' + timestr + '.jpg', end_img)
    print('Image saved.')

top, Canvas, saved_option, tfnet, B_capture = GUI.initGUI()


frame_buffer_size = 7 # The queue size for keeping video frame for processing.

# To Capture frames from the CCTV
fn = 0
#'rtsp://admin:123456@169.254.178.249:554/1' - Replace with 0 if Webcam feed.

cap = cv2.VideoCapture(fn)

# Creating a Queue for Frame_Buffer
frame_buffer = queue.Queue(maxsize=frame_buffer_size)

# Function to exit the program
def CloseAndExit():
    cap.release()# release camera
    cv2.destroyAllWindows()# release screen
    sys.exit() # exit program
    top.quit() # GUI exist
    top.destroy() #release GUI

def read_buffer():
    ret = True
    while (ret):
        if frame_buffer.full():
            frame_buffer.get()
        # Reading frame-by-frame
        ret, buffer_frame = cap.read()
        # We load the images to frame buffer to have it in queue
        frame_buffer.put(buffer_frame)
    CloseAndExit()

#Once the Initial GUI is set up, start the threading concept
initial_thread_count = threading.enumerate()
i,prev=1,(0,0)
# Start thead functions to continue their task parallelly
threading.Thread(target=read_buffer, daemon=True).start()


# Drawing bounding boxes
def draw_border(img, pt1, pt2, color, thickness, r, d,i):           

    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.line(img, (x1 + r + d, y1), (x2 - r - d, y1),color, 1)                  # Thin line
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.line(img, (x2, y1 + r + d), (x2, y2 - r - d), color, 1)                 # Thin line
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    #Bottom left

    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.line(img, (x1 + r + d, y2), (x2 - r - d, y2), color, 1)                 # Thin line
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right

    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.line(img, (x1, y1 + r + d), (x1, y2 - r - d), color, 1)                 # Thin line
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


while (cap.isOpened()):
    try:
        frame = frame_buffer.get()          #  Frames to show in initial GUI       
        frame_out = frame.copy()
        if GUI.saved_option != 0:           #  If Detect button is pressed : Start Detection
           if frame_buffer.empty() != True:
               frame = frame_buffer.get()
               frame_out = frame.copy()
               results = tfnet.return_predict(frame_out)
               for result in results:

                   if(result['label']=='no_hat'):
                       draw_border(frame_out, ((result['topleft']['x']),(result['topleft']['y'])), ((result['bottomright']['x']),(result['bottomright']['y'])), (0,0,255), 4, 0, 14,i)
                       frame_out = cv2.putText(frame_out, result['label'],((result['topleft']['x']),(result['topleft']['y']-10)), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)
                    
                   else:
                       draw_border(frame_out, ((result['topleft']['x']),(result['topleft']['y'])), ((result['bottomright']['x']),(result['bottomright']['y'])), (0,0,255), 4, 5, 14,1)
                       frame_out = cv2.putText(frame_out, result['label'],((result['topleft']['x']),(result['topleft']['y'])-30), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 1)                          
        
        end_img = frame_out                
        
        #end_img = cv2.flip(end_img, 1)
        show_img = cv2.cvtColor(end_img, cv2.COLOR_BGR2RGB)
        show_img = Image.fromarray(show_img)
        show_img= ImageTk.PhotoImage(image = show_img) 
        Canvas.create_image(0, 0, anchor = Tk.NW,  image = show_img)
        
        B_capture.configure(command = lambda: capture(end_img))
        
        top.update()
		
        if cv2.waitKey(10) == ord('q') or cv2.waitKey(10) == 27:
           CloseAndExit()
           break
    
    except:
        print(sys.exc_info()) 
        cap.release()
        cv2.destroyAllWindows()
        top.quit()
        top.destroy()