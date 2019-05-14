import cv2
from darkflow.net.build import TFNet
import numpy as np
import datetime
import time
import threading 
import math


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




def write_data(result):

    f=open("log.txt",'a+')
    for i in results:
        f.write(str(datetime.datetime.now())+','+i['label']+','+str(i['confidence'])+','+str(i['topleft']['x'])+','+str(i['topleft']['y'])+','+str(i['bottomright']['x'])+','+str(i['bottomright']['y'])+','+video_source+'\n')
        f.close

if __name__=="__main__":



    option = {
        'model': 'cfg/tiny-yolo-voc-3c.cfg',  # Model to load
        'load': -1,                           # Weights to load
        'threshold': 0.3,                     # Threshold value
        'gpu': 0                              # GPU value (0 - 1)         
        
    }
    video_source='camera'
    

    
    tfnet = TFNet(option)
    capture = cv2.VideoCapture(0)        # Place your path to your video file, or assign 0 to load webcam feed.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    initial_thread_count = threading.enumerate()
    i,prev=1,(0,0)
    while (capture.isOpened()):
        stime = time.time()
        ret, frame = capture.read()
        if ret:
            thread_count = threading.enumerate()
            results = tfnet.return_predict(frame)
            if (len(thread_count)==len(initial_thread_count)):
                threading.Timer(5.0, write_data,args=(results,)).start()    
            for result in results:

                if(result['label']=='no_hat'):
                    draw_border(frame, ((result['topleft']['x']),(result['topleft']['y'])), ((result['bottomright']['x']),(result['bottomright']['y'])), (0,0,255), 4, 0, 14,i)
                    
                #else:
                    draw_border(frame, ((result['topleft']['x']),(result['topleft']['y'])), ((result['bottomright']['x']),(result['bottomright']['y'])), (0,0,255), 4, 5, 14,1)
                frame = cv2.putText(frame, result['label'],((result['topleft']['x']),(result['bottomright']['y'])+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 1)    
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            capture.release()
            cv2.destroyAllWindows()
            break

#===============================================================================