from itertools import count
import cv2
import os 

# Opens the Video file
cam = cv2.VideoCapture("C:\\Users\\phili\\Videos\\KI\\WIN_20220707_10_02_45_Pro.mp4")

try: 
    #Einen Ordner mit dem Namen data erstellen
    if not os.path.exists('data'):
        os.makedirs('data')

#Wenn nicht erstellt, dann error
except OSError:
    print ('Error: Creating direcotry of data')

currentframe = 0

while(True):

    #reading from frame
    ret,frame = cam.read()

    if ret:
        if currentframe % 30 == 0:
            # if video still left continue creating images
            name = './data/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name)
            
            #writing the extracted images
            cv2.imwrite(name, frame)
        
        #increasing counter so that will
        #show how many frames are created
        currentframe += 1
    else:
        break

cam.release()
cv2.destroyAllWindows()