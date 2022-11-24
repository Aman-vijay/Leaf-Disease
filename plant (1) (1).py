from subprocess import call
from picamera import PiCamera
from time import sleep
import numpy as np
import cv2

camera = PiCamera()
camera.start_preview()
sleep(10)
camera.capture('/home/pi/Desktop/p52.jpg')
camera.stop_preview()
import numpy as np
import cv2

img = cv2.imread('/home/pi/Desktop/p52.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow('res2',res2)
cv2.imwrite('/home/pi/Desktop/res2.jpg',res2)

while(1):
    # Take each frame
    frame = cv2.imread('/home/pi/Desktop/p52.jpg')
 
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([25,120,150])
    upper_blue = np.array([62,174,250])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imwrite("mask.jpg",mask)
    cv2.imshow('res',res)
    cv2.imwrite("res.jpg",res)
    Upload = "/home/pi/Dropbox-Uploader/dropbox_uploader.sh upload /home/pi/res.jpg res.jpg"
    call ([Upload], shell=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

