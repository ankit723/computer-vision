import cv2
import time
import numpy as np
import handtrackingmodule as htm
import math as m
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

######################################################################
wCam, hCam = 1000, 1000
######################################################################


cap = cv2.VideoCapture(0)
cap.set(10, wCam)
cap.set(10, hCam)
pTime = 0

detector = htm.hand_detector(detectioncon=0.8, trackcon=0.7)



devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minvol = volRange[0]
maxvol = volRange[1]
volBar = 400


while True:
    success, img = cap.read()
    img = detector.findhands(img)
    lmList, bbox = detector.find_position(img, draw=True)
    if len(lmList) !=0:
        print(bbox)
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2) // 2, (y1+y2) // 2
        cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
        length = m.hypot(x2-x1, y2-y1)
        vol = np.interp(length,[35,150], [minvol, maxvol])
        volBar = np.interp(length,[35,150], [400, 150])
        print(int(vol))
        volume.SetMasterVolumeLevel(vol, None)

        if length<35:
            cv2.circle(img, (cx, cy), 8, (0, 255, 25), cv2.FILLED)
            
    cv2.rectangle(img, (50, 150), (85, 400), (0, 20, 399), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 20, 399), cv2.FILLED)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime 
    cv2.putText(img, f"FPS: {int(fps)}", (40, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("img", img)
    cv2.waitKey(1)
