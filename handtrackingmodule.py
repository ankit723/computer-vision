from typing import Text
import cv2
import mediapipe as mp
import time

class hand_detector():
    def __init__(self, mode = False, maxHands = 2, detectioncon = 0.7, trackcon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectioncon = detectioncon
        self.trackcon = trackcon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectioncon, self.trackcon)
        self.mpDraw = mp.solutions.drawing_utils


    def findhands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for self.handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, self.handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, handNo = 0, draw = True):
        xList = []
        yList = []
        bbox = []
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 6, (0, 0, 0), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), (0, 100, 50), 2)
        return lmList, bbox

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = hand_detector()
    while True:
        success, img = cap.read()
        img = detector.findhands(img)
        lmList = detector.find_position(img)
        
        cTime = time.time()           
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)



        cv2.imshow("Image", img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()