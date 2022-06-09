import math
from math import *
import screen_brightness_control as sbc
import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import *
from google.protobuf.json_format import MessageToDict


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
vc = cast(interface, POINTER(IAudioEndpointVolume))
Range = vc.GetVolumeRange()
minR, maxR = Range[0], Range[1]
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

bri = 0
briBar = 400
briPer = 0
vol = 0
volBar = 500
volPer = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []

    if results.multi_hand_landmarks:

        # Both Hands are present in image(frame)
        if len(results.multi_handedness) == 2:
            # Display 'Both Hands' on the image
            cv2.putText(img, 'Both Hands', (250, 50),cv2.FONT_HERSHEY_COMPLEX, 0.9,(0,0 , 255), 2)

        # If any hand present
        else:
            for i in results.multi_handedness:

                # Return weather it is Right or Left Hand
                label = MessageToDict(i)[
                    'classification'][0]['label']

                if label == 'Right':
                    # Display 'Left Hand' on left side of window
                    cv2.putText(img, 'BRIGHTNESS', (20, 50),cv2.FONT_HERSHEY_COMPLEX, 0.9,(0, 0, 255), 2)
                    if results.multi_hand_landmarks:
                        for handlandmark in results.multi_hand_landmarks:
                            for id, lm in enumerate(handlandmark.landmark):
                                h, w, _ = img.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                lmList.append([id, cx, cy])
                            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

                        if lmList != []:
                            x1, y1 = lmList[4][1], lmList[4][2]
                            x2, y2 = lmList[8][1], lmList[8][2]

                            cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
                            cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
                            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 200), 3)

                            length = hypot(x2 - x1, y2 - y1)

                            bright = np.interp(length, [15, 220], [0, 100])
                            print(bright, length)
                            sbc.set_brightness(int(bright))

                            bri = np.interp(length, [50, 300], [minR, maxR])
                            briBar = np.interp(length, [50, 300], [400, 150])
                            briPer = np.interp(length, [50, 300], [0, 100])

                            cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 200))
                            cv2.rectangle(img, (50, int(briBar)), (85, 400), (255, 0, 0), cv2.FILLED)
                            cv2.putText(img, f'{int(briPer)} %', (48, 440), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 200), 3)
                            cv2.putText(img, "DSUP", (425, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 200), 3)
                            cv2.putText(img, "______", (422, 39), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

                if label == 'Left':
                    # Display 'Left Hand' on left side of window
                    cv2.putText(img, 'VOLUME', (20, 50),cv2.FONT_HERSHEY_COMPLEX,0.9, (0, 0, 255), 2)
                    if results.multi_hand_landmarks:
                        for hand_in_frame in results.multi_hand_landmarks:
                            mpDraw.draw_landmarks(img, hand_in_frame, mpHands.HAND_CONNECTIONS)
                        for id, lm in enumerate(results.multi_hand_landmarks[0].landmark):
                            h, w, c = img.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            lmList.append([cx, cy])

                        if len(lmList) != 0:
                            x1, y1 = lmList[4][0], lmList[4][1]
                            x2, y2 = lmList[8][0], lmList[8][1]

                            cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
                            cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
                            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3, cv2.FILLED)
                            length = math.hypot(x2 - x1 - 30, y2 - y1 - 30)

                            vol = np.interp(length, [50, 300], [minR, maxR])
                            volBar = np.interp(length, [50, 300], [400, 150])
                            volPer = np.interp(length, [50, 300], [0, 100])

                            cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 200))
                            cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
                            cv2.putText(img, f'{int(volPer)} %', (48, 440), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 200), 3)
                            cv2.putText(img, "DSUP", (425, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 200), 3)
                            cv2.putText(img, "______", (422, 39), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
                            vc.SetMasterVolumeLevel(vol, None)


    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break