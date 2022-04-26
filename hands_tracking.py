import cv2
import mediapipe as mp
import numpy as np
from settings import *

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#### TASK ####
"""
1. 손을 인식하고, 손의 좌표를 출력
2. 손가락 접힘 감지
"""


gesture = {0:'Def', 1:'Shot', 2:'Reload'}
file=np.genfromtxt('gun.csv',delimiter=',')

angle=file[:,:-1].astype(np.float32)#0번인덱스 부터 마지막 인덱스(-1) 전까지 잘라라
label=file[:,-1].astype(np.float32)#마지막 인덱스(-1)만 가져와라


knn=cv2.ml.KNearest_create()#knn모델을 초기화
knn.train(angle,cv2.ml.ROW_SAMPLE,label)#knn 학습

class HandTracking:
    def __init__(self):
        self.hand_tracking = mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=1,
        )
        self.hand_x = 0
        self.hand_y = 0
        self.results = None
        self.hand_None = False

    def hand_location(self, img):
        h, w, c = img.shape
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hand_tracking.process(imgRGB)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.shot = False
        self.refill = False

        if self.results.multi_hand_landmarks:
            for res in self.results.multi_hand_landmarks:#인식되 손의 갯수만큼 포문을 돌면서
                mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS)# 그림을 그린다.
                joint=np.zeros((21,3))
                x, y = res.landmark[4].x, res.landmark[4].y
                self.hand_x = int(x*SCREEN_WIDTH)
                self.hand_y = int(y*SCREEN_HEIGHT)

                for j,lm in enumerate(res.landmark):#21개의 랜드 마크가 들어있는데 한점씩 반복문을 사용하여 처리한다.
                    joint[j]=[lm.x, lm.y, lm.z]
                    
                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                    v=v2-v1#(20,3) 팔목과 각 손가락 관절 사이의 벡터를 구한다.

                    v=v/np.expand_dims(np.linalg.norm(v,axis=1),axis=-1)#유닛벡터 구하기 벡터/벡터의 길이

                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,] 유닛벡터를 내적한 값의 아크 코사인을 구하면 각도를 구할 수 있다.
                    angle = np.degrees(angle)  # Convert radian to degree
                    angle=np.expand_dims(angle.astype(np.float32),axis=0)#float32 차원증가 keras or tensor 머신러닝 모델에 넣어서 추론할 때는 항상 맨앞 차원 하나를 추가한다.
                    _,results,_,_=knn.findNearest(angle,3)#statue,result,인접값,거리
                    # print(results)
                    idx=int(results[0][0])
                    gesture_name=gesture[idx]

                    # print(gesture_name)
                    if idx == 1:
                        self.shot = True
                        cv2.putText(img, text=gesture_name, org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,color=(255, 255, 255), thickness=2)
                        # print((x, y))
                    elif idx == 2:
                        self.refill = True
                        cv2.putText(img, text=gesture_name, org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=2, color=(255, 255, 255), thickness=2)
        return img

    def get_hand_center(self):
        return (self.hand_x, self.hand_y)
