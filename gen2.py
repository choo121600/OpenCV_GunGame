from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
import cv2



import numpy as np
import sys
import mediapipe as mp
import csv
form_class = uic.loadUiType("./ui/gen2.ui")[0]

class UIToolTab(QWidget,form_class):  ###########################화면구성
    def __init__(self, parent=None):
        super(UIToolTab, self).__init__(parent)
        self.setupUi(self)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        #self.setGeometry(0, 0, 661, 733)
        self.setFixedSize(661, 733)
        self.startUIToolTab()

    def startUIToolTab(self):  ####### 페이지 별 동작함수 구현
        self.ToolTab = UIToolTab(self)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowTitle("UIToolTab")
        self.setCentralWidget(self.ToolTab)

        self.flag = 0
        self.count=0
        self.ToolTab.save_button.clicked.connect(self.save_function)
        self.ToolTab.stop_button.clicked.connect(self.stop_function)
        self.capture = cv2.VideoCapture(0)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,  # 최대 손의 인식 갯수
            min_detection_confidence=0.5,  # 탐지 임계치
            min_tracking_confidence=0.5  # 추적 임계치
        )

        self.show()

        self.start_webcam()
    def save_function(self):

        self.flag=1



    def stop_function(self):
        if self.flag==1:
            self.count =self.count+1
        self.flag = 0


    def start_webcam(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)



    def update_frame(self):

        ret, self.image = self.capture.read()
        self.image = cv2.resize(self.image, (320, 240))
        self.image = cv2.flip(self.image, 1)
        input_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(input_img)  # 웹캠 이미지에서 손의 위치 관절 위치를 탐지한다.
        self.name = self.ToolTab.name.text()
        if result.multi_hand_landmarks is not None:  # 손이 인식 되면

            for res in result.multi_hand_landmarks:  # 인식되 손의 갯수만큼 포문을 돌면서
                self.mp_drawing.draw_landmarks(self.image, res, self.mp_hands.HAND_CONNECTIONS)  # 그림을 그린다.
                joint = np.zeros((21, 3))

                for j, lm in enumerate(res.landmark):  # 21개의 랜드 마크가 들어있는데 한점씩 반복문을 사용하여 처리한다.
                    joint[j] = [lm.x, lm.y, lm.z]
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
                v = v2 - v1  # (20,3) 팔목과 각 손가락 관절 사이의 벡터를 구한다.

                v = v / np.expand_dims(np.linalg.norm(v, axis=1), axis=-1)  # 유닛벡터 구하기 벡터/벡터의 길이

                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                            :]))  # [15,] 유닛벡터를 내적한 값의 아크 코사인을 구하면 각도를 구할 수 있다.

                #angle.append(self.count)
                angle = np.degrees(angle)  # Convert radian to degree
                angle=np.append(angle,np.array(self.count))
                print(angle)
                if self.flag==1:
                    str=self.name+'.csv'
                    f= open(str,'a',encoding='utf-8',newline='')
                    wr=csv.writer(f)
                    wr.writerow(angle)
                    f.close()


        self.displayImage(self.image, 1)

    def stop_webcam(self):
        self.timer.stop()
        # self.capture.release()

    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window == 1:
            self.ToolTab.imglabel.setPixmap(QPixmap.fromImage(outImage))
            self.ToolTab.imglabel.setScaledContents(True)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())