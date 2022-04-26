"""
Pose Module
By: Computer Vision Zone
Website: https://www.computervision.zone/
"""
import cv2
import mediapipe as mp
import math
import numpy as np

class PoseDetector:
    """
    Estimates Pose points of a human body using the mediapipe library.
    """

    def __init__(self, mode=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param upBody: Upper boy only flag
        :param smooth: Smoothness Flag
        :param detectionCon: Minimum Detection Confidence Threshold
        :param trackCon: Minimum Tracking Confidence Threshold
        """

        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS,
                                           self.mpDraw.DrawingSpec(color=(45,66,230),thickness=2,circle_radius=2),#포인트
                                           self.mpDraw.DrawingSpec(color=(245,245,230),thickness=2,circle_radius=2))#링크


        return img,self.results

    def findPosition(self, img, draw=True, bboxWithHands=False):
        self.lmList = []
        self.bboxInfo = {}
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                self.lmList.append([id, cx, cy, cz])

            # Bounding Box
            ad = abs(self.lmList[12][1] - self.lmList[11][1]) // 2
            if bboxWithHands:
                x1 = self.lmList[16][1] - ad
                x2 = self.lmList[15][1] + ad
            else:
                x1 = self.lmList[12][1] - ad
                x2 = self.lmList[11][1] + ad

            y2 = self.lmList[29][2] + ad
            y1 = self.lmList[1][2] - ad
            bbox = (x1, y1, x2 - x1, y2 - y1)
            cx, cy = bbox[0] + (bbox[2] // 2), \
                     bbox[1] + bbox[3] // 2

            self.bboxInfo = {"bbox": bbox, "center": (cx, cy)}

            if draw:
                cv2.rectangle(img, bbox, (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmList, self.bboxInfo

    def findAngle_p(self, img, p1, p2, p3, color0, color1, color2, draw=False, draw_up=False):
        v1 = p3 - p2
        v2 = p1 - p2
        v = v2 - v1
        v = v / np.expand_dims(np.linalg.norm(v, axis=1), axis=-1)

        x1, y1, z1 = p1
        x2, y2, z2 = p2
        x3, y3, z3 = p3

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        #if angle >90 : angle=180-angle
        #angle=abs(angle)
        # Draw
        if draw_up:
            cv2.putText(img, str(int(angle)) + "@", (x2 - 50, y2 - 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, color2, 2)

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), color0, 1)
            cv2.line(img, (x3, y3), (x2, y2), color0, 1)
            cv2.circle(img, (x1, y1), 5, color1, cv2.FILLED)
            cv2.circle(img, (x1, y1), 10, color1, 2)
            cv2.circle(img, (x2, y2), 5, color1, cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, color1, 2)
            cv2.circle(img, (x3, y3), 5, color1, cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, color1, 2)
            cv2.putText(img, str(int(angle))+"@", (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, color2, 2)
        return angle



    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1, z1 = self.lmList[p1][1:]
        x2, y2, z2 = self.lmList[p2][1:]
        x3, y3, z3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        #if angle >90 : angle=180-angle
        #angle=abs(angle)
        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (0, 255, 255), 3)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle))+"@", (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1,z1 = self.lmList[p1][1:]
        x2, y2 ,z1= self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        cv2.putText(img, str(int(length))+"D", (cx + 5, cy + 5),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def angleCheck(self, myAngle, targetAngle, addOn=20):
        return targetAngle - addOn < myAngle < targetAngle + addOn


def main():
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False)
        if bboxInfo:
            center = bboxInfo["center"]
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
