import cv2
import utils.PoseModule as po
import numpy as np

cap = cv2.VideoCapture(0)
detector = po.PoseDetector()

"""
case1: jap(r, l)
case2: straight(r, l)
case3: upper(r, l)
case4: hook(r, l)
"""

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    h, w, c = img.shape
    img, d = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=True)
    lmList = np.array(lmList)
    # print(lmList)
    cnt = 0
    if d.pose_landmarks is not None:
        joint = []
        # angle_li = np.array()
        for j, lm in enumerate(d.pose_landmarks.landmark):  # 21개의 랜드 마크가 들어있는데 한점씩 반복문을 사용하여 처리한다.
            joint.append([int(lm.x * w), int(lm.y * h), int(lm.z * w)])

        right_elbow = detector.findAngle_p(img, joint[11], joint[13], joint[15], (255, 255, 0), (255, 255, 0),
                                        (255, 255, 0), draw=True)
        left_elbow = detector.findAngle_p(img, joint[12], joint[14], joint[16], (255, 255, 0), (255, 255, 0),
                                        (255, 255, 0), draw=True)

        right_armpit = detector.findAngle_p(img, joint[13], joint[11], joint[23], (255, 255, 0), (255, 255, 0),
                                        (255, 255, 0), draw=True)
        left_armpit = detector.findAngle_p(img, joint[14], joint[12], joint[24], (255, 255, 0), (255, 255, 0),
                                        (255, 255, 0), draw=True)

        right_shoulder = detector.findAngle_p(img, joint[13], joint[11], joint[12], (255, 255, 0), (255, 255, 0),
                                        (255, 255, 0), draw_up=True)
        left_shoulder = detector.findAngle_p(img, joint[14], joint[12], joint[11], (255, 255, 0), (255, 255, 0),
                                        (255, 255, 0), draw_up=True)
        # print(right_elbow, left_elbow, right_shoulder, left_shoulder, right_armpit, left_armpit)
        angle_li = list(right_elbow)
        #
        # print(angle_li)
    cv2.putText(img, text=str(cnt), org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                color=(255, 255, 255), thickness=2)

    cv2.imshow('dict', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if not ret:
        break