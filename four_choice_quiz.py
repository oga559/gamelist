import cv2
import mediapipe as mp
import time
import numpy as np

mp_hands = mp.solutions.hands

device = 0 # カメラのデバイス番号
correct_xy = [442, 404]
incorrect_xy = [[200, 406], [200, 76], [443, 72]]

#カメラ画像とイラストの合成
def combi(frame_img):
    #写真を変数に格納
    question_img = cv2.imread('./imgs/four_choice/hackathon_quiz_q.jpg')
    ans1_img  = cv2.imread('./imgs/four_choice/hackathon_quiz_a1.jpg')
    ans2_img  = cv2.imread('./imgs/four_choice/hackathon_quiz_a2.jpg')
    ans3_img  = cv2.imread('./imgs/four_choice/hackathon_quiz_a3.jpg')
    ans4_img  = cv2.imread('./imgs/four_choice/hackathon_quiz_a4.jpg')
    # サイズの変更
    question_img = cv2.resize(question_img, dsize=None, fx=0.7, fy=0.7)
    ans1_img  = cv2.resize(ans1_img, dsize=None, fx=0.7, fy=0.7)
    ans2_img = cv2.resize(ans2_img, dsize=None, fx=0.7, fy=0.7)
    ans3_img = cv2.resize(ans3_img, dsize=None, fx=0.7, fy=0.7)
    ans4_img = cv2.resize(ans4_img, dsize=None, fx=0.7, fy=0.7)

    white = np.ones((frame_img.shape), dtype=np.uint8) * 255 #カメラ画像と同じサイズの白画像

    # 選択肢の貼り付け
    y_end = frame_img.shape[0] # ウィンドウのy座標の終端
    x_end = frame_img.shape[1] # ウィンドウのx座標の終端

    white[y_end - ans1_img.shape[0] : y_end, 0 : ans1_img.shape[1]] = ans1_img # print(ans1_img.shape[1], y_end - ans1_img.shape[0])
    white[0 : ans2_img.shape[0], 0 : ans2_img.shape[1]] = ans2_img # print(ans2_img.shape[1], ans2_img.shape[0])
    white[0 : ans3_img.shape[0], x_end - ans3_img.shape[1] : x_end] = ans3_img # print(x_end - ans3_img.shape[1], ans3_img.shape[0])
    white[y_end - ans4_img.shape[0] : y_end, x_end - ans4_img.shape[1] : x_end] = ans4_img # print(x_end - ans4_img.shape[1], y_end - ans4_img.shape[0])


    # 問題文の貼り付け
    q_y_start = (y_end - question_img.shape[0]) // 2
    q_x_start = (x_end - question_img.shape[1]) // 2
    white[q_y_start:q_y_start + question_img.shape[0], q_x_start:q_x_start + question_img.shape[1]] = question_img

    # カメラ画像に貼り付ける
    dwhite = white
    frame_img[dwhite!=[255, 255, 255]] = dwhite[dwhite!=[255, 255, 255]]

    return frame_img

def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)

    return frame_now

# 指のランドマークを表示
def drawFingertip(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        # 画面上の位置に変換
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])

    cv2.circle(image, (landmark_point[8][0], landmark_point[8][1]), 7, (0, 0, 255), -1)

    judge_correct(landmark_point[8][0], landmark_point[8][1], image)

def judge_correct(finger_point_x, finger_point_y, image):
    if finger_point_x > correct_xy[0] and  finger_point_y > correct_xy[1]:
        cv2.putText(image, "Correct!", (200, 370), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
    elif finger_point_x < incorrect_xy[0][0] and  finger_point_y > incorrect_xy[0][1]:
        cv2.putText(image, "Incorrect", (200, 370), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
    elif finger_point_x < incorrect_xy[1][0] and  finger_point_y < incorrect_xy[1][1]:
        cv2.putText(image, "Incorrect", (200, 370), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
    elif finger_point_x > incorrect_xy[2][0] and  finger_point_y < incorrect_xy[2][1]:
        cv2.putText(image, "Incorrect", (200, 370), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)


def main():
    # For webcam input:
    global device

    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Size:", ht, "x", wt, "/Fps: ", fps)

    start = time.perf_counter()
    frame_prv = -1

    cv2.namedWindow('four choice quiz', cv2.WINDOW_NORMAL)
    with mp_hands.Hands(
        #検出する手の数(1~2)
        max_num_hands = 1,
        #信用度の設定(0~1)
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            frame_now=getFrameNumber(start, fps)
            if frame_now == frame_prv:
                continue
            frame_prv = frame_now

            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue


            #反転処理
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = hands.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            combi(frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawFingertip(frame, hand_landmarks)
            cv2.imshow('four choice quiz', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == '__main__':
    main()