from asyncio import set_event_loop_policy
from tkinter import image_names
import cv2
from matplotlib import offsetbox
from matplotlib.pyplot import box
import mediapipe as mp
import time
import numpy as np

from numpy import imag

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

device = 0 
#つかんでいるか
moving = False
#縦x横
N = 4
panels = [None] * N * N
panel_points = [None] * N * N
#前の座標
prepoint = None
#つまんでいるかどうかのフラグ
pinch_flag = False
pinch_points = [None] * 10
pinches = [1] * 10

def set_up(onbox):
    init_box = [0] * N * N
    for n in onbox:
        init_box[n] = 1
    return init_box 
onbox = [0, 3, 6, 9, 11, 14]
panels = set_up(onbox)


def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)

    return frame_now

def dataset():
    data = [[None]*N]*N 
    dataset = []
    data[0][0] = cv2.imread('./imgs/u.jpg')
    data[0][1] = cv2.imread('./imgs/ma.jpg')
    data[0][2] = cv2.imread('./imgs/i.jpg')
    data[0][3] = cv2.imread('./imgs/shi.jpg')
    for i in range(4):
        dataset.append(data[0][i])

    data[1][0] = cv2.imread('./imgs/mi.jpg')
    data[1][1] = cv2.imread('./imgs/chi.jpg')
    data[1][2] = cv2.imread('./imgs/no.jpg')
    data[1][3] = cv2.imread('./imgs/ri.jpg')

    for i in range(4):
        dataset.append(data[1][i])

    data[2][0] = cv2.imread('./imgs/shi.jpg')
    data[2][1] = cv2.imread('./imgs/ro.jpg')
    data[2][2] = cv2.imread('./imgs/u.jpg')
    data[2][3] = cv2.imread('./imgs/to.jpg')

    for i in range(4):
        dataset.append(data[2][i])
    
    data[3][0] = cv2.imread('./imgs/ki.jpg')
    data[3][1] = cv2.imread('./imgs/ba.jpg')
    data[3][2] = cv2.imread('./imgs/shi.jpg')
    data[3][3] = cv2.imread('./imgs/ri.jpg')

    for i in range(4):
        dataset.append(data[3][i])
    d = []
    for i, img in enumerate(dataset):
        d.append(cv2.resize(img,dsize=(70,70)))
    return d

#マッチ棒とカメラ画像を合成
def combi(img):
    global panels, pinches, panel_points, pinch_points
    imgs = dataset()
    white = np.ones((img.shape), dtype=np.uint8) * 255 #カメラ画像と同じサイズの白画像
    #文字の配置 
    #左上
    firstpoint = (300,100)
    for i in range(N*N):
        j = i // 4
        y = firstpoint[1] + j*imgs[i].shape[0]
        k = i % 4
        x = firstpoint[0] + k*imgs[i].shape[1]
        panel_points[i] = [x, y, x+imgs[i].shape[1], y+imgs[i].shape[0]]  #panel_points = [x_start, y_start, x_end, y_end]
        cv2.rectangle(img, (x,y), (x+imgs[i].shape[1], y+imgs[i].shape[0]), color=(0, 0, 0), thickness=2)
        if panels[i] == 1:
            white[y:y+imgs[i].shape[0],x:x+imgs[i].shape[1]] = imgs[i]
    secondpoint = (250, 500)
    pinch_list = [i for i in range (N*N) if i not in onbox]
    pinch_images = []
    for pimg in pinch_list:
        pinch_images.append(imgs[pimg])
    for i, pinch_img in enumerate(pinch_images):
        j = i // 5
        y = secondpoint[1] + j*pinch_img.shape[0]
        k = i % 5
        x = secondpoint[0] + k*pinch_img.shape[1]
        pinch_points[i] = [x, y, x+pinch_img.shape[1], y+pinch_img.shape[0]]  #panel_points = [x_start, y_start, x_end, y_end]
        cv2.rectangle(img, (x,y), (x+pinch_img.shape[1], y+pinch_img.shape[0]), color=(0, 0, 0), thickness=2)
        if pinches[i] == 1:
            white[y:y+pinch_img.shape[0],x:x+pinch_img.shape[1]] = pinch_img

    
    dwhite = white
    img[dwhite!=[255, 255, 255]] = dwhite[dwhite!=[255, 255, 255]]
    return img

#つまんでいるかどうかの判定
def pinch(img, point):
    global panels, moving, prepoint,pinch_flag, pinches
    
    #つまんだ座標
    points = [(point[0][0]+point[1][0])//2,(point[0][1]+point[1][1])//2]
    #つまんでいると判断される場合
    if abs(point[0][0]-point[1][0])<=15 and abs(point[0][1]-point[1][1])<=25:
        cv2.circle(img, (points[0], points[1]), 7, (0, 255, 255), 3)
        
        #matchstick_point[x_start,y_start,x_end,y_end]
        for i, panel_point in enumerate(pinch_points):
            if moving==False and panel_point[0] <= points[0] <= panel_point[2]:
                if panel_point[1] <= points[1] <= panel_point[3]:
                    if  pinches != 0:
                        pinches[i] = 0
                        # print(i)
                        moving = True
                        pinch_flag = True
    #マッチ棒をとり、指を離した場合
    elif moving == True:
        for i, panel_point in enumerate(panel_points):
            if panel_point[0] <= prepoint[0] <= panel_point[2]:
                if panel_point[1] <= prepoint[1] <= panel_point[3]:
                    if panels[i] != 1:
                        panels[i] = 1
                        # print(i)
                        moving = False
                        pinch_flag = False
    prepoint = points    
    return pinch_flag

#問題の正解と同じ配置になっているかの判定を行う
def correct():
    pass
#マッチ棒を移動させる
def move(img, landmarks):
    global panels
    image_width, image_height = img.shape[1], img.shape[0]
    landmark_point = []
    mimg  = cv2.imread('./imgs/u.jpg')
    mimg = cv2.resize(mimg, dsize=(70,70))
    white = np.ones((img.shape), dtype=np.uint8) * 255

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        # 画面上の座標位置へ変換
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])

    x = mimg.shape[1]//2
    y = mimg.shape[0]//2
    point = [landmark_point[4],landmark_point[8]]
    flag = pinch(img,point)
    #つかんでいるマッチ棒の表示
    if flag:
        if landmark_point[8][1] >= y and landmark_point[8][1]<=img.shape[0]-y:
            if landmark_point[8][0] >= x and landmark_point[8][0]<=img.shape[1]-x:
                white[landmark_point[8][1]-y:landmark_point[8][1]+y,landmark_point[8][0]-x:landmark_point[8][0]+x] = mimg
                dwhite = white
                img[dwhite!=[255, 255, 255]] = dwhite[dwhite!=[255, 255, 255]]

    #正解に合わせてメッセージを表示
     #if correct():
      #  cv2.putText(img, "Great!", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)


def main():
    global device

    cap = cv2.VideoCapture(device)
    ret = cap.set(3, 1920)
    ret = cap.set(4, 1080)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


    print("Size:", ht, "x", wt, "/Fps: ", fps)

    start = time.perf_counter()
    frame_prv = -1

    cv2.namedWindow('quiz', cv2.WINDOW_NORMAL)
    with mp_hands.Hands(
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

            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = hands.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
           
            combi(frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    move(frame, hand_landmarks)
            cv2.imshow('quiz', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == '__main__':
    main()