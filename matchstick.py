import cv2
import mediapipe as mp
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

device = 0 
#つかんでいるか
moving = False
N = 22
matchstick = [None] * N
matchstick_points = [None] * N
#表示するマッチ棒の選択
init_box = [1] * N
offflags = [3,7,10,15,17]

for n in offflags:
    init_box[n] = 0
matchstick = init_box 
#前の座標
prepoint = None
#つまんでいるかどうかのフラグ
pinch_flag = False


def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)

    return frame_now

#マッチ棒とカメラ画像を合成
def combi(img):
    global matchstick, matchstick_point
    cv2.putText(img, "Let's move one and get the equation right!", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    bimg  = cv2.imread('./imgs/matchstick.jpg')
    bimg = cv2.resize(bimg, dsize=None, fx=0.2, fy=0.2)
    
    white = np.ones((img.shape), dtype=np.uint8) * 255 #カメラ画像と同じサイズの白画像
    #数字の配置
    x = 20
    y = 150
    for i in range(3):
        for j in range(2):
            #マッチ棒の向きの変更
            mimg = np.rot90(bimg,j)
            xd = mimg.shape[1]
            yd = mimg.shape[0]
            #マッチ棒が縦向きの場合  
            #左上から時計回り (0~N) 
            if j == 0:
                if matchstick[i*7] != 0:
                    white[y:yd+y,x:xd+x] = mimg
                matchstick_points[i*7] = [x,y,xd+x,y+yd]
                x += yd
                if matchstick[i*7+1] != 0:
                    white[y:yd+y,x:xd+x] = mimg
                matchstick_points[i*7+1] = [x,y,xd+x,y+yd]
                y += yd
                if matchstick[i*7+2] != 0:
                    white[y:yd+y,x:xd+x] = mimg
                matchstick_points[i*7+2] = [x,y,xd+x,y+yd]
                x -= yd
                if matchstick[i*7+3] != 0:
                    white[y:yd+y,x:xd+x] = mimg
                matchstick_points[i*7+3] = [x,y,xd+x,y+yd]
                y -= yd
            #マッチ棒が横向きの場合
            #上から順に
            if j==1:
                if matchstick[i*7+4] != 0:
                    white[y:yd+y,x:xd+x] = mimg
                matchstick_points[i*7+4] = [x,y,xd+x,y+yd]
                y += xd
                if matchstick[i*7+5] != 0: 
                    white[y:yd+y,x:xd+x] = mimg
                matchstick_points[i*7+5] = [x,y,xd+x,y+yd]
                y += xd
                if matchstick[i*7+6] != 0: 
                    white[y:yd+y,x:xd+x] = mimg
                matchstick_points[i*7+6] = [x,y,xd+x,y+yd]
                y -= 2*xd
        mimg = bimg
        x += mimg.shape[0] + 135

    #記号の配置
    #固定 (- , =)
    #(-)の配置
    mimg = np.rot90(bimg,1)
    xd = mimg.shape[1]
    yd = mimg.shape[0] 
    x,y = 40+bimg.shape[0], 150+bimg.shape[0]
    white[y:yd+y,x:xd+x] = mimg

    #(=)の配置
    x += x+bimg.shape[0]
    y -= 20
    white[y:yd+y,x:xd+x] = mimg
    y += 40
    white[y:yd+y,x:xd+x] = mimg
    
    #(|)の配置
    mimg = bimg
    xd = mimg.shape[1]
    yd = mimg.shape[0] 
    x, y = 80+bimg.shape[0], 150+(bimg.shape[0]//2)
    if matchstick[21] != 0: 
        white[y:yd+y,x:xd+x] = mimg
    matchstick_points[21] = [x,y,xd+x,y+yd]
    dwhite = white
    img[dwhite!=[255, 255, 255]] = dwhite[dwhite!=[255, 255, 255]]
    return img

#つまんでいるかどうかの判定
def pinch(img, point):
    global matchstick, moving, prepoint,pinch_flag
    
    #つまんだ座標
    points = [(point[0][0]+point[1][0])//2,(point[0][1]+point[1][1])//2]
    #つまんでいると判断される場合
    if abs(point[0][0]-point[1][0])<=15 and abs(point[0][1]-point[1][1])<=25:
        cv2.circle(img, (points[0], points[1]), 7, (0, 255, 255), 3)
        
        #マッチ棒があるか
        #matchstick_point[x_start,y_start,x_end,y_end]
        for i, matchstick_point in enumerate(matchstick_points):
            if moving==False and matchstick_point[0] <= points[0] <= matchstick_point[2]:
                if matchstick_point[1] <= points[1] <= matchstick_point[3]:
                    if matchstick[i] != 0:
                        matchstick[i] = 0
                        # print(i)
                        moving = True
                        pinch_flag = True
    #マッチ棒をとり、指を離した場合
    elif moving == True:
        for i, matchstick_point in enumerate(matchstick_points):
            if matchstick_point[0] <= prepoint[0] <= matchstick_point[2]:
                if matchstick_point[1] <= prepoint[1] <= matchstick_point[3]:
                    if matchstick[i] != 1:
                        matchstick[i] = 1
                        # print(i)
                        moving = False
                        pinch_flag = False
    prepoint = points    
    return pinch_flag

#問題の正解と同じ配置になっているかの判定を行う
def correct():
    flag = False
    correct_box = (1,1,1,0,1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,1,1,0)
    if tuple(matchstick)==correct_box:
        flag = True
    return flag

#マッチ棒を移動させる
def move(img, landmarks):
    global matchstick
    image_width, image_height = img.shape[1], img.shape[0]
    landmark_point = []
    mimg  = cv2.imread('./imgs/matchstick.jpg')
    mimg = cv2.resize(mimg, dsize=(12,100))
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
    if correct():
        cv2.putText(img, "Great!", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)


def main():
    global device

    cap = cv2.VideoCapture(device)
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