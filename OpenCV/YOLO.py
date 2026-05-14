import cv2
import datetime
import os

import time


image_path = 'media'

os.makedirs(image_path , exist_ok=True)

video_path = 'video'

os.makedirs(video_path , exist_ok=True)

model = YOLO('yolov8n.pt')


cap = cv2.VideoCapture('dog_video2.mp4')

if not cap:
    print('Camera not found')
    exit()

counter = 0
classes = ['car' , 'airplane' , 'cat' , 'dog']
cap.set(cv2.CAP_PROP_FRAME_WIDTH , 640)
cap.set(cv2.CAP_PROP_GIGA_FRAME_SENS_HEIGH , 480)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = float(cap.get(cv2.CAP_PROP_FPS))

if frame_fps == 0:
    frame_fps = 30.0
dtype = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
type_video = cv2.VideoWriter_fourcc(*"mp4v")

date_form = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
while True:
    fps_start = time.time()

    ret , frame = cap.read()

    if not ret:
        print('Frame not found')
        break

    result = model(frame , conf = 0.3)

    boxes = result[0].boxes
    for n in boxes:
        cls = int(n.cls[0])
        label = model.names[cls]
        conf = float(n.conf[0])

        if label not in classes:
            continue

        if conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, n.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f'{label} {conf * 100:.2f} %', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    fps_end = time.time()
    fps = 1 / (fps_end - fps_start)
    cv2.putText(frame, f'FPS : {round(fps, 2)}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, 'II 8', (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Date Time : {date_form}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Frame' , frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        dtype = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
        image_name = f'{image_path}/photo_{dtype}.jpg'
        cv2.imwrite(image_name , frame)
        counter += 1
        print(f'Картинка номер: {counter}')
        cv2.putText(frame , 'Saved' , (50 ,30) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0,0,0))
        cv2.imshow('S for take photo',frame)
        cv2.waitKey(300)

    elif key == ord('v'):
        video_name = f'{video_path}/video_{dtype}.mp4'
        video = cv2.VideoWriter(video_name , type_video , frame_fps , (frame_width , frame_height))
        video.write(frame)

    elif key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
