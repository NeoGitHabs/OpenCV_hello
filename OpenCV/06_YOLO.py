from ultralytics import YOLO
import cv2
import datetime
import os


model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Camera not found')
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print('Frame not found')
        break

    result = model(frame, stream=True)

    for i in result:
        for n in i.boxes:
            x, y, w, h = map(int, n.xyxy[0])

            conf = float(n.conf[0])
            cls = int(n.cls[0])
            label = model.names[cls]


            if conf < 0.5:
                continue

            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 255), 2)
            cv2.putText(frame, f'{label}{conf:.2f}', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    else:
        cv2.imshow('Frame with objects ', frame)

cap.release()
cv2.destroyAllWindows()
