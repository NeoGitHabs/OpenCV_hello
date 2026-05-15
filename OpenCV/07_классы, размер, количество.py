from ultralytics import YOLO
import cv2
import time

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Camera not found')
    exit()

classes = ['person', 'cat', 'dog', 'car']

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:

    fps_start = time.time()

    ret, frame = cap.read()

    if not ret:
        print('Frame not found')
        break

    results = model(frame, conf=0.5)

    person_count = 0

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            if label not in classes:
                continue

            if label == 'person':
                person_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f'{label} {round(conf * 100, 1)}%',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    fps = 1 / (time.time() - fps_start)

    cv2.putText(
        frame,
        f'FPS: {round(fps, 1)}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f'Person count: {person_count}',
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.imshow('Frame with objects', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
