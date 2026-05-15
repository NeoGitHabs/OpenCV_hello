import cv2
import datetime
import os

save_dir = 'media'
os.makedirs(save_dir, exist_ok=True)


face_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print('Camera not found')
    exit()

counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print('Frame not found')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    test_data = face_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in test_data:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255, 0), 2)
        cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0, 255), 2)


    date_form = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')


    cv2.putText(frame, date_form, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (203, 192, 255), 4)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        date_type = datetime.datetime.now().strftime('%d%m%Y_%H%M%S')
        image_name = f'{save_dir}/photo_{date_type}.jpg'
        cv2.imwrite(image_name, frame)
        counter += 1
        print(f'Number image {counter}')

        cv2.putText(frame, 'Saved', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('S for take a photo', frame)
        cv2.waitKey(300)
    else:
        cv2.imshow('Video', frame)

cap.release()
cv2.destroyAllWindows()
