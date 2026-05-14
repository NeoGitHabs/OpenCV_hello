import cv2
import datetime


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Сбой камеры")
    exit()

face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # каскад лиц
eye_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")                  # каскад глаз
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")                # каскад улыбок

if smile_cascade.empty() or face_cascade.empty() or eye_cascade.empty():
    print("Ошибки при загрузки каскадов!")
    exit()

detect_smile = True                                        # флаг детекции улыбки
detect_eyes  = True                                        # флаг детекции глаз
save_face    = False                                       # флаг сохранения снимка
face_counter = 0                                           # счётчик обработанных лиц

print("Детекторы запущены!")
print(" [ e ] - вкл/выкл детекцию глаз")
print(" [ s ] - вкл/выкл детекцию улыбки")
print(" [ c ] - сделать снимок")
print(" [ q ] - выход")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Ошибка чтении кадра!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)         # перевод в серый для каскадов

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,                                   # масштаб уменьшения изображения на каждом шаге
        minNeighbors=5,                                    # минимум соседних прямоугольников для подтверждения
        minSize=(30, 30)                                   # минимальный размер лица в пикселях
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),                                        # верхний левый угол
            (x + w, y + h),                                # нижний правый угол
            (255, 0, 0),                                   # синяя рамка
            2
        )

        cv2.putText(
            frame,
            "Face",                                        # подпись над рамкой лица
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        roi_gray  = gray[y:y + h, x:x + w]                # ROI лица (серый) — для глаз и улыбки
        roi_color = frame[y:y + h, x:x + w]               # ROI лица (цветной) — для отрисовки

        face_counter += 1                                  # увеличиваем счётчик лиц

        if detect_eyes:
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(20, 20)
            )

            for (ex, ey, ew, eh) in eyes:
                eye_center = (x + ex + ew // 2, y + ey + eh // 2)  # центр глаза в координатах кадра
                radius = int((ew + eh) / 4)                         # радиус окружности
                cv2.circle(
                    frame,
                    eye_center,
                    radius,
                    (0, 255, 0),                           # зелёный круг вокруг глаза
                    2
                )

        if detect_smile:
            roi_smile = roi_gray[int(h / 2):h, :]         # нижняя половина лица — область рта

            smiles = smile_cascade.detectMultiScale(
                roi_smile,
                scaleFactor=1.8,
                minNeighbors=20,                           # высокий порог — меньше ложных срабатываний
                minSize=(25, 25)
            )

            if len(smiles) > 0:
                cv2.putText(
                    frame,
                    "Smile :)",                             # подпись под рамкой лица
                    (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),                         # жёлтый цвет
                    2
                )

    cv2.putText(
        frame,
        f"Faces: {len(faces)}",                            # количество лиц в кадре
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    status_y = 60                                          # начальная Y-позиция статусных строк

    if detect_eyes:
        cv2.putText(
            frame,
            "Eyes: On",                                    # статус детекции глаз
            (10, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
        status_y += 25                                     # сдвигаем вниз для следующей строки

    if detect_smile:
        cv2.putText(
            frame,
            "Smile: On",                                   # статус детекции улыбки
            (10, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )
        status_y += 25

    cv2.imshow("Face Detection", frame)                    # отображение кадра

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Завершение работы (q)!")
        break
    elif key == ord('e'):
        detect_eyes = not detect_eyes
        print(f"Детекция глаз {'Вкл' if detect_eyes else 'Выкл'}")
    elif key == ord('s'):
        detect_smile = not detect_smile
        print(f"Детекция улыбки {'Вкл' if detect_smile else 'Выкл'}")   # исправлено: было detect_eyes
    elif key == ord('c'):
        filename = f"Screenshot_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.jpg"  # исправлено: незаконченный strftime
        cv2.imwrite(filename, frame)
        print(f"Снимок сохранён: {filename}")

cap.release()
cv2.destroyAllWindows()

print(f"Всего обработано лиц: {face_counter}")
