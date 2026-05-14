import cv2
import datetime
import numpy as np


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Камера не работает')
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = float(cap.get(cv2.CAP_PROP_FPS))

if frame_fps == 0:
    frame_fps = 30.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

unique_video_name = datetime.datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
save_video = cv2.VideoWriter(f'Test_video_{unique_video_name}.mp4', fourcc, frame_fps, (frame_width, frame_height))

# ----------------------------------------------------------------------------------------------------------------------
print("Детектор движение запущен")
print("[ s ] - сделать снимок движения")
print("[ r ] - сброс фона")
print("[ + ] - увеличить чувствителность")
print("[ - ] - уменьшить чувствителность")
print("[ q ] - выход")

ret, frame1 = cap.read()  # считываем первый кадр с камеры
ret, frame2 = cap.read()  # считываем второй кадр

sensitivity = 10  # чувствительность обнаружения движения
min_area = 500    # минимальная площадь объекта

while True:

    # вычисляем абсолютную разницу между двумя кадрами
    # всё, что изменилось между кадрами — будет выделено
    diff = cv2.absdiff(
        frame1,
        frame2
    )

    # переводим изображение в оттенки серого
    # так проще анализировать движение
    gray = cv2.cvtColor(
        diff,
        cv2.COLOR_BGR2GRAY
    )

    # размываем изображение
    # это уменьшает шумы и мелкие помехи
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # превращаем изображение в чёрно-белое
    # всё, что ярче sensitivity — становится белым
    _, thresh = cv2.threshold(
        blur,
        sensitivity,
        255,
        cv2.THRESH_BINARY
    )

    # расширяем белые области
    # помогает соединить разорванные части объекта
    dilated = cv2.dilate(
        thresh,
        None,
        iterations=3
    )

    # ищем контуры объектов на обработанном изображении
    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # создаём копию кадра для отображения
    # чтобы не изменять оригинальный frame1
    display_frame = frame1.copy()

    # флаг наличия движения
    motion_detected = False

    # перебираем все найденные контуры
    for contour in contours:

        # вычисляем площадь контура
        area = cv2.contourArea(contour)

        # игнорируем слишком маленькие объекты
        # это помогает убрать шумы
        if area < min_area:
            continue

        # если найден подходящий объект — фиксируем движение
        motion_detected = True

        # получаем координаты прямоугольника вокруг объекта
        x, y, w, h = cv2.boundingRect(contour)

        # рисуем зелёную рамку вокруг движущегося объекта
        cv2.rectangle(
            display_frame,
            (x, y),  # верхний левый угол
            (x + w, y + h),  # нижний правый угол
            (0, 255, 0),  # цвет рамки (зелёный)
            2  # толщина линии
        )

        # выводим площадь объекта над рамкой
        cv2.putText(
            display_frame,
            f'Площадь: {int(area)}',
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # текст статуса системы
    status = "Обнаружено движение" if motion_detected else "Нет движения"

    # цвет статуса:
    # красный — движение есть
    # зелёный — движения нет
    color = (0, 0, 255) if motion_detected else (0, 255, 0)

    # Запись видео -----------------------------------------------------------------------------------------------------
    current_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    # рисуем дату и время
    cv2.putText(
        display_frame,
        current_time,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # выводим статус движения
    cv2.putText(
        display_frame,
        status,
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    # записываем кадр в видео
    save_video.write(display_frame)

    # показываем окно
    cv2.imshow('Image', display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print('Запись остановлен по команде - q')
        break
    elif key == ord('s'):
        filename = f"Motion_{cv2.getTickCount()}.jpg"
        filename2 = f"Motion2_{cv2.getTickCount()}.jpg"

        cv2.imwrite(filename, display_frame)
        cv2.imwrite(filename2, thresh)

        print(f"Снимок сохранён: {filename}")
        print(f"Снимок сохранён: {filename2}")

    elif key == ord('r'):
        print("Сброс фонового кадра")
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
    elif key == ord('+'):
        sensitivity = min(
            sensitivity + 5,
            100
        )
        print(f"Чувствительность: {sensitivity}")
    elif key == ord('-'):
        sensitivity = max(
            sensitivity - 5,
            5
        )
        print(f"Чувствительность: {sensitivity}")

cap.release()
save_video.release()
cv2.destroyAllWindows()
