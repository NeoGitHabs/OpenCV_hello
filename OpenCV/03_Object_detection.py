import cv2
import datetime
import numpy as np


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Камера не работает')
    exit()

frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # ширина кадра
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # высота кадра
frame_fps    = float(cap.get(cv2.CAP_PROP_FPS))          # частота кадров

if frame_fps == 0:
    frame_fps = 30.0                                       # fallback если камера не вернула FPS

fourcc = cv2.VideoWriter_fourcc(*'mp4v')                  # кодек для записи mp4

unique_video_name = datetime.datetime.now().strftime("%d_%m_%Y_%H-%M-%S")

save_video = cv2.VideoWriter(                             # инициализация записи видео
    f'Test_video_{unique_video_name}.mp4',
    fourcc,
    frame_fps,
    (frame_width, frame_height)
)

# ----------------------------------------------------------------------------------------------------------------------
print("Детектор движение запущен")
print("[ s ] - сделать снимок движения")
print("[ r ] - сброс фона")
print("[ + ] - увеличить чувствителность")
print("[ - ] - уменьшить чувствителность")
print("[ q ] - выход")

ret, frame1 = cap.read()                                  # первый кадр
ret, frame2 = cap.read()                                  # второй кадр

sensitivity = 10                                          # чувствительность обнаружения движения
min_area    = 500                                         # минимальная площадь объекта

while True:

    diff = cv2.absdiff(                                    # абсолютная разница между кадрами —
        frame1,                                            # всё изменённое становится видимым
        frame2
    )

    gray = cv2.cvtColor(                                   # перевод в оттенки серого —
        diff,                                              # упрощает анализ движения
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(                               # размытие по Гауссу —
        gray,                                              # убирает шумы и мелкие помехи
        (5, 5),
        0
    )

    _, thresh = cv2.threshold(                             # бинаризация: всё ярче sensitivity
        blur,                                              # становится белым (255), остальное — чёрным
        sensitivity,
        255,
        cv2.THRESH_BINARY
    )

    dilated = cv2.dilate(                                  # расширение белых областей —
        thresh,                                            # соединяет разрозненные части объекта
        None,
        iterations=3
    )

    contours, _ = cv2.findContours(                        # поиск контуров объектов
        dilated,                                           # на обработанном изображении
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    display_frame  = frame1.copy()                         # копия кадра для отрисовки (оригинал не трогаем)
    motion_detected = False                                # флаг наличия движения

    for contour in contours:

        area = cv2.contourArea(contour)                    # площадь текущего контура

        if area < min_area:                                # слишком мелкий объект — это шум, пропускаем
            continue

        motion_detected = True                             # движение зафиксировано

        x, y, w, h = cv2.boundingRect(contour)            # координаты ограничивающего прямоугольника

        cv2.rectangle(
            display_frame,
            (x, y),                                        # верхний левый угол
            (x + w, y + h),                                # нижний правый угол
            (0, 255, 0),                                   # зелёный цвет рамки
            2                                              # толщина линии
        )

        cv2.putText(
            display_frame,
            f'Площадь: {int(area)}',                       # площадь объекта над рамкой
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    status = "Обнаружено движение" if motion_detected else "Нет движения"   # текст статуса
    color  = (0, 0, 255)           if motion_detected else (0, 255, 0)      # красный / зелёный

    # Запись видео -----------------------------------------------------------------------------------------------------
    current_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")   # текущее время

    cv2.putText(
        display_frame,
        current_time,                                      # дата и время на кадре
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        status,                                            # статус движения на кадре
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    save_video.write(display_frame)                        # запись кадра в файл
    cv2.imshow('Image', display_frame)                     # отображение окна

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print('Запись остановлен по команде - q')
        break

    elif key == ord('s'):
        filename  = f"Motion_{cv2.getTickCount()}.jpg"
        filename2 = f"Motion2_{cv2.getTickCount()}.jpg"
        cv2.imwrite(filename,  display_frame)              # снимок с рамками
        cv2.imwrite(filename2, thresh)                     # снимок бинарной маски
        print(f"Снимок сохранён: {filename}")
        print(f"Снимок сохранён: {filename2}")

    elif key == ord('r'):
        print("Сброс фонового кадра")
        ret, frame1 = cap.read()                           # обновляем оба опорных кадра
        ret, frame2 = cap.read()

    elif key == ord('+'):
        sensitivity = min(sensitivity + 5, 100)           # не выше 100
        print(f"Чувствительность: {sensitivity}")

    elif key == ord('-'):
        sensitivity = max(sensitivity - 5, 5)             # не ниже 5
        print(f"Чувствительность: {sensitivity}")

    ret, frame1 = cap.read()                              # сдвигаем кадры вперёд
    frame2 = frame1
    ret, frame1 = cap.read()

cap.release()                                              # освобождаем камеру
save_video.release()                                       # закрываем файл записи
cv2.destroyAllWindows()                                    # закрываем все окна