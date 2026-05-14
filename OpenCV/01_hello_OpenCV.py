import cv2
import datetime

# проверка доступных камер
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Камера найдена: индекс {i}")
        cap.release()
    else:
        print(f"Индекс {i} — нет камеры")

cap = cv2.VideoCapture(0) # web_camera
# cap2 = cv2.VideoCapture(1) # front_camera
# cap3 = cv2.VideoCapture(1) # other_camera
# cap4 = cv2.VideoCapture('video_name')

# если не работает камера - выходим
if not cap.isOpened():
    print('Камера не работает')
    exit()

# берём: ширину и высоту видео и ещё FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = float(cap.get(cv2.CAP_PROP_FPS))

if frame_fps == 0:
    frame_fps = 30.0

# запись видео + формат
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

unique_video_name = datetime.datetime.now().strftime("%d_%m_%Y_%H-%M-%S")

# как записать видео: названия, формат, FPS, (ширина и высота кадра)
save_video = cv2.VideoWriter(f'Test_video_{unique_video_name}.mp4', fourcc, frame_fps, (frame_width, frame_height))


while True:
    ret, frame = cap.read() # кадр, данный кадра(видео)

    if not ret:
        print('Кадр не получен')
        break

    current_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    cv2.putText(
        frame, # данные
        current_time, # дата и время, отображаемые поверх кадра в виде текста
        (10, 50), # позиция текста (x, y)
        cv2.FONT_HERSHEY_SIMPLEX, # шрифт
        1, # масштаб шрифта
        (0, 255, 0), # цвет в BGR (зелёный),
        2  # толщина шрифта
    )

    save_video.write(frame) # запись до показа, чтобы сохранить текст на кадре
    cv2.imshow('Image: ', frame) # выводить видео в экран

    if cv2.waitKey(1) & 0xFF == ord("q"): # q — остановить запись
        break


cap.release() # очистить кеш
save_video.release() # сохранить видео
cv2.destroyAllWindows() # закрыть все окна
