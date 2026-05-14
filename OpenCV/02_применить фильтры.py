import cv2
import datetime


filters_video = 'normal'
print('1 - оригинал')
print('2 - чёрно-белый')
print('3 - размытие')
print('4 - контур')
print('5 - выйти')

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

while True:
    ret, frame = cap.read()

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

    # Применение фильтра в зависимости от выбора пользователя
    if filters_video == 2:
        # Чёрно-белый: конвертируем BGR → GRAY, затем обратно GRAY → BGR,
        # чтобы сохранить трёхканальный формат для VideoWriter
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filter_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif filters_video == 3:
        # Размытие: Гауссово размытие с ядром 11×11(нечётный) и сигмой 0 (авторасчёт)
        filter_frame = cv2.GaussianBlur(frame, (11, 11), 0)

    elif filters_video == 4:
        # Контур: конвертируем в оттенки серого, применяем алгоритм Canny
        # (пороги 50 и 100), затем возвращаем в BGR для совместимости
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 100)
        filter_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    else:  # filters_video == 1 — оригинал без изменений
        filter_frame = frame

    save_video.write(filter_frame)
    cv2.imshow('Image: ', filter_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('2'):
        filters_video = 2
    elif key == ord('3'):
        filters_video = 3
    elif key == ord('4'):
        filters_video = 4
    elif key == ord('5'):
        break


cap.release()
save_video.release()
cv2.destroyAllWindows()


# import streamlit as st
#
# st.title('Веб-камера с фильтрами')
#
# filter_choice = st.radio(
#     'Выберите фильтр:',
#     options=[1, 2, 3, 4],
#     format_func=lambda x: {1: 'Оригинал', 2: 'Чёрно-белый', 3: 'Размытие', 4: 'Контур'}[x]
# )
#
# run = st.checkbox('Запустить камеру')
# frame_placeholder = st.empty()
#
# cap = cv2.VideoCapture(0)
#
# while run:
#     ret, frame = cap.read()
#
#     if not ret:
#         st.error('Кадр не получен')
#         break
#
#     current_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
#     cv2.putText(frame, current_time, (10, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#     if filter_choice == 2:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         filter_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#     elif filter_choice == 3:
#         filter_frame = cv2.GaussianBlur(frame, (11, 11), 0)
#     elif filter_choice == 4:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, 50, 100)
#         filter_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#     else:
#         filter_frame = frame
#
#     # BGR → RGB для корректного отображения в Streamlit
#     frame_placeholder.image(cv2.cvtColor(filter_frame, cv2.COLOR_BGR2RGB))
#
# cap.release()
