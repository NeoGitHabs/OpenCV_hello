from ultralytics import YOLO
import streamlit as st
import cv2
import time


st.title('OpenCV + YOLO + Streamlit')

st.sidebar.header('Настройки')

model_name = st.sidebar.selectbox(
    'Выберите модель:',
    ['yolov8n.pt', 'yolo11n.pt']
)

count_conf = st.sidebar.slider(
    'Минимальная точность:',
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

start_button = st.sidebar.button('Запустить')

model = YOLO(model_name)

print_image = st.image([])

if start_button:

    st.success('Модель загружена')

    cap = cv2.VideoCapture(0)

    prev_time = time.time()

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            st.error('Камера не работает')
            break

        results = model(frame, conf=count_conf)

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])
                label = model.names[cls]
                conf = round(float(box.conf[0]), 2)

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    f'{label} {conf * 100:.0f}%',
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(
            frame,
            f'FPS: {fps:.1f}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print_image.image(rgb_frame)

    cap.release()
    st.info('Видео остановлено')

else:
    st.info('Нажмите кнопку запуска')
