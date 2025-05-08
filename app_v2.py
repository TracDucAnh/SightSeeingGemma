import streamlit as st
import cv2
import requests
import time
import io
from PIL import Image as PIL_Image
import numpy as np
import pyttsx3
import threading

# Các biến toàn cục (URL server, khoảng thời gian chụp)
NGROK_URL = "https://076a-34-124-132-116.ngrok-free.app"
API_ENDPOINT = f"{NGROK_URL}/describe_image"
CAPTURE_INTERVAL = 5  # giây
MARK_SPEAK = 0

st.title("SightSeeing Genma")

camera_placeholder = st.empty()
description_placeholder = st.empty()
status_placeholder = st.empty()

default_prompt = """Bạn đang nhìn ảnh từ camera trước mặt người dùng khiếm thị.

                                 Hãy mô tả nội dung ảnh thành 6 dòng ngắn, mỗi dòng chỉ 1–2 câu. Ngôn ngữ đơn giản, dễ hiểu, không dùng từ hoa mỹ. Ghi đúng theo hướng nhìn của người dùng, không nhắc lại cấu trúc yêu cầu. Nếu không thấy rõ vật thể, hãy ghi “Không thể xác định”.

                                 1. Cảnh báo nguy hiểm: một trong các mức sau và mô tả ngắn lý do nếu có hoặc lời khuyên cho người khiếm thị nếu có:
                                 - Báo cáo an toàn.
                                 - Báo cáo nguy hiểm.
                                 - Báo cáo không có nguy hiểm.
                                 - Báo cáo nguy hiểm (nhẹ / vừa / nghiêm trọng)
                                 2. Mô tả trước mặt người dùng.
                                 3. Mô tả bên phải ảnh (tức bên trái của người dùng, ghi là “Bên Trái”).
                                 4. Mô tả bên trái ảnh (tức bên phải của người dùng, ghi là “Bên Phải”).
                                 5. Mô tả phía trên đầu người dùng.
                                 6. Mô tả mặt đất dưới chân người dùng.

                                 Ví dụ:
                                 Cảnh báo không có nguy hiểm.
                                 Trước mặt là con đường đô thị rộng rãi, nhiều xe máy đang lưu thông theo làn. Đây là đường một chiều.
                                 Bên Trái là hàng cây xanh mát treo dày đặc cờ đỏ sao vàng và biểu tượng búa liềm.
                                 Bên Phải là làn đường dành cho ô tô, xa hơn là cụm bong bóng nhiều màu.
                                 Phía trên là hệ thống đèn và trang trí ngôi sao xanh dọc trục đường.
                                 Dưới chân là mặt đường bằng phẳng, vạch phân làn rõ ràng và lề đường sạch sẽ.
                                                                        """

# default_prompt = "Hãy mô tả ngắn gọn môi trương xung quanh, câu trả lời chỉ bao gồm việc mô tả, không trả lời thêm gì khác."

def speak_in_thread(text):
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    if len(voices) > 1:
        engine.setProperty("voice", voices[1].id)
    elif voices:
        engine.setProperty("voice", voices[0].id) # Sử dụng giọng nói mặc định nếu chỉ có một
    engine.setProperty("rate", 130)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def speak(text):
    thread = threading.Thread(target=speak_in_thread, args=(text,))
    thread.daemon = True
    thread.start()

# Phát thông báo chào mừng khi ứng dụng khởi chạy
if "welcome_spoken" not in st.session_state:
    speak("Xin chào, tôi là một trợ lý ảo hỗ trợ bạn quan sát môi trường xung quanh")
    st.session_state["welcome_spoken"] = True

running = st.checkbox("Bật/Tắt Camera")
send_button = st.button("Gửi ảnh để mô tả") # Tạo nút bên ngoài vòng lặp
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    status_placeholder.error("Không thể mở camera!")
    running = False

if "camera_on" not in st.session_state:
    st.session_state["camera_on"] = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        status_placeholder.error("Không nhận được frame.")
        running = False
        break

    current_camera_on = running
    if current_camera_on and not st.session_state["camera_on"]:
        speak("Đã bật máy ảnh.")
        st.session_state["camera_on"] = True
    elif not current_camera_on and st.session_state["camera_on"]:
        speak("Đã tắt máy ảnh.")
        st.session_state["camera_on"] = False

    # Hiển thị frame hiện tại từ camera
    if running:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL_Image.fromarray(frame_rgb)
        camera_placeholder.image(img, caption="Đang Trực Tiếp", use_column_width=True)

        # Gửi ảnh lên server nếu nút được nhấn
        if send_button:
            status_placeholder.info("Đang gửi ảnh để mô tả")
            speak("Đang xử lý ảnh")
            try:
                _, img_encoded = cv2.imencode('.jpg', frame)
                image_bytes = img_encoded.tobytes()
                files = {'image': ('image.jpg', image_bytes, 'image/jpeg')}
                data = {'prompt': default_prompt}
                response = requests.post(API_ENDPOINT, files=files, data=data)
                response.raise_for_status()
                result = response.json()
                description = result.get('description')
                if description:
                    description_placeholder.info(f"[{time.strftime('%H:%M:%S')}] Mô tả: {description}")
                    speak(description)
                elif 'error' in result:
                    description_placeholder.error(f"[{time.strftime('%H:%M:%S')}] Lỗi từ server: {result['error']}")
                status_placeholder.success("Đã nhận phản hồi mô tả.")
                # Reset trạng thái nút để tránh gửi liên tục
                st.session_state["send_button_pressed"] = True # Sử dụng session state cho nút
            except requests.exceptions.RequestException as e:
                description_placeholder.error(f"[{time.strftime('%H:%M:%S')}] Lỗi kết nối: {e}")
            except Exception as e:
                description_placeholder.error(f"[{time.strftime('%H:%M:%S')}] Lỗi xử lý: {e}")
        if "send_button_pressed" in st.session_state and st.session_state["send_button_pressed"]:
            st.session_state["send_button_pressed"] = False
            send_button = False # Reset trạng thái nút sau khi xử lý

    else:
        camera_placeholder.empty() # Xóa placeholder khi tắt camera

    time.sleep(1 / 30) # Hiển thị với tốc độ khoảng 30 FPS

if cap.isOpened():
    cap.release()
status_placeholder.info("Máy ảnh đã tắt.") # Thông báo cuối cùng khi camera đóng
if "camera_on" in st.session_state and st.session_state["camera_on"]:
    speak("Máy ảnh đã tắt.")