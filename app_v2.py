import streamlit as st
import cv2
import requests
import time
import io
from PIL import Image as PIL_Image
import numpy as np
import pyttsx3
import threading
import warnings
import logging
from PIL import ImageEnhance

warnings.filterwarnings("ignore")
logging.getLogger('streamlit').setLevel(logging.ERROR)


# Các biến toàn cục (URL server, khoảng thời gian chụp)
NGROK_URL = "https://regularly-tender-kite.ngrok-free.app"
API_ENDPOINT = f"{NGROK_URL}/describe_image"
CAPTURE_INTERVAL = 5  # giây
MARK_SPEAK = 0


st.sidebar.title("Điều hướng")
page = st.sidebar.radio("Chọn trang", ["SightSeeingGemma", "Trang chủ", "Thông tin"])


# === Trang chủ ===
if page == "Trang chủ":
    col1, col2 = st.columns([1, 3])  # Chia layout: 1 phần logo, 3 phần chữ

    with col1:
        st.image("imgs/logo.png")  # Thay đổi kích thước logo nếu cần

    with col2:
        st.markdown("""
            <div style='display: flex; align-items: center; height: 100%'>
                <h1 style='margin: 0px 20px 0px 0px;'>Chào mừng đến với dự án SightSeeingGemma</h1>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
> *"The smart assistant helps the visually impaired see the world through AI vision and language."*  
> *"Let us be the guiding light accompanying you."*

---

## 1. Giới thiệu chung

### Mục tiêu

**SightSeeingGemma** là một hệ thống hỗ trợ người khiếm thị, sử dụng mô hình đa phương thức **Gemma-3-4b-it** để chuyển đổi hình ảnh từ camera thành mô tả ngôn ngữ tự nhiên theo thời gian thực (Image-Text-to-Text).  
Hệ thống giúp phát hiện nguy hiểm và mô tả sinh động môi trường xung quanh thông qua lời nói.

### Điểm nổi bật

- Ứng dụng công nghệ AI đa phương thức.
- Mô tả ngữ cảnh một cách trung thực, trực quan.
- Phát hiện nguy hiểm và chuyển thành thông báo bằng âm thanh.
- Tích hợp dễ dàng vào thiết bị đeo thông minh.

---

## 2. Hiện trạng và vấn đề

### Khó khăn của người khiếm thị

Người khiếm thị thường gặp nhiều khó khăn trong việc phát hiện mối nguy hiểm xung quanh do không thể quan sát được môi trường.  
Việc không nhận biết kịp thời có thể dẫn đến những rủi ro trong sinh hoạt hằng ngày.

### Hạn chế của công nghệ hiện có

Các công nghệ hỗ trợ hiện tại còn hạn chế trong khả năng phân tích bối cảnh theo thời gian thực, không thể cảnh báo ngay tức thì, và thiếu sự mô tả cụ thể trong môi trường sống thực tế.

---

## 3. Phương pháp tiếp cận

### Cách hoạt động

Hệ thống hoạt động bằng cách chụp ảnh từ camera theo chu kỳ, gửi đến mô hình AI xử lý hình ảnh (Gemma-3-4b-it) trên nền tảng Google Colab.  
Mô hình trả về mô tả và cảnh báo, sau đó được chuyển thành lời nói thông qua chức năng **Text2Speech** để thông báo cho người dùng.

### Tính năng chính

- Phân tích ảnh thời gian thực bằng AI.
- Nhận diện mối nguy hiểm và mô tả cụ thể.
- Chuyển đổi mô tả thành âm thanh phản hồi.
- Có thể điều khiển từ xa, cập nhật qua server.
- Hỗ trợ tích hợp vào kính thông minh, thiết bị đeo.

---

## 4. Kiến trúc hệ thống

### Thành phần

- **Thiết bị camera** (thiết bị đeo hoặc máy tính)
- **Hệ thống xử lý local**: Chạy ứng dụng Streamlit
- **Ngrok**: Trung gian tạo địa chỉ công khai
- **Mô hình AI**: Google Gemma-3-4b-it trên Colab
- **Chức năng Text2Speech**: Phát mô tả bằng giọng nói

### Luồng dữ liệu

1. Ảnh được chụp từ camera và gửi đến server.
2. Server xử lý ảnh và sinh mô tả kèm cảnh báo.
3. Ứng dụng nhận mô tả và phát bằng giọng nói.
4. Lặp lại sau mỗi khoảng thời gian định sẵn.

---

## 5. Tác động và tiềm năng

### Lợi ích xã hội

**SightSeeingGemma** không chỉ là bước tiến về công nghệ mà còn thể hiện tinh thần nhân đạo sâu sắc – giúp người khiếm thị "nhìn" thấy thế giới, nhận biết nguy hiểm và hòa nhập với xã hội một cách an toàn, độc lập hơn.

### Khả năng mở rộng

- Dễ dàng tích hợp vào các thiết bị đeo cá nhân.
- Có thể triển khai trên server lớn để phục vụ nhiều người dùng.
    """)

# === Trang thông tin ===
elif page == "Thông tin":
    st.title("Thông tin dự án SightSeeingGemma")
    st.markdown("""
## 1. Giới thiệu dự án

Dự án **SightSeeingGemma** là một đề tài nghiên cứu được phát triển nhằm tham gia cuộc thi **Thiết kế sáng tạo sản phẩm công nghệ dành cho người khuyết tật năm 2025**, do **Trung tâm Phát triển Khoa học & Công nghệ Trẻ** tổ chức.

Mục tiêu của dự án là xây dựng một hệ thống trợ lý thông minh giúp người khiếm thị nhận diện môi trường xung quanh và các nguy hiểm tiềm ẩn thông qua camera, sử dụng trí tuệ nhân tạo để mô tả hình ảnh bằng lời nói một cách sinh động và tức thời.

---

## 2. Thông tin dự án

### Tên dự án
**SightSeeingGemma**

### Giảng viên hướng dẫn
**PGS. TS Quản Thành Thơ**: trưởng khoa Khoa học và Kỹ thuật máy tính tại trường Đại học Bách Khoa, Đại học Quốc gia TPHCM

### Thành viên nhóm thực hiện
- **Đinh Trác Đức Anh**: Trưởng nhóm, sinh viên năm 2 ngành Khoa học máy tính, trường Đại học Bách Khoa, Đại học Quốc gia TPHCM. Thành viên nhóm nghiên cứu URA.
- **Tạ Tiến Tài**: Sinh viên năm 2 ngành Khoa học máy tính, trường Đại học Bách Khoa, Đại học Quốc gia TPHCM. Thành viên nhóm nghiên cứu URA.
- **Hứa Tuệ Minh**: Sinh viên năm 1 ngành Khoa học máy tính, trường Đại học Bách Khoa, Đại học Quốc gia TPHCM. Thành viên nhóm nghiên cứu URA.
- **Trịnh Hữu Trí** Sinh viên năm 1 ngành Khoa học máy tính, trường Đại học Bách Khoa, Đại học Quốc gia TPHCM.Thành viên nhóm nghiên cứu URA.

### Đối tượng hưởng lợi
Người khiếm thị hoặc người bị suy giảm chức năng thị giác – tức là họ không thể nhìn nhưng vẫn có khả năng nghe. Đây là nhóm đối tượng cần sự hỗ trợ tức thời từ công nghệ mô tả ngôn ngữ thay cho thị giác.

---

## 3. Cuộc thi tham dự

Dự án được phát triển để đăng ký tham dự cuộc thi:

**Tên cuộc thi:** Cuộc thi Thiết kế sáng tạo sản phẩm, công nghệ dành cho người khuyết tật năm 2025  
**Đơn vị tổ chức:** Trung tâm Phát triển Khoa học & Công nghệ Trẻ  
**Hạn đăng ký:** 31/5/2025  
**Đối tượng dự thi:** Công dân Việt Nam dưới 35 tuổi, đang sinh sống, học tập, làm việc trong và ngoài nước """) 

    st.image("imgs/poster-SP-nguoi-khuyet-tat-2025-1080x720.png", caption="Trang thông tin cuộc thi", use_container_width =True)

    st.markdown("""---

## 4. Ý nghĩa dự án

Dự án không chỉ mang tính công nghệ mà còn thể hiện tinh thần nhân văn sâu sắc. Bằng việc kết hợp giữa AI thị giác và ngôn ngữ, SightSeeingGemma giúp người khiếm thị:

- Nhận biết môi trường sống một cách sinh động và an toàn hơn
- Tăng cường khả năng tự lập và hòa nhập cộng đồng
- Được tiếp cận công nghệ hiện đại một cách dễ dàng và nhân đạo

Dự án góp phần hướng đến một xã hội công bằng hơn, nơi mọi người – dù có khuyết tật – đều được hỗ trợ để sống trọn vẹn và bình đẳng.

    """)
    st.image("poster.png", caption="Poster dự án", use_container_width =True)

# === Trang SightSeeingGemma ===
elif page == "SightSeeingGemma":
    # 👉 Toàn bộ nội dung camera và xử lý ảnh đưa vào đây
    # 👉 Bạn giữ nguyên phần code xử lý camera, gửi ảnh... từ mã gốc bạn gửi ở trên
    # Ví dụ gợi ý:
    # Hiển thị logo và tiêu đề trong cùng một hàng
    col1, col2 = st.columns([1, 3])  # Chia layout: 1 phần logo, 3 phần chữ

    with col1:
        st.image("imgs/logo.png")  # Thay đổi kích thước logo nếu cần

    with col2:
        st.markdown("""
            <div style='display: flex; align-items: center; height: 100%'>
                <h1 style='margin: 20px 20px 0px 0px;'>SightSeeingGemma</h1>
            </div>
        """, unsafe_allow_html=True)

    st.sidebar.title("Tuỳ chọn thiết lập ảnh")
    # Widget điều khiển trong sidebar
    brightness = st.sidebar.slider("BRIGHTNESS", 0, 100, 50)
    contrast = st.sidebar.slider("CONTRAST", 0, 100, 50)
    sharpness = st.sidebar.slider("SHARPNESS", 0, 100, 50)
    saturation = st.sidebar.slider("SATURATION", 0, 100, 50)
    camera_placeholder = st.empty()
    description_placeholder = st.empty()
    status_placeholder = st.empty()

    default_prompt = """Bạn chính là đôi mắt thay thế cho người khiếm thị. Hình ảnh này là những gì người khiếm thị đang nhìn thấy thông qua camera gắn trên kính mắt của họ. Bạn cần mô tả lại thế giới trước mặt họ một cách trung thực, rõ ràng và theo đúng hướng nhìn của họ.

    Mô tả thành 6 dòng ngắn, mỗi dòng 1–2 câu. Tập trung vào yếu tố có thể gây nguy hiểm và tập trung mô tả những con người nhận diện được trong tấm ảnh. Ngôn ngữ đơn giản, dễ hiểu, không dùng từ hoa mỹ. Không bịa đặt, không tưởng tượng, không suy đoán.

    Lưu ý:

    Mọi mô tả đều phải từ góc nhìn của người khiếm thị, không phải góc nhìn của người quan sát ảnh từ bên ngoài.

    Tất cả những người trong ảnh đều là những người mà camera của người khiếm thị nhìn thấy.

    Tuyệt đối không được tưởng tượng hoặc suy đoán nếu thông tin không có trong ảnh.

    Trung thực và an toàn là ưu tiên hàng đầu.

    Trả lời theo định dạng sau:

    1. Cảnh báo nguy hiểm (BẮT BUỘC PHẢI CÓ CÂU NÀY TRONG MỌI MÔ TẢ) trong các mức sau và nói lý do tại sao (nếu có):

      Báo cáo nguy hiểm (nhẹ / vừa / nghiêm trọng) nếu có yếu tố nguy hiểm.

      Báo cáo an toàn nếu không có yếu tố gây nguy hiểm.

      Báo cáo không có nguy hiểm nếu không có yếu tố gây nguy hiểm.

    2. Mô tả những gì ngay phía trước người khiếm thị (Trung tâm bức ảnh, nếu không thể xác định, ghi "Không thể xác định").

    3. Mô tả những gì ở bên trái người dùng (tức là bên phải của ảnh, nếu không thể xác định, ghi "Không thể xác định"). Bắt đầu bằng “Bên Trái: ...”

    4. Mô tả những gì ở bên phải người dùng (tức là bên trái của ảnh, nếu không thể xác định, ghi "Không thể xác định"). Bắt đầu bằng “Bên Phải: …”

    5. Mô tả phía trên ảnh (nếu không thể xác định, ghi "Không thể xác định").

    6. Mô tả phía dưới ảnh (nếu không thể xác định, ghi "Không thể xác định").
    """

    # default_prompt = "Hãy mô tả ngắn gọn môi trương xung quanh, câu trả lời chỉ bao gồm việc mô tả, không trả lời thêm gì khác."


    def speak_in_thread(text):
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        if len(voices) > 1:
            engine.setProperty("voice", voices[89].id)
        elif voices:
            engine.setProperty("voice", voices[0].id) # Sử dụng giọng nói mặc định nếu chỉ có một
        engine.setProperty("rate", 190)
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

            # Áp dụng các điều chỉnh từ slider
            enhancer_brightness = ImageEnhance.Brightness(img)
            img = enhancer_brightness.enhance(brightness / 50)  # 50 là mức trung bình

            enhancer_contrast = ImageEnhance.Contrast(img)
            img = enhancer_contrast.enhance(contrast / 50)

            enhancer_sharpness = ImageEnhance.Sharpness(img)
            img = enhancer_sharpness.enhance(sharpness / 50)

            enhancer_color = ImageEnhance.Color(img)
            img = enhancer_color.enhance(saturation / 50)

            # Hiển thị ảnh đã chỉnh sửa
            camera_placeholder.image(img, caption="Đang Trực Tiếp")

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

        time.sleep(1 / 60) # Hiển thị với tốc độ khoảng 30 FPS

    if cap.isOpened():
        cap.release()
    status_placeholder.info("Máy ảnh đã tắt.") # Thông báo cuối cùng khi camera đóng
    if "camera_on" in st.session_state and st.session_state["camera_on"]:
        speak("Máy ảnh đã tắt.")