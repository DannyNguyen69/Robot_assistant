# Text-to-Speech cho Qwen Chatbot

Module chuyển đổi văn bản tiếng Việt thành giọng nói, có thể tích hợp dễ dàng với Qwen chatbot.

## Tính năng

- ✅ Chuyển đổi văn bản tiếng Việt thành giọng nói
- ✅ Tích hợp sẵn với Qwen chatbot
- ✅ Hỗ trợ nhiều trình phát âm thanh (mpg123, ffplay, aplay, cvlc)
- ✅ Tự động dọn dẹp file tạm
- ✅ Có thể bật/tắt tính năng TTS
- ✅ Hàm tiện ích để sử dụng nhanh

## Cài đặt

### 1. Cài đặt dependencies

```bash
# Cài đặt Python packages
pip install gTTS playsound==1.2.2 PyGObject

# Cài đặt trình phát âm thanh (Arch Linux)
sudo pacman -S mpg123 alsa-utils ffmpeg vlc

# Hoặc Ubuntu/Debian
sudo apt install mpg123 alsa-utils ffmpeg vlc
```

### 2. Cài đặt từ requirements.txt

```bash
pip install -r requirements.txt
```

## Sử dụng

### Cách 1: Sử dụng module cơ bản

```python
from tts_module import VietnameseTTS

# Tạo instance TTS
tts = VietnameseTTS()

# Chuyển văn bản thành giọng nói
tts.text_to_speech("Xin chào, tôi là Qwen!")

# Dọn dẹp
tts.cleanup()
```

### Cách 2: Sử dụng hàm tiện ích

```python
from tts_module import speak_text, speak_qwen_response

# Phát văn bản tùy chỉnh
speak_text("Đây là một đoạn văn bản tùy chỉnh")

# Phát response từ Qwen
qwen_response = "Tôi có thể giúp bạn trả lời câu hỏi!"
speak_qwen_response(qwen_response)
```

### Cách 3: Tích hợp với Qwen chatbot

```python
from qwen_tts_integration import QwenWithTTS

# Tạo Qwen với TTS
qwen_tts = QwenWithTTS(enable_tts=True, lang='vi')

# Chat với TTS
response = qwen_tts.chat_with_tts("Xin chào")

# Dọn dẹp
qwen_tts.cleanup()
```

### Cách 4: Tích hợp vào Qwen thật

```python
from tts_module import VietnameseTTS

class QwenWithVoice:
    def __init__(self):
        self.tts = VietnameseTTS()
        # Khởi tạo Qwen model ở đây
    
    def chat(self, user_input):
        # Gọi Qwen API/model
        qwen_response = self.call_qwen_api(user_input)
        
        # Hiển thị response
        print(f"Qwen: {qwen_response}")
        
        # Phát giọng nói
        self.tts.speak_response(qwen_response)
        
        return qwen_response
```

## Cấu hình

### Thay đổi ngôn ngữ

```python
# Tiếng Việt (mặc định)
tts = VietnameseTTS(lang='vi')

# Tiếng Anh
tts = VietnameseTTS(lang='en')

# Tiếng Trung
tts = VietnameseTTS(lang='zh')
```

### Thay đổi tốc độ nói

```python
# Tốc độ bình thường (mặc định)
tts = VietnameseTTS(slow=False)

# Tốc độ chậm
tts = VietnameseTTS(slow=True)
```

### Sử dụng playsound thay vì lệnh hệ thống

```python
# Sử dụng lệnh hệ thống (mặc định, ổn định hơn)
tts = VietnameseTTS(use_system_player=True)

# Sử dụng playsound (cần PyGObject)
tts = VietnameseTTS(use_system_player=False)
```

## Ví dụ chạy

```bash
# Chạy demo cơ bản
python tts_module.py

# Chạy demo tích hợp với Qwen
python qwen_tts_integration.py
```

## Troubleshooting

### Lỗi "No module named 'gi'"
```bash
# Cài đặt PyGObject
pip install PyGObject

# Hoặc cài gói hệ thống
sudo pacman -S python-gobject  # Arch Linux
sudo apt install python3-gi    # Ubuntu/Debian
```

### Lỗi "No module named 'gtts'"
```bash
# Cài đặt lại gTTS
pip install gTTS

# Hoặc cài bằng python -m pip
python -m pip install gTTS
```

### Không phát được âm thanh
```bash
# Cài đặt trình phát âm thanh
sudo pacman -S mpg123 alsa-utils ffmpeg

# Kiểm tra thiết bị âm thanh
aplay -l
```

### Lỗi playsound trên Python 3.13
Sử dụng playsound==1.2.2 hoặc chuyển sang Python 3.10/3.11.

## Cấu trúc file

```
Text2speech/
├── tts_module.py              # Module TTS chính
├── qwen_tts_integration.py    # Tích hợp với Qwen
├── test.py                    # File test cũ
├── requirements.txt           # Dependencies
└── README.md                 # Hướng dẫn này
```

## License

MIT License - Sử dụng tự do cho mục đích cá nhân và thương mại. 