#!/usr/bin/env python3
import requests
import json
import readline  # Cải thiện trải nghiệm input
from tts_module import VietnameseTTS
import re

# --- Cấu hình ---
OLLAMA_API_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen3:0.6b"  # Thay đổi model nếu cần
SYSTEM_PROMPT = (
    "Bạn là Lisa. bạn là trợ lí tư vấn thông tin "
    "Chỉ trả lời trực tiếp cho người dùng bằng tiếng Việt, "
    "khi ai hỏi bạn là ai thì trả lời bạn là trợ lý được tạo ra bởi IS LAB của trường đại học sư phạm kĩ thuật TP.HCM."
)

def remove_think_blocks(text):
    return re.sub(r"", "", text, flags=re.DOTALL).strip()

class QwenInteractiveChat:
    """
    Lớp quản lý phiên chat tương tác với Qwen và TTS.
    """
    def __init__(self, enable_tts=True, lang='vi'):
        """
        Khởi tạo session chat.
        
        Args:
            enable_tts (bool): Bật/tắt TTS.
            lang (str): Ngôn ngữ cho TTS.
        """
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.enable_tts = enable_tts
        if self.enable_tts:
            self.tts = VietnameseTTS(lang=lang)
        else:
            self.tts = None

    def _get_qwen_response(self, stream=True):
        """
        Gửi yêu cầu đến Ollama và xử lý phản hồi.
        
        Args:
            stream (bool): Bật/tắt chế độ stream.
            
        Returns:
            str: Toàn bộ nội dung phản hồi từ model, hoặc None nếu có lỗi.
        """
        payload = {
            "model": MODEL_NAME,
            "messages": self.messages,
            "stream": stream,
            "think": False
        }
        
        full_response = ""
        try:
            # Gửi yêu cầu POST
            response_stream = requests.post(OLLAMA_API_URL, json=payload, stream=stream)
            response_stream.raise_for_status()
            
            print("🤖 Qwen: ", end="", flush=True)
            
            # Xử lý từng dòng dữ liệu stream
            for line in response_stream.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    chunk = data.get("message", {}).get("content", "")
                    print(chunk, end="", flush=True)
                    full_response += chunk
                    
                    # Dừng lại khi message cuối cùng được gửi
                    if data.get("done"):
                        break
            
            print()  # In dòng mới sau khi hoàn thành
            return full_response
            
        except requests.RequestException as e:
            print(f"\n❌ Lỗi khi gọi API Ollama: {e}")
            return None
        except json.JSONDecodeError:
            print("\n❌ Lỗi giải mã JSON từ API.")
            return None

    def start(self):
        """
        Bắt đầu vòng lặp chat tương tác.
        """
        print(" Chatbot Qwen với Giọng nói đã sẵn sàng.")
        print("   (Gõ 'quit' hoặc 'exit' để thoát)")
        print("=" * 50)
        
        try:
            while True:
                # Lấy input từ người dùng
                user_input = input("👤 Bạn: ").strip()
                
                if not user_input:
                    continue
                
                # Kiểm tra lệnh thoát
                if user_input.lower() in ["quit", "exit"]:
                    print("👋 Tạm biệt!")
                    break
                
                # Thêm tin nhắn của người dùng vào lịch sử
                self.messages.append({"role": "user", "content": user_input})
                
                # Lấy phản hồi từ Qwen
                qwen_response = self._get_qwen_response()
                clean_response = remove_think_blocks(qwen_response)
                
                if clean_response:
                    # Thêm phản hồi của Qwen vào lịch sử để duy trì ngữ cảnh
                    self.messages.append({"role": "assistant", "content": clean_response})
                    
                    # Phát giọng nói nếu TTS được bật
                    if self.enable_tts and self.tts:
                        print("... 🔊 Đang phát giọng nói ...")
                        self.tts.text_to_speech(clean_response, auto_play=True, auto_delete=True)
                
                print("-" * 50)
                
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Tạm biệt!")
        finally:
            # Dọn dẹp tài nguyên
            if self.enable_tts and self.tts:
                self.tts.cleanup()

if __name__ == "__main__":
    # Khởi tạo và bắt đầu session chat
    chat_session = QwenInteractiveChat(enable_tts=True)
    chat_session.start() 