from gtts import gTTS
import os
import subprocess
import tempfile
from typing import Optional, Union

class VietnameseTTS:
    """
    Module chuyển đổi văn bản tiếng Việt thành giọng nói.
    Có thể tích hợp với các chatbot như Qwen.
    """
    
    def __init__(self, lang='vi', slow=False, use_system_player=True):
        """
        Khởi tạo VietnameseTTS.
        
        Args:
            lang (str): Ngôn ngữ (mặc định: 'vi' cho tiếng Việt)
            slow (bool): Tốc độ nói chậm (mặc định: False)
            use_system_player (bool): Sử dụng lệnh hệ thống để phát (mặc định: True)
        """
        self.lang = lang
        self.slow = slow
        self.use_system_player = use_system_player
        self.temp_dir = tempfile.mkdtemp()
        
    def text_to_speech(self, text: str, filename: Optional[str] = None, 
                      auto_play: bool = True, auto_delete: bool = True) -> Optional[str]:
        """
        Chuyển đổi văn bản thành giọng nói.
        
        Args:
            text (str): Văn bản cần chuyển đổi
            filename (str, optional): Tên file output (mặc định: tự động tạo)
            auto_play (bool): Tự động phát âm thanh (mặc định: True)
            auto_delete (bool): Tự động xóa file sau khi phát (mặc định: True)
            
        Returns:
            str: Đường dẫn đến file âm thanh
        """
        try:
            # Tạo tên file nếu không được cung cấp
            if filename is None:
                filename = os.path.join(self.temp_dir, f"tts_{hash(text) % 10000}.mp3")
            
            # Tạo giọng nói
            tts = gTTS(text=text, lang=self.lang, slow=self.slow)
            tts.save(filename)
            
            print(f"✅ Đã tạo file âm thanh: {filename}")
            
            # Phát âm thanh nếu được yêu cầu
            if auto_play:
                self.play_audio(filename)
                
                # Xóa file nếu được yêu cầu
                if auto_delete:
                    try:
                        os.remove(filename)
                        print(f"🗑️ Đã xóa file tạm: {filename}")
                    except:
                        pass
            
            return filename
            
        except Exception as e:
            print(f"❌ Lỗi khi chuyển đổi văn bản thành giọng nói: {e}")
            return None
    
    def play_audio(self, filepath: str) -> bool:
        """
        Phát file âm thanh.
        
        Args:
            filepath (str): Đường dẫn đến file âm thanh
            
        Returns:
            bool: True nếu phát thành công, False nếu thất bại
        """
        try:
            if self.use_system_player:
                return self._play_with_system(filepath)
            else:
                return self._play_with_playsound(filepath)
        except Exception as e:
            print(f"❌ Lỗi khi phát âm thanh: {e}")
            return False
    
    def _play_with_system(self, filepath: str) -> bool:
        """Phát âm thanh bằng lệnh hệ thống."""
        try:
            # Thử các lệnh phát âm thanh khác nhau
            players = ['mpg123', 'ffplay', 'aplay', 'cvlc']
            
            for player in players:
                try:
                    if player == 'mpg123':
                        subprocess.run([player, filepath], check=True, 
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    elif player == 'ffplay':
                        subprocess.run([player, '-nodisp', '-autoexit', filepath], 
                                     check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    elif player == 'aplay':
                        subprocess.run([player, filepath], check=True, 
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    elif player == 'cvlc':
                        subprocess.run([player, '--play-and-exit', filepath], 
                                     check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    print(f"🔊 Đã phát: {filepath}")
                    return True
                    
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            print("⚠️ Không tìm thấy trình phát âm thanh phù hợp")
            return False
            
        except Exception as e:
            print(f"❌ Lỗi khi phát bằng lệnh hệ thống: {e}")
            return False
    
    def _play_with_playsound(self, filepath: str) -> bool:
        """Phát âm thanh bằng thư viện playsound."""
        try:
            import playsound
            playsound.playsound(filepath)
            print(f"🔊 Đã phát: {filepath}")
            return True
        except ImportError:
            print("⚠️ Thư viện playsound chưa được cài đặt")
            return False
        except Exception as e:
            print(f"❌ Lỗi khi phát bằng playsound: {e}")
            return False
    
    def speak_response(self, response: str, auto_delete: bool = True) -> Optional[str]:
        """
        Phát giọng nói cho response từ chatbot.
        
        Args:
            response (str): Response từ chatbot
            auto_delete (bool): Tự động xóa file sau khi phát
            
        Returns:
            Optional[str]: Đường dẫn đến file âm thanh hoặc None nếu lỗi
        """
        print(f"🤖 Qwen: {response}")
        return self.text_to_speech(response, auto_play=True, auto_delete=auto_delete)
    
    def cleanup(self):
        """Dọn dẹp các file tạm."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Đã dọn dẹp thư mục tạm: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ Không thể dọn dẹp thư mục tạm: {e}")

# Hàm tiện ích để tích hợp nhanh
def speak_text(text: str, lang: str = 'vi', auto_play: bool = True) -> Optional[str]:
    """
    Hàm tiện ích để chuyển văn bản thành giọng nói nhanh chóng.
    
    Args:
        text (str): Văn bản cần chuyển đổi
        lang (str): Ngôn ngữ (mặc định: 'vi')
        auto_play (bool): Tự động phát âm thanh
        
    Returns:
        Optional[str]: Đường dẫn đến file âm thanh hoặc None nếu lỗi
    """
    tts = VietnameseTTS(lang=lang)
    return tts.text_to_speech(text, auto_play=auto_play)

def speak_qwen_response(response: str) -> Optional[str]:
    """
    Hàm tiện ích để phát response từ Qwen.
    
    Args:
        response (str): Response từ Qwen
        
    Returns:
        Optional[str]: Đường dẫn đến file âm thanh hoặc None nếu lỗi
    """
    tts = VietnameseTTS()
    return tts.speak_response(response)

# Ví dụ sử dụng
if __name__ == "__main__":
    # Tạo instance TTS
    tts = VietnameseTTS()
    
    # Ví dụ 1: Chuyển văn bản thành giọng nói
    print("=== Ví dụ 1: Chuyển văn bản thành giọng nói ===")
    tts.text_to_speech("Xin chào, tôi là Qwen, trợ lý AI của bạn!")
    
    # Ví dụ 2: Phát response từ chatbot
    print("\n=== Ví dụ 2: Phát response từ chatbot ===")
    tts.speak_response("Tôi có thể giúp bạn trả lời các câu hỏi và thực hiện nhiều tác vụ khác nhau.")
    
    # Ví dụ 3: Sử dụng hàm tiện ích
    print("\n=== Ví dụ 3: Sử dụng hàm tiện ích ===")
    speak_text("Cảm ơn bạn đã sử dụng dịch vụ của tôi!")
    
    # Dọn dẹp
    tts.cleanup() 