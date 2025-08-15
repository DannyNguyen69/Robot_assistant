#!/usr/bin/env python3
import requests
import json
import readline  # Cải thiện trải nghiệm input
from tts_module import VietnameseTTS
import re
import os
from pathlib import Path

# --- Cấu hình ---
OLLAMA_API_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen3:0.6b"  # Thay đổi model nếu cần
DATA_FILE = "data.json"  # File chứa dữ liệu tham khảo

def load_knowledge_base(file_path):
    """
    Đọc file JSON chứa dữ liệu kiến thức.
    
    Args:
        file_path (str): Đường dẫn tới file JSON
        
    Returns:
        dict: Dữ liệu từ file JSON hoặc dict rỗng nếu có lỗi
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✅ Đã tải dữ liệu từ {file_path}")
                return data
        else:
            print(f"⚠️ File {file_path} không tồn tại")
            return {}
    except json.JSONDecodeError:
        print(f"❌ Lỗi đọc file JSON: {file_path}")
        return {}
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {e}")
        return {}

def create_system_prompt_with_data(knowledge_base):
    """
    Tạo system prompt bao gồm dữ liệu từ JSON.
    
    Args:
        knowledge_base (dict): Dữ liệu kiến thức từ JSON
        
    Returns:
        str: System prompt hoàn chỉnh
    """
    base_prompt = (
        "Bạn là Lisa, trợ lý tư vấn thông tin được tạo ra bởi IS LAB của trường Đại học Sư phạm Kỹ thuật TP.HCM. "
        "Hãy trả lời bằng tiếng Việt một cách thân thiện và chính xác.\n\n"
    )
    
    if knowledge_base:
        base_prompt += "THÔNG TIN THAM KHẢO:\n"
        base_prompt += json.dumps(knowledge_base, ensure_ascii=False, indent=2)
        base_prompt += "\n\nHãy sử dụng thông tin trên để trả lời câu hỏi của người dùng khi phù hợp. "
        base_prompt += "Nếu thông tin không có trong dữ liệu tham khảo, hãy trả lời dựa trên kiến thức chung của bạn."
    
    return base_prompt

def remove_think_blocks(text):
    """Loại bỏ các think blocks khỏi response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def search_in_knowledge_base(query, knowledge_base):
    """
    Tìm kiếm thông tin liên quan trong knowledge base.
    
    Args:
        query (str): Câu hỏi của người dùng
        knowledge_base (dict|list): Dữ liệu kiến thức
        
    Returns:
        list: Danh sách thông tin liên quan
    """
    if not knowledge_base:
        return []
    
    relevant_info = []
    query_lower = query.lower()
    query_words = query_lower.split()
    
    # Xử lý đặc biệt cho dữ liệu dạng prompt-response
    if isinstance(knowledge_base, list) and knowledge_base and isinstance(knowledge_base[0], dict):
        if "prompt" in knowledge_base[0] and "response" in knowledge_base[0]:
            # Dữ liệu dạng FAQ prompt-response
            for i, item in enumerate(knowledge_base):
                prompt = item.get("prompt", "").lower()
                response = item.get("response", "")
                
                # Tính điểm tương đồng
                score = 0
                for word in query_words:
                    if word in prompt:
                        score += 2  # Điểm cao hơn nếu từ khóa xuất hiện trong prompt
                    elif word in response.lower():
                        score += 1  # Điểm thấp hơn nếu từ khóa xuất hiện trong response
                
                if score > 0:
                    relevant_info.append({
                        "path": f"faq_{i}",
                        "key": f"Q&A_{i+1}",
                        "value": f"Q: {item['prompt']}\nA: {item['response']}",
                        "score": score,
                        "prompt": item["prompt"],
                        "response": item["response"]
                    })
            
            # Sắp xếp theo điểm tương đồng
            relevant_info.sort(key=lambda x: x["score"], reverse=True)
            return relevant_info
    
    # Xử lý dữ liệu dạng thông thường
    def search_recursive(data, path=""):
        """Tìm kiếm đệ quy trong nested dict/list."""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, (dict, list)):
                    search_recursive(value, current_path)
                else:
                    if any(keyword in str(value).lower() or keyword in key.lower() 
                          for keyword in query_words):
                        relevant_info.append({
                            "path": current_path,
                            "key": key,
                            "value": value
                        })
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]" if path else f"item_{i}"
                if isinstance(item, (dict, list)):
                    search_recursive(item, current_path)
                else:
                    if any(keyword in str(item).lower() for keyword in query_words):
                        relevant_info.append({
                            "path": current_path,
                            "key": f"item_{i}",
                            "value": item
                        })
    
    search_recursive(knowledge_base)
    return relevant_info

class QwenInteractiveChat:
    """
    Lớp quản lý phiên chat tương tác với Qwen, TTS và JSON data.
    """
    def __init__(self, enable_tts=True, lang='vi', data_file=DATA_FILE):
        """
        Khởi tạo session chat.
        
        Args:
            enable_tts (bool): Bật/tắt TTS.
            lang (str): Ngôn ngữ cho TTS.
            data_file (str): Đường dẫn tới file JSON chứa dữ liệu.
        """
        # Tải knowledge base
        self.knowledge_base = load_knowledge_base(data_file)
        
        # Tạo system prompt với dữ liệu
        system_prompt = create_system_prompt_with_data(self.knowledge_base)
        self.messages = [{"role": "system", "content": system_prompt}]
        
        # Cấu hình TTS
        self.enable_tts = enable_tts
        if self.enable_tts:
            self.tts = VietnameseTTS(lang=lang)
        else:
            self.tts = None

    def reload_knowledge_base(self, data_file=DATA_FILE):
        """
        Tải lại dữ liệu từ file JSON.
        
        Args:
            data_file (str): Đường dẫn tới file JSON
        """
        self.knowledge_base = load_knowledge_base(data_file)
        system_prompt = create_system_prompt_with_data(self.knowledge_base)
        self.messages[0] = {"role": "system", "content": system_prompt}
        print("🔄 Đã tải lại dữ liệu và cập nhật system prompt")

    def _enhance_user_query(self, user_input):
        """
        Tăng cường câu hỏi của người dùng bằng thông tin liên quan từ knowledge base.
        
        Args:
            user_input (str): Câu hỏi gốc của người dùng
            
        Returns:
            str: Câu hỏi được tăng cường với context
        """
        if not self.knowledge_base:
            return user_input
        
        # Tìm thông tin liên quan
        relevant_info = search_in_knowledge_base(user_input, self.knowledge_base)
        
        if relevant_info:
            # Xử lý đặc biệt cho dữ liệu FAQ
            if relevant_info and "prompt" in relevant_info[0]:
                enhanced_query = f"Câu hỏi của người dùng: {user_input}\n\n"
                enhanced_query += "Các câu hỏi và câu trả lời tương tự từ cơ sở dữ liệu:\n"
                
                for i, info in enumerate(relevant_info[:2]):  # Lấy 2 câu hỏi tương tự nhất
                    enhanced_query += f"\n{i+1}. Câu hỏi: {info['prompt']}\n"
                    enhanced_query += f"   Trả lời: {info['response']}\n"
                
                enhanced_query += f"\nDựa trên thông tin trên, hãy trả lời câu hỏi: '{user_input}' một cách tự nhiên và chính xác."
                return enhanced_query
            else:
                # Xử lý dữ liệu thông thường
                enhanced_query = f"Câu hỏi: {user_input}\n\nThông tin liên quan từ dữ liệu:\n"
                for info in relevant_info[:3]:  # Giới hạn 3 thông tin liên quan nhất
                    enhanced_query += f"- {info['key']}: {info['value']}\n"
                enhanced_query += "\nVui lòng trả lời dựa trên thông tin trên nếu phù hợp."
                return enhanced_query
        
        return user_input

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
            
            print("🤖 Lisa: ", end="", flush=True)
            
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

    def show_help(self):
        """Hiển thị hướng dẫn sử dụng."""
        print("\n📋 HƯỚNG DẪN SỬ DỤNG:")
        print("• Gõ câu hỏi bình thường để chat")
        print("• 'reload' - Tải lại dữ liệu từ JSON")
        print("• 'info' - Xem thông tin về dữ liệu đã tải")
        print("• 'help' - Hiển thị hướng dẫn này")
        print("• 'quit' hoặc 'exit' - Thoát chương trình")
        print()

    def show_data_info(self):
        """Hiển thị thông tin về dữ liệu đã tải."""
        if not self.knowledge_base:
            print("📊 Chưa có dữ liệu nào được tải.")
            return
        
        print(f"📊 THÔNG TIN DỮ LIỆU:")
        
        # Xử lý cả dictionary và list
        if isinstance(self.knowledge_base, dict):
            print(f"• Loại dữ liệu: Dictionary")
            print(f"• Số lượng key chính: {len(self.knowledge_base)}")
            print(f"• Các key chính: {list(self.knowledge_base.keys())}")
        elif isinstance(self.knowledge_base, list):
            print(f"• Loại dữ liệu: List")
            print(f"• Số lượng items: {len(self.knowledge_base)}")
            
            # Kiểm tra xem có phải là dữ liệu FAQ không
            if (self.knowledge_base and isinstance(self.knowledge_base[0], dict) and
                "prompt" in self.knowledge_base[0] and "response" in self.knowledge_base[0]):
                print(f"• Định dạng: FAQ (Prompt-Response)")
                print(f"• Số câu hỏi có sẵn: {len(self.knowledge_base)}")
                
                # Hiển thị vài câu hỏi mẫu
                print("• Một số câu hỏi mẫu:")
                for i, item in enumerate(self.knowledge_base[:3]):
                    print(f"  {i+1}. {item.get('prompt', 'N/A')}")
                if len(self.knowledge_base) > 3:
                    print(f"  ... và {len(self.knowledge_base) - 3} câu hỏi khác")
            elif self.knowledge_base and isinstance(self.knowledge_base[0], dict):
                first_item_keys = list(self.knowledge_base[0].keys())
                print(f"• Keys của item đầu tiên: {first_item_keys}")
        else:
            print(f"• Loại dữ liệu: {type(self.knowledge_base).__name__}")
        
        def count_items(data, depth=0):
            count = 0
            if isinstance(data, dict):
                count += len(data)
                for value in data.values():
                    if isinstance(value, (dict, list)):
                        count += count_items(value, depth + 1)
            elif isinstance(data, list):
                count += len(data)
                for item in data:
                    if isinstance(item, (dict, list)):
                        count += count_items(item, depth + 1)
            return count
        
        total_items = count_items(self.knowledge_base)
        print(f"• Tổng số items: {total_items}")
        print()

    def start(self):
        """
        Bắt đầu vòng lặp chat tương tác.
        """
        print("🎯 Chatbot Lisa với dữ liệu JSON đã sẵn sàng!")
        print("   (Gõ 'help' để xem hướng dẫn)")
        print("=" * 50)
        
        # Hiển thị thông tin dữ liệu đã tải
        if self.knowledge_base:
            self.show_data_info()
        
        try:
            while True:
                # Lấy input từ người dùng
                user_input = input("👤 Bạn: ").strip()
                
                if not user_input:
                    continue
                
                # Xử lý các lệnh đặc biệt
                if user_input.lower() in ["quit", "exit"]:
                    print("👋 Tạm biệt!")
                    break
                elif user_input.lower() == "help":
                    self.show_help()
                    continue
                elif user_input.lower() == "info":
                    self.show_data_info()
                    continue
                elif user_input.lower() == "reload":
                    self.reload_knowledge_base()
                    continue
                
                # Tăng cường câu hỏi với thông tin từ knowledge base
                enhanced_query = self._enhance_user_query(user_input)
                
                # Thêm tin nhắn của người dùng vào lịch sử (sử dụng câu hỏi gốc)
                self.messages.append({"role": "user", "content": enhanced_query})
                
                # Lấy phản hồi từ Qwen
                qwen_response = self._get_qwen_response()
                clean_response = remove_think_blocks(qwen_response)
                
                if clean_response:
                    # Thêm phản hồi của Qwen vào lịch sử để duy trì ngữ cảnh
                    self.messages.append({"role": "assistant", "content": clean_response})
                    
                    # Phát giọng nói nếu TTS được bật
                    if self.enable_tts and self.tts:
                        print("🔊 Đang phát giọng nói...")
                        self.tts.text_to_speech(clean_response, auto_play=True, auto_delete=True)
                
                print("-" * 50)
                
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Tạm biệt!")
        finally:
            # Dọn dẹp tài nguyên
            if self.enable_tts and self.tts:
                self.tts.cleanup()

def create_sample_data_file():
    """
    Tạo file data.json mẫu nếu chưa tồn tại.
    """
    if not os.path.exists(DATA_FILE):
        sample_data = {
            "thong_tin_truong": {
                "ten": "Trường Đại học Sư phạm Kỹ thuật TP.HCM",
                "ten_viet_tat": "HCMUTE",
                "dia_chi": "1 Võ Văn Ngân, Thủ Đức, TP.HCM",
                "website": "https://hcmute.edu.vn",
                "dien_thoai": "028-38968641"
            },
            "khoa_vien": {
                "CNTT": {
                    "ten_day_du": "Khoa Công nghệ Thông tin",
                    "truong_khoa": "TS. Nguyễn Văn A",
                    "cac_nganh": ["Công nghệ Thông tin", "Khoa học Máy tính", "An toàn Thông tin"]
                },
                "DIEN": {
                    "ten_day_du": "Khoa Điện - Điện tử",
                    "truong_khoa": "TS. Trần Văn B",
                    "cac_nganh": ["Kỹ thuật Điện", "Kỹ thuật Điện tử", "Tự động hóa"]
                }
            },
            "is_lab": {
                "ten": "Phòng thí nghiệm Hệ thống Thông tin (IS LAB)",
                "mo_ta": "Phòng lab nghiên cứu về AI, Machine Learning và các hệ thống thông tin",
                "thanh_vien": ["Sinh viên CNTT", "Nghiên cứu sinh", "Giảng viên hướng dẫn"],
                "du_an": ["Chatbot AI", "Hệ thống quản lý", "Ứng dụng mobile"]
            },
            "hoc_phi": {
                "dai_hoc": {
                    "hoc_phi_1_tin_chi": 330000,
                    "don_vi": "VND",
                    "ghi_chu": "Học phí có thể thay đổi theo từng năm học"
                }
            },
            "lien_he": {
                "phong_dao_tao": "028-38968641",
                "phong_ctsv": "028-38968642",
                "email": "info@hcmute.edu.vn"
            }
        }
        
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã tạo file mẫu {DATA_FILE}")

if __name__ == "__main__":
    # Tạo file dữ liệu mẫu nếu chưa có
    create_sample_data_file()
    
    # Khởi tạo và bắt đầu session chat
    chat_session = QwenInteractiveChat(enable_tts=True)
    chat_session.start()