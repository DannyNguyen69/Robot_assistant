#!/usr/bin/env python3
"""
Lisa Chatbot - Vietnamese AI Assistant
Created by IS LAB, Ho Chi Minh City University of Technology and Education
"""

import requests
import json
import readline
import re
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from tts_module import VietnameseTTS

# --- Configuration ---
@dataclass
class Config:
    """Configuration class for the chatbot"""
    OLLAMA_API_URL: str = "http://127.0.0.1:11434/api/chat"
    MODEL_NAME: str = "qwen3:0.6b"
    DATA_FILE: str = "data.json"
    MAX_RETRIES: int = 3
    TIMEOUT: int = 30
    MAX_FAQ_RESULTS: int = 2
    MAX_STRUCTURED_RESULTS: int = 3

config = Config()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lisa_chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and parsing of knowledge base data"""
    
    @staticmethod
    def load_knowledge_base(file_path: str) -> Dict[str, Any]:
        """
        Load knowledge base from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary containing knowledge data or empty dict if error
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} does not exist")
                return {}
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded data from {file_path}")
                return data
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return {}

class PromptBuilder:
    """Builds system prompts with knowledge base integration"""
    
    @staticmethod
    def create_system_prompt(knowledge_base: Dict[str, Any]) -> str:
        """
        Create system prompt with knowledge base data.
        
        Args:
            knowledge_base: Knowledge base data
            
        Returns:
            Complete system prompt string
        """
        base_prompt = (
            "Bạn là Lisa, trợ lý tư vấn thông tin được tạo ra bởi IS LAB của "
            "trường Đại học Sư phạm Kỹ thuật TP.HCM. "
            "Hãy trả lời bằng tiếng Việt một cách thân thiện và chính xác.\n\n"
        )
        
        if knowledge_base:
            base_prompt += "THÔNG TIN THAM KHẢO:\n"
            base_prompt += json.dumps(knowledge_base, ensure_ascii=False, indent=2)
            base_prompt += (
                "\n\nHãy sử dụng thông tin trên để trả lời câu hỏi của người dùng khi phù hợp. "
                "Nếu thông tin không có trong dữ liệu tham khảo, hãy trả lời dựa trên kiến thức chung của bạn."
            )
        
        return base_prompt

@dataclass
class SearchResult:
    """Data class for search results"""
    path: str
    key: str
    value: str
    score: int
    result_type: str
    prompt: Optional[str] = None
    response: Optional[str] = None

class KnowledgeSearcher:
    """Handles searching within the knowledge base"""
    
    def __init__(self, knowledge_base: Dict[str, Any]):
        self.knowledge_base = knowledge_base
    
    def search(self, query: str) -> List[SearchResult]:
        """
        Search for relevant information in knowledge base.
        
        Args:
            query: User's query
            
        Returns:
            List of relevant search results
        """
        if not self.knowledge_base:
            return []
        
        query_words = query.lower().split()
        results = []
        
        if isinstance(self.knowledge_base, dict):
            # Handle FAQ data
            if "faq_data" in self.knowledge_base:
                faq_results = self._search_faq_data(query_words, self.knowledge_base["faq_data"])
                results.extend(faq_results)
            
            # Handle structured data
            if "structured_data" in self.knowledge_base:
                structured_results = self._search_structured_data(query_words, self.knowledge_base["structured_data"])
                results.extend(structured_results)
            
            # Handle other dictionary structures
            if not any(key in self.knowledge_base for key in ["faq_data", "structured_data"]):
                structured_results = self._search_structured_data(query_words, self.knowledge_base)
                results.extend(structured_results)
        
        elif isinstance(self.knowledge_base, list) and self.knowledge_base:
            if isinstance(self.knowledge_base[0], dict) and "prompt" in self.knowledge_base[0]:
                faq_results = self._search_faq_data(query_words, self.knowledge_base)
                results.extend(faq_results)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def _search_faq_data(self, query_words: List[str], faq_list: List[Dict]) -> List[SearchResult]:
        """Search in FAQ data"""
        results = []
        
        for i, item in enumerate(faq_list):
            prompt = item.get("prompt", "").lower()
            response = item.get("response", "")
            
            score = sum(
                2 if word in prompt else (1 if word in response.lower() else 0)
                for word in query_words
            )
            
            if score > 0:
                results.append(SearchResult(
                    path=f"faq_{i}",
                    key=f"Q&A_{i+1}",
                    value=f"Q: {item['prompt']}\nA: {item['response']}",
                    score=score,
                    result_type="faq",
                    prompt=item["prompt"],
                    response=item["response"]
                ))
        
        return results
    
    def _search_structured_data(self, query_words: List[str], structured_data: Any) -> List[SearchResult]:
        """Search in structured data"""
        results = []
        
        def search_recursive(data: Any, path: str = "") -> None:
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    if isinstance(value, (dict, list)):
                        search_recursive(value, current_path)
                    else:
                        if any(keyword in str(value).lower() or keyword in key.lower() 
                              for keyword in query_words):
                            results.append(SearchResult(
                                path=current_path,
                                key=key,
                                value=str(value),
                                score=1,
                                result_type="structured"
                            ))
            
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    current_path = f"{path}[{i}]" if path else f"item_{i}"
                    
                    if isinstance(item, (dict, list)):
                        search_recursive(item, current_path)
                    else:
                        if any(keyword in str(item).lower() for keyword in query_words):
                            results.append(SearchResult(
                                path=current_path,
                                key=f"item_{i}",
                                value=str(item),
                                score=1,
                                result_type="structured"
                            ))
        
        search_recursive(structured_data)
        return results

class OllamaClient:
    """Handles communication with Ollama API"""
    
    def __init__(self, api_url: str, model_name: str):
        self.api_url = api_url
        self.model_name = model_name
    
    def get_response(self, messages: List[Dict], stream: bool = True) -> Optional[str]:
        """
        Get response from Ollama API.
        
        Args:
            messages: Conversation messages
            stream: Enable streaming response
            
        Returns:
            Full response content or None if error
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "think": False
        }
        
        full_response = ""
        
        try:
            response_stream = requests.post(
                self.api_url, 
                json=payload, 
                stream=stream,
                timeout=config.TIMEOUT
            )
            response_stream.raise_for_status()
            
            print("🤖 Lisa: ", end="", flush=True)
            
            for line in response_stream.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    chunk = data.get("message", {}).get("content", "")
                    print(chunk, end="", flush=True)
                    full_response += chunk
                    
                    if data.get("done"):
                        break
            
            print()  # New line after completion
            return full_response
            
        except requests.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            print(f"\n❌ Lỗi khi gọi API Ollama: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            print("\n❌ Lỗi giải mã JSON từ API.")
            return None

class QwenInteractiveChat:
    """Main chat interface class"""
    
    def __init__(self, enable_tts: bool = True, lang: str = 'vi', data_file: str = None):
        """
        Initialize chat session.
        
        Args:
            enable_tts: Enable/disable TTS
            lang: Language for TTS
            data_file: Path to JSON data file
        """
        self.data_file = data_file or config.DATA_FILE
        self.knowledge_base = DataLoader.load_knowledge_base(self.data_file)
        self.searcher = KnowledgeSearcher(self.knowledge_base)
        self.ollama_client = OllamaClient(config.OLLAMA_API_URL, config.MODEL_NAME)
        
        # Initialize conversation with system prompt
        system_prompt = PromptBuilder.create_system_prompt(self.knowledge_base)
        self.messages = [{"role": "system", "content": system_prompt}]
        
        # Setup TTS
        self.enable_tts = enable_tts
        if self.enable_tts:
            try:
                self.tts = VietnameseTTS(lang=lang)
            except Exception as e:
                logger.warning(f"TTS initialization failed: {e}")
                self.tts = None
                self.enable_tts = False
        else:
            self.tts = None
    
    def reload_knowledge_base(self) -> None:
        """Reload knowledge base from file"""
        self.knowledge_base = DataLoader.load_knowledge_base(self.data_file)
        self.searcher = KnowledgeSearcher(self.knowledge_base)
        
        system_prompt = PromptBuilder.create_system_prompt(self.knowledge_base)
        self.messages[0] = {"role": "system", "content": system_prompt}
        
        print("🔄 Đã tải lại dữ liệu và cập nhật system prompt")
        logger.info("Knowledge base reloaded successfully")
    
    def _enhance_user_query(self, user_input: str) -> str:
        """
        Enhance user query with relevant knowledge base context.
        
        Args:
            user_input: Original user query
            
        Returns:
            Enhanced query with context
        """
        if not self.knowledge_base:
            return user_input
        
        relevant_info = self.searcher.search(user_input)
        
        if not relevant_info:
            return user_input
        
        enhanced_query = f"Câu hỏi của người dùng: {user_input}\n\n"
        
        # Separate FAQ and structured results
        faq_info = [info for info in relevant_info if info.result_type == "faq"]
        structured_info = [info for info in relevant_info if info.result_type == "structured"]
        
        # Add FAQ information (higher priority)
        if faq_info:
            enhanced_query += "Các câu hỏi và câu trả lời tương tự từ FAQ:\n"
            for i, info in enumerate(faq_info[:config.MAX_FAQ_RESULTS]):
                enhanced_query += f"\n{i+1}. Câu hỏi: {info.prompt}\n"
                enhanced_query += f"   Trả lời: {info.response}\n"
        
        # Add structured information
        if structured_info:
            enhanced_query += "\nThông tin chi tiết từ cơ sở dữ liệu:\n"
            for info in structured_info[:config.MAX_STRUCTURED_RESULTS]:
                enhanced_query += f"- {info.key}: {info.value}\n"
        
        enhanced_query += f"\nDựa trên thông tin trên, hãy trả lời câu hỏi: '{user_input}' một cách tự nhiên và chính xác."
        return enhanced_query
    
    def _remove_think_blocks(self, text: str) -> str:
        """Remove think blocks from response"""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    def show_help(self) -> None:
        """Display usage instructions"""
        help_text = """
📋 HƯỚNG DẪN SỬ DỤNG:
• Gõ câu hỏi bình thường để chat
• 'reload' - Tải lại dữ liệu từ JSON
• 'info' - Xem thông tin về dữ liệu đã tải
• 'help' - Hiển thị hướng dẫn này
• 'quit' hoặc 'exit' - Thoát chương trình
"""
        print(help_text)
    
    def show_data_info(self) -> None:
        """Display information about loaded data"""
        if not self.knowledge_base:
            print("📊 Chưa có dữ liệu nào được tải.")
            return
        
        print("📊 THÔNG TIN DỮ LIỆU:")
        
        def count_items(data: Any, depth: int = 0) -> int:
            """Recursively count items in nested data structures"""
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
        
        if isinstance(self.knowledge_base, dict):
            print(f"• Loại dữ liệu: Dictionary")
            print(f"• Số lượng key chính: {len(self.knowledge_base)}")
            print(f"• Các key chính: {list(self.knowledge_base.keys())}")
            
            # Check for hybrid structure
            if "faq_data" in self.knowledge_base and "structured_data" in self.knowledge_base:
                print("• Định dạng: Hybrid (FAQ + Structured Data)")
                
                faq_data = self.knowledge_base.get("faq_data", [])
                if isinstance(faq_data, list):
                    print(f"• Số câu hỏi FAQ: {len(faq_data)}")
                    if faq_data:
                        print("• Một số câu hỏi FAQ mẫu:")
                        for i, item in enumerate(faq_data[:3]):
                            print(f"  {i+1}. {item.get('prompt', 'N/A')}")
                        if len(faq_data) > 3:
                            print(f"  ... và {len(faq_data) - 3} câu hỏi khác")
                
                structured_data = self.knowledge_base.get("structured_data", {})
                if isinstance(structured_data, dict):
                    print(f"• Structured data keys: {list(structured_data.keys())}")
            
            elif "faq_data" in self.knowledge_base:
                print("• Định dạng: FAQ only")
                faq_data = self.knowledge_base.get("faq_data", [])
                print(f"• Số câu hỏi FAQ: {len(faq_data)}")
            
            elif "structured_data" in self.knowledge_base:
                print("• Định dạng: Structured Data only")
                structured_data = self.knowledge_base.get("structured_data", {})
                print(f"• Structured data keys: {list(structured_data.keys())}")
            
            else:
                print("• Định dạng: Custom Dictionary")
        
        elif isinstance(self.knowledge_base, list):
            print(f"• Loại dữ liệu: List")
            print(f"• Số lượng items: {len(self.knowledge_base)}")
            
            if (self.knowledge_base and isinstance(self.knowledge_base[0], dict) and
                "prompt" in self.knowledge_base[0] and "response" in self.knowledge_base[0]):
                print("• Định dạng: FAQ (Prompt-Response)")
                print(f"• Số câu hỏi có sẵn: {len(self.knowledge_base)}")
                
                print("• Một số câu hỏi mẫu:")
                for i, item in enumerate(self.knowledge_base[:3]):
                    print(f"  {i+1}. {item.get('prompt', 'N/A')}")
                if len(self.knowledge_base) > 3:
                    print(f"  ... và {len(self.knowledge_base) - 3} câu hỏi khác")
        else:
            print(f"• Loại dữ liệu: {type(self.knowledge_base).__name__}")
        
        total_items = count_items(self.knowledge_base)
        print(f"• Tổng số items: {total_items}")
        print()
    
    def start(self) -> None:
        """Start interactive chat loop"""
        print("🎯 Chatbot Lisa với dữ liệu JSON đã sẵn sàng!")
        print("   (Gõ 'help' để xem hướng dẫn)")
        print("=" * 50)
        
        if self.knowledge_base:
            self.show_data_info()
        
        try:
            while True:
                user_input = input("👤 Bạn: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
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
                
                # Process query
                enhanced_query = self._enhance_user_query(user_input)
                self.messages.append({"role": "user", "content": enhanced_query})
                
                # Get response from Ollama
                qwen_response = self.ollama_client.get_response(self.messages)
                
                if qwen_response:
                    clean_response = self._remove_think_blocks(qwen_response)
                    
                    # Add to conversation history
                    self.messages.append({"role": "assistant", "content": clean_response})
                    
                    # Play TTS if enabled
                    if self.enable_tts and self.tts:
                        try:
                            print("🔊 Đang phát giọng nói...")
                            self.tts.text_to_speech(clean_response, auto_play=True, auto_delete=True)
                        except Exception as e:
                            logger.warning(f"TTS playback failed: {e}")
                
                print("-" * 50)
                
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Tạm biệt!")
        finally:
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Clean up resources"""
        if self.enable_tts and self.tts:
            try:
                self.tts.cleanup()
            except Exception as e:
                logger.warning(f"TTS cleanup failed: {e}")

def create_sample_data_file(file_path: str = None) -> None:
    """Create sample data file if it doesn't exist"""
    file_path = file_path or config.DATA_FILE
    
    if os.path.exists(file_path):
        print(f"⚠️ File {file_path} đã tồn tại")
        return
    
    sample_data = {
        "faq_data": [
            {
                "prompt": "Đại học Sư phạm Kỹ thuật TP.HCM có bao nhiêu ngành?",
                "response": "Đại học Sư phạm Kỹ thuật TP.HCM hiện có hơn 40 ngành đào tạo ở các lĩnh vực như kỹ thuật, công nghệ, sư phạm kỹ thuật, kinh tế và ngoại ngữ. Để biết chính xác số ngành, bạn có thể tham khảo trên trang web chính thức của trường tại https://hcmute.edu.vn."
            },
            {
                "prompt": "Đại học Sư phạm Kỹ thuật TP.HCM có ngành Công nghệ thông tin không?",
                "response": "Có, Đại học Sư phạm Kỹ thuật TP.HCM có ngành Công nghệ Thông tin, thuộc Khoa Công nghệ Thông tin. Ngành này đào tạo về lập trình, phát triển phần mềm, trí tuệ nhân tạo, và các lĩnh vực liên quan."
            },
            {
                "prompt": "Đại học Sư phạm Kỹ thuật TP.HCM có ngành nào liên quan đến kỹ thuật ô tô?",
                "response": "Có, trường có ngành Kỹ thuật Ô tô, thuộc Khoa Cơ khí - Động lực, tập trung vào thiết kế, chế tạo, bảo trì và sửa chữa ô tô."
            },
            {
                "prompt": "Bạn là ai?",
                "response": "Tôi là Lisa, trợ lý được tạo ra bởi IS LAB của Trường Đại học Sư phạm Kỹ thuật TP.HCM."
            }
        ],
        "structured_data": {
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
    }
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã tạo file mẫu {file_path} với cả FAQ và structured data")
    except Exception as e:
        logger.error(f"Error creating sample data file: {e}")
        print(f"❌ Lỗi khi tạo file mẫu: {e}")

def main():
    """Main function"""
    # Create sample data file if it doesn't exist
    create_sample_data_file()
    
    # Initialize and start chat session
    try:
        chat_session = QwenInteractiveChat(enable_tts=True)
        chat_session.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Lỗi nghiêm trọng: {e}")

if __name__ == "__main__":
    main()