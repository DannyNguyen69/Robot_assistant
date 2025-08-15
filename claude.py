# =============================================================================
# QWEN3-0.6B FINE-TUNING TRÊN GOOGLE COLAB
# Hướng dẫn chi tiết từ A-Z cho người mới bắt đầu
# =============================================================================

# =============================================================================
# BƯỚC 1: KIỂM TRA VÀ CÀI ĐẶT MÔI TRƯỜNG
# =============================================================================

print("🔍 KIỂM TRA MÔI TRƯỜNG GOOGLE COLAB")
print("=" * 50)

# Kiểm tra GPU
!nvidia-smi

# Kiểm tra Python version
import sys
print(f"🐍 Python version: {sys.version}")

# Kiểm tra packages hiện tại
import pkg_resources
installed_packages = [d.project_name for d in pkg_resources.working_set]
key_packages = ['torch', 'transformers', 'datasets', 'accelerate']

print("\n📦 Packages hiện tại:")
for pkg in key_packages:
    if pkg in installed_packages:
        version = pkg_resources.get_distribution(pkg).version
        print(f"   ✅ {pkg}: {version}")
    else:
        print(f"   ❌ {pkg}: chưa cài đặt")

print("\n" + "=" * 50)

# =============================================================================
# BƯỚC 2: GỠ CÀI ĐẶT PACKAGES CŨ VÀ CÀI ĐẶT MỚI
# =============================================================================

print("🧹 DỌNG DẸP VÀ CÀI ĐẶT PACKAGES MỚI")
print("=" * 50)

# Gỡ cài đặt packages cũ để tránh xung đột
print("🗑️ Gỡ cài đặt packages cũ...")
!pip uninstall -y torch torchvision torchaudio transformers datasets accelerate peft -q

# Cài đặt PyTorch với CUDA 12.1 support
print("🔥 Cài đặt PyTorch với CUDA support...")
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# Cài đặt Transformers version mới nhất hỗ trợ Qwen3
print("🤖 Cài đặt Transformers >= 4.51.0 cho Qwen3...")
!pip install transformers>=4.51.0 -q

# Cài đặt các dependencies khác
print("📚 Cài đặt datasets, accelerate, tokenizers...")
!pip install datasets>=2.21.0 accelerate>=0.34.0 tokenizers>=0.19.1 -q

# Cài đặt PEFT cho parameter-efficient fine-tuning
print("⚡ Cài đặt PEFT...")
!pip install peft>=0.12.0 -q

# Cài đặt thêm các utilities
print("🛠️ Cài đặt utilities...")
!pip install packaging tqdm -q

print("\n✅ CÀI ĐẶT HOÀN TẤT!")
print("⚠️  QUAN TRỌNG: RESTART RUNTIME NGAY BÂY GIỜ!")
print("   👆 Vào: Runtime > Restart session")
print("   🔄 Sau đó chạy cell tiếp theo")

# =============================================================================
# BƯỚC 3: XÁC MINH CÀI ĐẶT (CHẠY SAU KHI RESTART)
# =============================================================================

print("🔍 XÁC MINH CÀI ĐẶT SAU RESTART")
print("=" * 50)

import torch
import transformers
import datasets
import accelerate
from packaging import version

# Hiển thị thông tin versions
print("📋 THÔNG TIN PACKAGES:")
print(f"   🔥 PyTorch: {torch.__version__}")
print(f"   🤖 Transformers: {transformers.__version__}")
print(f"   📊 Datasets: {datasets.__version__}")
print(f"   ⚡ Accelerate: {accelerate.__version__}")

# Kiểm tra CUDA
print(f"\n💻 THÔNG TIN PHẦN CỨNG:")
print(f"   🎮 CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   🎯 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   🔢 CUDA version: {torch.version.cuda}")
    print(f"   💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   🆔 GPU Count: {torch.cuda.device_count()}")

# Kiểm tra version Transformers
required_version = "4.51.0"
current_version = transformers.__version__

print(f"\n🔍 KIỂM TRA COMPATIBILITY:")
if version.parse(current_version) >= version.parse(required_version):
    print(f"   ✅ Transformers OK: {current_version} >= {required_version}")
else:
    print(f"   ❌ Transformers cũ: {current_version} < {required_version}")
    print("   🔧 Chạy: !pip install transformers>=4.51.0 --upgrade")

# Test imports quan trọng
print(f"\n🧪 TEST IMPORTS:")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import Dataset
    print("   ✅ Tất cả imports thành công!")
except ImportError as e:
    print(f"   ❌ Import error: {e}")

print("\n🎉 SẴN SÀNG BẮT ĐẦU FINE-TUNING!")

# =============================================================================
# BƯỚC 4: TẢI MODEL QWEN3-0.6B
# =============================================================================

print("🤖 TẢI QWEN3-0.6B MODEL")
print("=" * 50)

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
import gc
import json

# Dọn cache GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 Đã dọn cache GPU")

# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Sử dụng device: {device}")

# Model configuration
model_name = "Qwen/Qwen3-0.6B"
print(f"📥 Đang tải {model_name}...")

try:
    # Tải tokenizer
    print("🔤 Đang tải tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        use_fast=True
    )
    print("   ✅ Tokenizer đã tải xong")
    
    # Tải model với cấu hình tối ưu cho Colab
    print("🧠 Đang tải model (có thể mất vài phút)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    print("   ✅ Model đã tải xong")
    
    # Cấu hình tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("   ✅ Đã cấu hình pad token")
    
    # Thông tin model
    print(f"\n📊 THÔNG TIN MODEL:")
    print(f"   🔢 Số parameters: {model.num_parameters():,}")
    print(f"   💻 Device: {next(model.parameters()).device}")
    print(f"   🎯 Data type: {next(model.parameters()).dtype}")
    print(f"   🏗️ Model type: {model.config.model_type}")
    
    # Thông tin GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   💾 GPU memory allocated: {allocated:.2f} GB")
        print(f"   💾 GPU memory reserved: {reserved:.2f} GB")
    
    print("\n✅ MODEL ĐÃ SẴN SÀNG CHO FINE-TUNING!")
    
except Exception as e:
    print(f"❌ Lỗi khi tải model: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# BƯỚC 5: CHUẨN BỊ DỮ LIỆU TRAINING
# =============================================================================

print("\n📚 CHUẨN BỊ DỮ LIỆU TRAINING")
print("=" * 50)

# Dữ liệu training với nội dung tiếng Việt đa dạng
training_data = [
    {
        "input": "Xin chào, bạn có thể giới thiệu về bản thân không?",
        "output": "Chào bạn! Tôi là Qwen3, một AI assistant thông minh được phát triển bởi Alibaba Cloud. Tôi có thể giúp bạn trả lời câu hỏi, viết code, phân tích dữ liệu, dịch thuật và nhiều tác vụ khác. Tôi được fine-tune để hiểu và giao tiếp bằng tiếng Việt một cách tự nhiên. Bạn cần tôi giúp gì không?"
    },
    {
        "input": "Viết một hàm Python tính số Fibonacci",
        "output": "Đây là các cách tính số Fibonacci trong Python:\n\n```python\n# Phương pháp đệ quy đơn giản\ndef fibonacci_recursive(n):\n    if n <= 1:\n        return n\n    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)\n\n# Phương pháp tối ưu với memoization\ndef fibonacci_memo(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        memo[n] = n\n    else:\n        memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)\n    return memo[n]\n\n# Phương pháp lặp (hiệu quả nhất)\ndef fibonacci_iterative(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\n# Test\nfor i in range(10):\n    print(f'F({i}) = {fibonacci_iterative(i)}')\n```"
    },
    {
        "input": "Machine Learning là gì? Giải thích cho người mới bắt đầu",
        "output": "Machine Learning (ML) là một nhánh của trí tuệ nhân tạo cho phép máy tính học từ dữ liệu:\n\n**Ví dụ đơn giản:**\n- Thay vì lập trình 'nếu email có từ spam thì là thư rác'\n- ML sẽ học từ hàng nghìn email để tự nhận diện spam\n\n**3 loại chính:**\n1. **Supervised Learning**: Học từ dữ liệu có nhãn\n2. **Unsupervised Learning**: Tìm pattern trong dữ liệu\n3. **Reinforcement Learning**: Học qua thử-sai và reward\n\n**Ứng dụng:** Gợi ý sản phẩm, nhận dạng giọng nói, chatbot, dự đoán giá cổ phiếu...\n\nBạn muốn tìm hiểu sâu về phần nào?"
    },
    {
        "input": "Cách học lập trình hiệu quả cho người mới?",
        "output": "Lộ trình học lập trình hiệu quả:\n\n**Bước 1: Chọn ngôn ngữ đầu tiên (2-4 tuần)**\n- Python: dễ học, ứng dụng rộng\n- JavaScript: cho web development\n- Java: cho enterprise apps\n\n**Bước 2: Nắm vững nền tảng (4-8 tuần)**\n- Biến, vòng lặp, điều kiện\n- Functions và data structures\n- Debugging và error handling\n\n**Bước 3: Thực hành project (8-12 tuần)**\n- Calculator, todo app\n- Web scraper, mini game\n- Database interaction\n\n**Tips thành công:**\n- Code mỗi ngày ít nhất 1 giờ\n- Tham gia community (Stack Overflow)\n- Đọc code người khác trên GitHub\n- Đừng chỉ xem video, hãy code theo\n\nKiên trì là chìa khóa!"
    },
    {
        "input": "Giải thích thuật toán Quick Sort",
        "output": "Quick Sort sử dụng phương pháp 'divide and conquer':\n\n**Nguyên lý:**\n1. Chọn một phần tử làm pivot\n2. Partition: đưa phần tử < pivot về trái, > pivot về phải\n3. Recursively sort các phần con\n\n**Code Python:**\n```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    \n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    \n    return quicksort(left) + middle + quicksort(right)\n\n# Test\narr = [3, 6, 8, 10, 1, 2, 1]\nsorted_arr = quicksort(arr)\nprint(sorted_arr)  # [1, 1, 2, 3, 6, 8, 10]\n```\n\n**Độ phức tạp:**\n- Best/Average: O(n log n)\n- Worst case: O(n²)\n- Space: O(log n)"
    },
    {
        "input": "REST API là gì? Cách thiết kế API tốt?",
        "output": "REST API là kiến trúc thiết kế web services:\n\n**Nguyên tắc REST:**\n- Stateless: mỗi request độc lập\n- Resource-based: URL đại diện cho resources\n- HTTP methods: GET, POST, PUT, DELETE\n- JSON format cho data\n\n**Ví dụ thiết kế:**\n```\nGET /api/users          # Lấy danh sách users\nGET /api/users/123      # Lấy user có id=123\nPOST /api/users         # Tạo user mới\nPUT /api/users/123      # Update user 123\nDELETE /api/users/123   # Xóa user 123\n```\n\n**Best practices:**\n- Sử dụng nouns trong URL\n- Proper HTTP status codes\n- Versioning: /api/v1/users\n- Authentication & authorization\n- Rate limiting\n- Documentation (Swagger)"
    },
    {
        "input": "Docker là gì và tại sao nên sử dụng?",
        "output": "Docker là platform containerization:\n\n**Container vs VM:**\n- Container: Chia sẻ OS kernel, nhẹ hơn\n- VM: Mỗi VM có OS riêng, nặng hơn\n\n**Ưu điểm Docker:**\n- Lightweight và fast startup\n- Portable: 'Build once, run anywhere'\n- Isolated: Ứng dụng cô lập\n- Scalable: Dễ scale up/down\n\n**Basic commands:**\n```bash\ndocker pull nginx\ndocker run -p 8080:80 nginx\ndocker ps\ndocker stop container_id\n```\n\n**Dockerfile example:**\n```dockerfile\nFROM node:18-alpine\nWORKDIR /app\nCOPY package*.json ./\nRUN npm install\nCOPY . .\nEXPOSE 3000\nCMD ['npm', 'start']\n```"
    },
    {
        "input": "Blockchain hoạt động như thế nào?",
        "output": "Blockchain là sổ cái phân tán ghi lại giao dịch:\n\n**Cấu trúc:**\n- Chuỗi các block liên kết bằng hash\n- Mỗi block chứa: hash, previous hash, transactions, timestamp\n- Immutable: Không thể thay đổi sau khi confirm\n\n**Đặc điểm:**\n- Decentralized: Không có cơ quan trung ương\n- Transparent: Mọi giao dịch công khai\n- Secure: Mã hóa cryptographic\n\n**Consensus mechanisms:**\n- Proof of Work (Bitcoin): Miners giải puzzle\n- Proof of Stake (Ethereum 2.0): Validators stake token\n\n**Ứng dụng:**\n- Cryptocurrency (Bitcoin, Ethereum)\n- Smart contracts\n- Supply chain tracking\n- Digital identity\n- Voting systems"
    }
]

print(f"📊 Đã tạo {len(training_data)} ví dụ training")

# =============================================================================
# BƯỚC 6: FORMAT DỮ LIỆU CHO QWEN3
# =============================================================================

def format_qwen3_chat(input_text, output_text):
    """Format dữ liệu theo Qwen3 chat format"""
    return f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"

# Format training data
print("🔄 Đang format dữ liệu cho Qwen3...")
formatted_texts = []
for item in training_data:
    text = format_qwen3_chat(item['input'], item['output'])
    formatted_texts.append(text)

print("📝 Ví dụ formatted text:")
print(formatted_texts[0][:200] + "...")

# Function tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,  # Sử dụng 1K tokens cho training nhanh hơn
        return_tensors="pt"
    )

# Tạo dataset
try:
    print("📊 Đang tạo dataset...")
    dataset = Dataset.from_dict({"text": formatted_texts})
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )
    print(f"✅ Dataset đã tạo: {len(tokenized_dataset)} examples")
    print(f"📏 Sample token length: {len(tokenized_dataset[0]['input_ids'])}")
    
except Exception as e:
    print(f"❌ Lỗi tạo dataset: {e}")

# =============================================================================
# BƯỚC 7: THIẾT LẬP TRAINING
# =============================================================================

print("\n⚙️ THIẾT LẬP TRAINING CONFIGURATION")
print("=" * 50)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Không sử dụng masked language modeling
)

# Training arguments được tối ưu cho Colab
training_args = TrainingArguments(
    output_dir="./qwen3_colab_results",
    overwrite_output_dir=True,
    
    # Training parameters
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Batch size nhỏ cho Colab
    gradient_accumulation_steps=4,   # Tăng effective batch size
    
    # Learning rate
    learning_rate=1e-5,              # Learning rate thấp cho Qwen3
    warmup_steps=100,
    lr_scheduler_type="cosine",
    
    # Optimization
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="adamw_torch",
    
    # Memory optimization
    gradient_checkpointing=True,     # Tiết kiệm memory
    fp16=torch.cuda.is_available(),  # Mixed precision training
    bf16=False,
    dataloader_num_workers=0,        # Tránh lỗi multiprocessing trên Colab
    
    # Logging và saving
    logging_steps=5,
    save_steps=100,
    save_total_limit=2,
    prediction_loss_only=True,
    remove_unused_columns=False,
    
    # Misc
    report_to=None,                  # Tắt wandb/tensorboard
    load_best_model_at_end=False,
    ddp_find_unused_parameters=False,
)

# Tạo trainer
try:
    print("🏗️ Đang tạo Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    print("✅ Trainer đã sẵn sàng!")
    
    # Hiển thị cấu hình training
    print(f"\n📋 TRAINING CONFIGURATION:")
    print(f"   🎯 Model: {model_name}")
    print(f"   💻 Device: {device}")
    print(f"   📊 Training examples: {len(tokenized_dataset)}")
    print(f"   🔄 Epochs: {training_args.num_train_epochs}")
    print(f"   📦 Batch size: {training_args.per_device_train_batch_size}")
    print(f"   ⚡ Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"   📈 Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"   📚 Learning rate: {training_args.learning_rate}")
    print(f"   🎯 Mixed precision: {training_args.fp16}")
    
except Exception as e:
    print(f"❌ Lỗi tạo trainer: {e}")

# =============================================================================
# BƯỚC 8: BẮT ĐẦU TRAINING
# =============================================================================

print("\n" + "=" * 60)
print("🚀 BẮT ĐẦU QWEN3-0.6B FINE-TUNING TRÊN COLAB")
print("=" * 60)

try:
    # Dọn cache trước khi training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("⏰ Training bắt đầu... (có thể mất 10-30 phút)")
    print("📊 Theo dõi loss để xem progress")
    print("-" * 40)
    
    # Bắt đầu training
    training_output = trainer.train()
    
    print("\n" + "=" * 40)
    print("🎉 TRAINING HOÀN THÀNH THÀNH CÔNG!")
    print("=" * 40)
    print(f"📊 Final training loss: {training_output.training_loss:.4f}")
    
except Exception as e:
    print(f"❌ Lỗi trong quá trình training: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# BƯỚC 9: LƯU MODEL
# =============================================================================

print("\n💾 LƯU MODEL ĐÃ FINE-TUNE")
print("=" * 50)

output_dir = "./qwen3_0.6b_finetuned"

try:
    print(f"📁 Đang lưu model vào {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Model đã được lưu thành công!")
    
    # Lưu vào Google Drive (optional)
    print("\n☁️ Lưu vào Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        
        import shutil
        drive_path = '/content/drive/MyDrive/qwen3_0.6b_finetuned'
        
        # Tạo thư mục nếu chưa có
        import os
        os.makedirs(os.path.dirname(drive_path), exist_ok=True)
        
        # Copy model
        if os.path.exists(drive_path):
            shutil.rmtree(drive_path)
        shutil.copytree(output_dir, drive_path)
        
        print(f"✅ Model cũng đã được lưu vào Google Drive: {drive_path}")
        
    except Exception as drive_error:
        print(f"⚠️ Không thể lưu vào Google Drive: {drive_error}")
        print("💡 Bạn có thể download model từ thư mục ./qwen3_0.6b_finetuned")
    
except Exception as e:
    print(f"❌ Lỗi khi lưu model: {e}")

# =============================================================================
# BƯỚC 10: TEST MODEL ĐÃ FINE-TUNE
# =============================================================================

print("\n🧪 TEST QWEN3 MODEL ĐÃ FINE-TUNE")
print("=" * 50)

def test_qwen3(prompt, max_new_tokens=200, temperature=0.7):
    """Test Qwen3 với prompt"""
    # Format prompt theo Qwen3 chat format
    full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = full_response[len(full_prompt):].split("<|im_end|>")[0].strip()
    return response

# Test cases
test_prompts = [
    "Xin chào, bạn khỏe không?",
    "Viết code Python đọc file CSV",
    "Giải thích về Neural Networks",
    "Cách tối ưu database query?",
    "So sánh Python và JavaScript",
    "Tôi mới học lập trình, bắt đầu từ đâu?"
]

print("🔍 Đang test model với các câu hỏi mẫu...\n")

for i, prompt in enumerate(test_prompts, 1):
    print(f"🔍 Test {i}/6:")
    print(f"👤 Input: {prompt}")
    print(f"🤖 Qwen3 Output:")
    
    try:
        response = test_qwen3(prompt)
        print(response)
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("-" * 50)

print("🏁 HOÀN THÀNH TẤT CẢ TESTS!")

# =============================================================================
# BƯỚC 11: HƯỚNG DẪN SỬ DỤNG MODEL
# =============================================================================

print("\n📖 HƯỚNG DẪN SỬ DỤNG MODEL ĐÃ FINE-TUNE")
print("=" * 50)

usage_guide = """
🎉 CHÚC MỪNG! Bạn đã fine-tune thành công Qwen3-0.6B!

📁 MODEL LOCATION:
   - Local: ./qwen3_0.6b_finetuned/
   - Google Drive: /content/drive/MyDrive/qwen3_0.6b_finetuned/

💡 CÁCH SỬ DỤNG MODEL:

1. Load model đã fine-tune: