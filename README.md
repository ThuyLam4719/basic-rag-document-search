Chatbot đọc các file PDF trong `docs/`, tìm đoạn liên quan rồi tạo câu trả lời bằng model chat Ollama. 
## Cài đặt
1. Cài Ollama và mở ứng dụng Ollama.

2. Tải các model cần dùng:
   ollama pull nomic-embed-text:v1.5
   ollama pull qwen3:4b

3. Cài các thư viện Python:
   pip install -r requirements.txt


4. Đặt các file PDF vào thư mục `docs/` rồi chạy:
   python loader.py

Gõ `q` để kết thúc. 
