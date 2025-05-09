# Ứng Dụng Phân Loại Ảnh Mèo và Chó

Ứng dụng web này sử dụng mô hình Deep Learning (ResNet50) để phân loại hình ảnh mèo và chó. Người dùng có thể tải lên hình ảnh và nhận kết quả phân loại kèm theo xác suất dự đoán.

## Demo

![Demo ứng dụng](demo.gif)

## Tính năng

- ✅ Phân loại hình ảnh thành mèo hoặc chó
- ✅ Giao diện người dùng thân thiện với khả năng kéo và thả
- ✅ Hiển thị xác suất dự đoán cho từng lớp
- ✅ Xử lý lỗi và thông báo cho người dùng
- ✅ Tự động xóa ảnh đã tải lên sau khi xử lý

## Cài đặt và Sử dụng

### Yêu cầu

- Python 3.8+
- PyTorch 2.0+
- Flask 3.0+
- Các thư viện phụ thuộc khác (xem file requirements.txt)

### Cài đặt

1. Clone repository:

```
git clone https://github.com/KienPT0610/cat-dog-classifier.git
cd cat-dog-classifier
```

2. Cài đặt các gói phụ thuộc:

```
pip install -r requirements.txt
```

### Chạy ứng dụng

```
python web_app.py
```

Mở trình duyệt và truy cập http://127.0.0.1:5000/

## Cách sử dụng

1. Kéo và thả hình ảnh vào khu vực tải lên, hoặc nhấp để chọn file
2. Nhấn nút "Dự đoán" để xử lý ảnh
3. Xem kết quả phân loại và xác suất

## Kiến trúc

### Mô hình

- **Mạng Neural**: ResNet50 (được huấn luyện trước trên ImageNet và fine-tuned)
- **Lớp đầu ra**: 2 lớp (mèo và chó)

### Tiền xử lý

Các ảnh đầu vào được xử lý như sau:

- Thay đổi kích thước thành 224x224 pixels
- Chuyển đổi thành tensor
- Chuẩn hóa với mean=[0.485, 0.456, 0.406] và std=[0.229, 0.224, 0.225]
