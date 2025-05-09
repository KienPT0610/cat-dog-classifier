import os
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads' # thư mục lưu trữ ảnh
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Đảm bảo thư mục uploads tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Kiểm tra phần mở rộng tệp hợp lệ
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Tải mô hình
def load_model():
    model = models.resnet50(weights=None)
    num_classes = 2  # Số lượng lớp trong mô hình (2 lớp: cat và dog)
    model.fc = nn.Linear(model.fc.in_features, num_classes) 
    model_path = 'best_model.pth' # đọc file pth đã huấn luyện
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # chuyển mô hình sang chế độ đánh giá
    return model

# Tiền xử lý hình ảnh
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # chuyển đổi kích thước ảnh
        transforms.ToTensor(), # chuyển đổi ảnh sang tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # chuẩn hóa ảnh
    ])
    return transform(image).unsqueeze(0)

# Dự đoán hình ảnh
def predict_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_image(image)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted_class = torch.max(output, 1)
    
    return predicted_class.item(), probabilities[0]

# Tải mô hình khi khởi động
model = load_model()
class_names = ['cat', 'dog']  # Cập nhật với tên lớp thực tế của bạn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file nào được tải lên'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Không có file nào được chọn'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            class_idx, probabilities = predict_image(file_path, model)
            class_name = class_names[class_idx]
            confidence = probabilities[class_idx].item() * 100
            
            # Tạo kết quả cho tất cả các lớp
            all_probs = {}
            for i, name in enumerate(class_names):
                all_probs[name] = float(probabilities[i].item() * 100)
            
            # Xóa file ảnh sau khi dự đoán
            os.remove(file_path)
            
            return jsonify({
                'prediction': class_name,
                'confidence': float(confidence),
                'probabilities': all_probs
            })
        
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'File không được cho phép'})

if __name__ == '__main__':
    # Sử dụng debug=True trong môi trường phát triển, False trong production
    debug_mode = os.environ.get('FLASK_DEBUG', 'True') == 'True'
    host = '0.0.0.0'  # Cho phép truy cập từ bên ngoài
    port = int(os.environ.get('PORT', 5000))
    app.run(host=host, port=port, debug=debug_mode) 