import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. TIỀN XỬ LÝ ẢNH BẰNG OPENCV
# ==========================================

def crop_brain_contour(image):
    """
    Tiền xử lý ảnh nâng cao bằng OpenCV:
    1. Chuyển sang ảnh xám (Grayscale)
    2. Lọc mờ (Gaussian Blur) để loại bỏ nhiễu
    3. Phân ngưỡng (Thresholding) tách biệt hộp sọ
    4. Tìm và trích xuất đường bao (Contours) lớn nhất
    5. Cắt (Crop) bỏ phần nền đen bao quanh hộp sọ
    """
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Lọc nhiễu
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Phân ngưỡng nhị phân để tạo mặt nạ hộp sọ
    _, thresh = cv2.threshold(gray_blurred, 45, 255, cv2.THRESH_BINARY)
    
    # Loại bỏ các đốm nhiễu nhỏ bằng Phép xói mòn (Erosion) và Phình giãn (Dilation)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Tìm các đường bao quanh các vùng sáng
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return image  # Trả về ảnh gốc nếu không tìm thấy contour
        
    # Lấy đường bao lớn nhất (chính là hộp sọ)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Tìm 4 điểm cực trị (trên, dưới, trái, phải) của đường bao để cắt
    ext_left = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    ext_right = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
    ext_top = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
    ext_bot = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
    
    # Cắt ảnh theo các điểm cực trị này
    new_image = image[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
    return new_image

def preprocess_image_for_model(img, img_size=(224, 224)):
    """
    Tiền xử lý ảnh toàn diện: Cắt biên -> Resize -> Chuẩn hóa định dạng PyTorch
    """
    # 1. Cắt phần viền đen bằng OpenCV
    cropped_img = crop_brain_contour(img)
    
    # 2. Resize về kích thước chuẩn đầu vào của CNN
    resized_img = cv2.resize(cropped_img, img_size, interpolation=cv2.INTER_CUBIC)
    
    # 3. Chuẩn hóa giá trị pixel về khoảng [0, 1] giống như chuẩn hóa tập Train ở Lab 7
    normalized_img = resized_img.astype(np.float32) / 255.0
    
    # 4. Chuyển đổi từ định dạng HWC (OpenCV) sang CHW (PyTorch) và thêm Batch Dimension
    # PyTorch yêu cầu đầu vào dạng: [Batch, Channels, Height, Width]
    tensor_img = np.transpose(normalized_img, (2, 0, 1)) # (3, 224, 224)
    tensor_img = torch.tensor(tensor_img).unsqueeze(0) # (1, 3, 224, 224)
    
    return cropped_img, resized_img, tensor_img


# ==========================================
# 2. MẠNG NƠ-RON TÍCH CHẬP
# ==========================================

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()
        
        # --- BLOCK TÍCH CHẬP 1 ---
        # Conv2d đóng vai trò như bộ quét đặc trưng (Feature Scanner)
        # Giúp tự động nhận diện các đường cạnh, kết cấu hình ảnh
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Giảm 1/2 kích thước ảnh
        
        # --- BLOCK TÍCH CHẬP 2 ---
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # --- BLOCK TÍCH CHẬP 3 (Lớp chứa thông tin không gian cuối cùng trước khi phân loại) ---
        # Đây chính là lớp cuối cùng chúng ta sẽ dùng để chạy Grad-CAM
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Adaptive pooling giúp chuyển đổi đầu ra về kích thước cố định (7x7) bất kể kích thước ảnh gốc
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # --- CÁC TẦNG PHẲNG / TUYẾN TÍNH (Chính là MLP trong Lab 7) ---
        # Tại đây, chúng ta "duỗi phẳng" ma trận đặc trưng thành một mảng số dài để đưa vào phân loại nhị phân/đa lớp
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # Lớp ẩn 128 nơ-ron
        self.dropout = nn.Dropout(0.3)        # Ngăn chặn Overfitting (Học vẹt)
        self.fc2 = nn.Linear(128, num_classes) # Lớp đầu ra 4 nơ-ron ứng với 4 loại bệnh
        
    def forward(self, x):
        # Đi qua Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Đi qua Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Đi qua Block 3 (Lớp trích xuất quan trọng nhất)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Thu nhỏ ma trận
        x = self.avgpool(x)
        
        # Duỗi phẳng (Flatten) giống như chuẩn hóa đầu vào Lab 7
        x = x.view(x.size(0), -1)
        
        # Đi qua các tầng ẩn và hàm kích hoạt ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Lớp đầu ra sinh ra các giá trị điểm số chưa kích hoạt (logits)
        x = self.fc2(x)
        return x


# ==========================================
# 3. GIẢI THUẬT GRAD-CAM
# ==========================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Thiết lập các hàm "Hook" để tự động thu thập đạo hàm (gradients) và kích hoạt (activations)
        # của lớp tích chập cuối cùng khi chạy Lan truyền xuôi & ngược.
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_heatmap(self, input_tensor, class_idx):
        """
        Thuật toán Grad-CAM tự xây dựng:
        1. Chạy lan truyền xuôi để lấy dự đoán
        2. Chạy lan truyền ngược cho nhãn lớp được chọn để lấy đạo hàm
        3. Tính toán trọng số tầm quan trọng của các bộ lọc dựa trên trung bình đạo hàm toàn cục (GAP)
        4. Kết hợp tuyến tính các bộ kích hoạt với trọng số
        5. Đi qua hàm kích hoạt ReLU và chuẩn hóa về khoảng [0, 255] làm bản đồ nhiệt
        """
        # Reset gradients
        self.model.zero_grad()
        
        # 1. Lan truyền xuôi (Forward Pass)
        output = self.model(input_tensor)
        
        # Lấy class có xác suất cao nhất nếu không chỉ định rõ
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        # Lấy điểm số của lớp cần giải thích
        score = output[0][class_idx]
        
        # 2. Lan truyền ngược (Backward Pass) để tính đạo hàm riêng của score đối với các lớp convolution
        score.backward()
        
        # Lấy activations và gradients từ Hook
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # 3. Tính toán trọng số GAP (Global Average Pooling) của gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # 4. Nhân tuyến tính các lớp activation với trọng số tương ứng
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # 5. Đi qua hàm kích hoạt ReLU
        cam = torch.relu(cam)
        cam = cam.cpu().detach().numpy()
        
        # Tránh lỗi chia cho 0 nếu tất cả giá trị đều bằng 0
        if np.max(cam) == 0:
            cam = np.ones(cam.shape)
            
        # Chuẩn hóa về [0, 1]
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        # Thay đổi kích thước bản đồ nhiệt về trùng với ảnh gốc (224x224)
        cam_resized = cv2.resize(cam, (224, 224))
        
        # Nhân với 255 để chuyển về định dạng ảnh grayscale 8-bit chuẩn OpenCV
        heatmap = np.uint8(255 * cam_resized)
        
        return heatmap


class BrainTumorCNN_V2(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN_V2, self).__init__()
        
        # Tăng gấp đôi số lượng filters (Capacity) cho mỗi lớp
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 4 - Tăng lên 256 filters
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling (GAP)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout(0.5)
        # Lớp phân loại trực tiếp
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ConvBlock(nn.Module):
    """
    Block Tích chập Kép (Double Convolution Block)
    (Conv2d -> BatchNorm2d -> ReLU) x2 -> MaxPool2d
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class BrainTumorCNN_V3(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN_V3, self).__init__()
        
        # --- FEATURE EXTRACTOR BẰNG DOUBLE CONV BLOCKS ---
        self.features = nn.Sequential(
            ConvBlock(3, 32),    # Đầu ra: (32, 112, 112)
            ConvBlock(32, 64),   # Đầu ra: (64, 56, 56)
            ConvBlock(64, 128),  # Đầu ra: (128, 28, 28)
            ConvBlock(128, 256)  # Đầu ra: (256, 14, 14)
        )
        
        # --- GLOBAL AVERAGE POOLING ---
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # Đầu ra: (256, 1, 1)
        
        # --- CLASSIFICATION HEAD ---
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        
        # Khởi tạo trọng số tự động (Kaiming He)
        self._init_weights()

    def _init_weights(self):
        """Khởi tạo trọng số tối ưu giúp mô hình hội tụ nhanh hơn ngay từ đầu"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

