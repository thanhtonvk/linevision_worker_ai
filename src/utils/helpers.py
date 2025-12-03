"""
Utils functions for tennis analysis
Các hàm tiện ích cho phân tích tennis
"""

import cv2
import numpy as np

def distored_image(img):
    """
    Xử lý distortion cho hình ảnh (placeholder function)
    """
    return img

def get_alignment_matrix(src_points):
    """
    Tạo ma trận alignment từ các điểm góc
    """
    # Kiểm tra số lượng điểm
    if len(src_points) != 4:
        print(f"Warning: Expected 4 corner points, got {len(src_points)}. Using first 4 points.")
        src_points = src_points[:4]
    
    # Điểm đích cho sân tennis chuẩn (4 góc)
    dst_points = np.array([
        [0, 0],      # Top-left
        [300, 0],    # Top-right
        [300, 600],  # Bottom-right
        [0, 600]     # Bottom-left
    ], dtype=np.float32)
    
    # Tạo ma trận perspective transformation
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return matrix

def check_in_out(point):
    """
    Kiểm tra điểm có trong sân tennis không
    """
    x, y = point
    
    # Sân tennis chuẩn: 300x600
    if 0 <= x <= 300 and 0 <= y <= 600:
        return "IN"
    else:
        return "OUT"

def create_visual_background():
    """
    Tạo background cho visualization
    """
    # Background 500x800, sân 300x600 ở giữa
    top_left_x = (500 - 300) // 2  # 100
    top_left_y = (800 - 600) // 2  # 100
    return top_left_x, top_left_y

def draw_ball_on_visual(x, y, top_left_x, top_left_y):
    """
    Chuyển đổi tọa độ bóng sang tọa độ visualization
    """
    new_x = top_left_x + x
    new_y = top_left_y + y
    return new_x, new_y

def create_tennis_court_visualization(width=500, height=800, court_width=300, court_height=600):
    """
    Tạo visualization sân tennis
    """
    # Tạo background
    court_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Tính vị trí sân ở giữa
    court_x_start = (width - court_width) // 2
    court_y_start = (height - court_height) // 2
    court_x_end = court_x_start + court_width
    court_y_end = court_y_start + court_height
    
    # Vẽ sân tennis
    cv2.rectangle(court_img, (court_x_start, court_y_start), (court_x_end, court_y_end), (0, 255, 0), 2)
    
    # Vẽ lưới
    net_x = court_x_start + court_width // 2
    cv2.line(court_img, (net_x, court_y_start), (net_x, court_y_end), (0, 0, 0), 3)
    
    # Vẽ đường ngang giữa
    mid_y = court_y_start + court_height // 2
    cv2.line(court_img, (court_x_start, mid_y), (court_x_end, mid_y), (0, 0, 0), 1)
    
    # Vẽ grid 4x4
    cell_w = court_width // 4
    cell_h = court_height // 4
    
    for i in range(1, 4):
        x = court_x_start + i * cell_w
        cv2.line(court_img, (x, court_y_start), (x, court_y_end), (0, 0, 0), 1)
    
    for i in range(1, 4):
        y = court_y_start + i * cell_h
        cv2.line(court_img, (court_x_start, y), (court_x_end, y), (0, 0, 0), 1)
    
    return court_img

def save_video_with_annotations(frames, positions, direction_flags, output_path, fps=30):
    """
    Lưu video với annotations
    """
    if not frames:
        return
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for i, frame in enumerate(frames):
        frame_copy = frame.copy()
        
        # Vẽ vị trí bóng
        if i < len(positions) and positions[i] != (-1, -1):
            x, y = int(positions[i][0]), int(positions[i][1])
            
            # Chọn màu theo loại thay đổi hướng
            if i < len(direction_flags):
                if direction_flags[i] == 1:  # Bóng chạm đất
                    color = (0, 0, 255)  # Đỏ
                elif direction_flags[i] == 2:  # Bóng được đánh bởi người
                    color = (0, 255, 0)  # Xanh lá
                else:
                    color = (255, 0, 0)  # Xanh dương
            else:
                color = (255, 0, 0)
            
            cv2.circle(frame_copy, (x, y), 6, color, -1)
            cv2.circle(frame_copy, (x, y), 4, (255, 255, 255), 2)
        
        out.write(frame_copy)
    
    out.release()
    print(f"Đã lưu video với annotations: {output_path}")

def calculate_distance(p1, p2):
    """
    Tính khoảng cách giữa 2 điểm
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_angle(p1, p2, p3):
    """
    Tính góc giữa 3 điểm
    """
    # Vector từ p1 đến p2
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    # Vector từ p2 đến p3
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Tính cos của góc
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Chuyển đổi sang độ
    angle = np.arccos(cos_angle) * 180 / np.pi
    return angle