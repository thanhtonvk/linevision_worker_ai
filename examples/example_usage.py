# =============================================================================
# EXAMPLE USAGE - VÍ DỤ SỬ DỤNG TENNIS ANALYSIS MODULE
# =============================================================================

from tennis_analysis_module import TennisAnalysisModule
import cv2
import numpy as np

def main():
    """
    Ví dụ sử dụng module phân tích tennis
    """
    # Khởi tạo module
    analyzer = TennisAnalysisModule(
        ball_model_path="ball_best.pt",
        person_model_path="yolov8m.pt",
        pose_model_path="yolov8n-pose.pt"
    )
    
    # Phân tích video
    video_path = "crop_video/part_000.mp4"  # Thay đổi đường dẫn video của bạn
    
    print("Bắt đầu phân tích video...")
    results = analyzer.analyze_video(
        video_path=video_path,
        ball_conf=0.7,
        person_conf=0.6,
        angle_threshold=50,
        intersection_threshold=100,
        court_bounds=(100, 100, 400, 500)  # Điều chỉnh theo video của bạn
    )
    
    # In kết quả
    print("\n" + "="*80)
    print("KẾT QUẢ PHÂN TÍCH")
    print("="*80)
    
    # 1. Thông tin cú đánh tốc độ cao nhất
    print("\n1. CÚ ĐÁNH TỐC ĐỘ CAO NHẤT:")
    print("-" * 40)
    highest_speed = results['highest_speed_info']
    print(f"Frame: {highest_speed['frame']}")
    print(f"Thời gian: {highest_speed['time_seconds']:.2f} giây")
    print(f"Tốc độ: {highest_speed['velocity']:.2f} pixels/second")
    print(f"Người chơi: {highest_speed['person_id']}")
    print(f"Góc mở vai trung bình: {highest_speed['shoulder_angle']:.2f}°")
    print(f"Góc khụy gối trung bình: {highest_speed['knee_bend_angle']:.2f}°")
    
    if highest_speed['cropped_image'] is not None:
        cv2.imwrite("highest_speed_player_crop.jpg", highest_speed['cropped_image'])
        print("✅ Đã lưu ảnh crop: highest_speed_player_crop.jpg")
    
    # 2. Danh sách người chơi hay nhất
    print("\n2. DANH SÁCH NGƯỜI CHƠI HAY NHẤT:")
    print("-" * 40)
    best_players = results['best_players']
    for rank, player in enumerate(best_players, 1):
        print(f"\nRank #{rank} - Người chơi {player['player_id']}:")
        print(f"  Điểm số: {player['score']:.2f}")
        print(f"  Tỉ lệ bóng trong sân: {player['in_court_ratio']:.2%}")
        print(f"  Tốc độ bóng trung bình: {player['avg_ball_speed']:.2f} pixels/second")
        print(f"  Góc mở vai trung bình: {player['avg_shoulder_angle']:.2f}°")
        print(f"  Góc khụy gối trung bình: {player['avg_knee_bend_angle']:.2f}°")
        print(f"  Tổng số cú đánh: {player['total_hits']}")
        
        if player['cropped_image'] is not None:
            filename = f"player_{player['player_id']}_rank_{rank}_crop.jpg"
            cv2.imwrite(filename, player['cropped_image'])
            print(f"  ✅ Đã lưu ảnh crop: {filename}")
    
    # 3. Thống kê trận đấu
    print("\n3. THỐNG KÊ TRẬN ĐẤU:")
    print("-" * 40)
    stats = results['match_statistics']
    print(f"Tỉ lệ đối kháng (Rally Ratio): {stats['rally_ratio']:.2%}")
    print(f"Tỉ lệ bóng trong sân: {stats['in_court_ratio']:.2%}")
    print(f"Tỉ lệ bóng ngoài sân: {stats['out_court_ratio']:.2%}")
    print(f"Tổng số cú đánh: {stats['total_hits']}")
    print(f"Cú đánh trong sân: {stats['total_in_court']}")
    print(f"Cú đánh ngoài sân: {stats['total_out_court']}")
    
    # 4. Video visualization
    print("\n4. VIDEO VISUALIZATION:")
    print("-" * 40)
    print(f"Đường dẫn: {results['visualization_video_path']}")
    print("✅ Video đã được tạo với đầy đủ annotations")
    
    print("\n" + "="*80)
    print("HOÀN THÀNH!")
    print("="*80)

if __name__ == "__main__":
    main()

