# =============================================================================
# VISUALIZATION CLASS - TẠO BIỂU ĐỒ VÀ BÁO CÁO
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TennisVisualizer:
    """
    Class để tạo các visualization và báo cáo cho tennis analysis
    """
    
    def __init__(self):
        pass
    
    def create_pose_visualization(self, frames, person_detections, pose_detections, output_path="tennis_pose_analysis.mp4"):
        """Tạo video visualization cho pose estimation và person tracking"""
        print("Đang tạo video visualization...")
        
        if not frames:
            print("Không có frames để xử lý!")
            return
        
        # Tạo video writer
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        # COCO keypoint connections
        skeleton = [
            [0, 1], [0, 2], [1, 3], [2, 4],  # Head
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
            [5, 11], [6, 12], [11, 12],  # Torso
            [11, 13], [12, 14], [13, 15], [14, 16]  # Legs
        ]
        
        # Colors for different persons
        person_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        for frame_idx, frame in enumerate(frames):
            vis_frame = frame.copy()
            
            # Draw person detections
            if frame_idx < len(person_detections):
                for person_data in person_detections[frame_idx]:
                    person_id = person_data['person_id']
                    bbox = person_data['person']['bbox']
                    pose = person_data['pose']
                    
                    # Color for this person
                    color = person_colors[person_id % len(person_colors)]
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis_frame, f"Person {person_id}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw pose keypoints
                    if pose is not None:
                        keypoints = pose['keypoints']
                        conf = pose['conf']
                        
                        # Draw keypoints
                        for i, (x, y) in enumerate(keypoints):
                            if conf[i] > 0.5:  # Only draw confident keypoints
                                cv2.circle(vis_frame, (int(x), int(y)), 3, color, -1)
                        
                        # Draw skeleton
                        for connection in skeleton:
                            pt1_idx, pt2_idx = connection
                            if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and 
                                conf[pt1_idx] > 0.5 and conf[pt2_idx] > 0.5):
                                
                                pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                                pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                                cv2.line(vis_frame, pt1, pt2, color, 2)
            
            # Add frame info
            cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(vis_frame)
            
            if frame_idx % 100 == 0:
                print(f"Đã xử lý {frame_idx}/{len(frames)} frames...")
        
        out.release()
        print(f"✅ Đã tạo video: {output_path}")

    def create_technique_analysis_plot(self, technique_analysis, output_path="tennis_technique_analysis.png"):
        """Tạo biểu đồ phân tích kỹ thuật tennis"""
        person_stats = technique_analysis['person_stats']
        
        if not person_stats:
            print("Không có dữ liệu người chơi để phân tích!")
            return
        
        # Tạo subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('TENNIS TECHNIQUE ANALYSIS', fontsize=16, fontweight='bold')
        
        # 1. Hit accuracy by person
        ax1 = axes[0, 0]
        person_ids = list(person_stats.keys())
        hits_in_court = [person_stats[pid]['hits_in_court'] for pid in person_ids]
        hits_out_court = [person_stats[pid]['hits_out_court'] for pid in person_ids]
        
        x = np.arange(len(person_ids))
        width = 0.35
        
        ax1.bar(x - width/2, hits_in_court, width, label='In Court', color='green', alpha=0.7)
        ax1.bar(x + width/2, hits_out_court, width, label='Out Court', color='red', alpha=0.7)
        
        ax1.set_xlabel('Person ID')
        ax1.set_ylabel('Number of Hits')
        ax1.set_title('Hit Accuracy by Person')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Person {pid}' for pid in person_ids])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy percentage
        ax2 = axes[0, 1]
        accuracy_percentages = []
        for pid in person_ids:
            total_hits = person_stats[pid]['hits_in_court'] + person_stats[pid]['hits_out_court']
            if total_hits > 0:
                accuracy = person_stats[pid]['hits_in_court'] / total_hits * 100
            else:
                accuracy = 0
            accuracy_percentages.append(accuracy)
        
        bars = ax2.bar([f'Person {pid}' for pid in person_ids], accuracy_percentages, 
                       color=['green' if acc > 70 else 'orange' if acc > 50 else 'red' for acc in accuracy_percentages])
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Hit Accuracy Percentage')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracy_percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # 3. Technique errors
        ax3 = axes[1, 0]
        error_counts = {}
        for pid, stats in person_stats.items():
            for error in stats['technique_errors']:
                error_type = error['type']
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        if error_counts:
            error_types = list(error_counts.keys())
            error_values = list(error_counts.values())
            
            bars = ax3.bar(error_types, error_values, color='red', alpha=0.7)
            ax3.set_ylabel('Number of Errors')
            ax3.set_title('Technique Errors by Type')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, error_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        str(value), ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, 'No technique errors detected', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Technique Errors by Type')
        
        # 4. Total hits by person
        ax4 = axes[1, 1]
        total_hits = [person_stats[pid]['hits_in_court'] + person_stats[pid]['hits_out_court'] 
                      for pid in person_ids]
        
        bars = ax4.bar([f'Person {pid}' for pid in person_ids], total_hits, 
                       color='blue', alpha=0.7)
        ax4.set_ylabel('Total Hits')
        ax4.set_title('Total Hits by Person')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, total_hits):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Đã tạo biểu đồ: {output_path}")

    def create_court_accuracy_visualization(self, technique_analysis, output_path="tennis_court_accuracy.png"):
        """Tạo biểu đồ visualization cho độ chính xác cú đánh"""
        person_stats = technique_analysis['person_stats']
        court_accuracy = technique_analysis['court_accuracy']
        
        # Lọc chỉ những người có cú đánh
        active_persons = {pid: stats for pid, stats in person_stats.items() if stats['total_hits'] > 0}
        
        if not active_persons:
            print("Không có người chơi nào có cú đánh để tạo biểu đồ!")
            return
        
        # Tạo subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('TENNIS COURT ACCURACY ANALYSIS', fontsize=16, fontweight='bold')
        
        # 1. Biểu đồ cột so sánh trong/ngoài sân
        ax1 = axes[0, 0]
        person_ids = list(active_persons.keys())
        hits_in_court = [active_persons[pid]['hits_in_court'] for pid in person_ids]
        hits_out_court = [active_persons[pid]['hits_out_court'] for pid in person_ids]
        
        x = np.arange(len(person_ids))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, hits_in_court, width, label='Trong sân', color='green', alpha=0.7)
        bars2 = ax1.bar(x + width/2, hits_out_court, width, label='Ngoài sân', color='red', alpha=0.7)
        
        ax1.set_xlabel('Người chơi')
        ax1.set_ylabel('Số cú đánh')
        ax1.set_title('So sánh cú đánh trong/ngoài sân')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Người {pid}' for pid in person_ids])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Thêm giá trị trên cột
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 2. Biểu đồ tỷ lệ chính xác
        ax2 = axes[0, 1]
        accuracy_percentages = [active_persons[pid]['accuracy_percentage'] for pid in person_ids]
        
        bars = ax2.bar([f'Người {pid}' for pid in person_ids], accuracy_percentages, 
                       color=['green' if acc > 70 else 'orange' if acc > 50 else 'red' for acc in accuracy_percentages])
        ax2.set_ylabel('Tỷ lệ chính xác (%)')
        ax2.set_title('Tỷ lệ chính xác từng người chơi')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # Thêm giá trị trên cột
        for bar, acc in zip(bars, accuracy_percentages):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # 3. Biểu đồ tròn tổng hợp
        ax3 = axes[1, 0]
        labels = ['Trong sân', 'Ngoài sân']
        sizes = [court_accuracy['total_in_court'], court_accuracy['total_out_court']]
        colors = ['green', 'red']
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Phân bố tổng cú đánh')
        
        # 4. Biểu đồ so sánh tổng cú đánh
        ax4 = axes[1, 1]
        total_hits = [active_persons[pid]['total_hits'] for pid in person_ids]
        
        bars = ax4.bar([f'Người {pid}' for pid in person_ids], total_hits, 
                       color='blue', alpha=0.7)
        ax4.set_ylabel('Tổng cú đánh')
        ax4.set_title('Tổng cú đánh từng người chơi')
        ax4.grid(True, alpha=0.3)
        
        # Thêm giá trị trên cột
        for bar, value in zip(bars, total_hits):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(value)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Đã tạo biểu đồ độ chính xác: {output_path}")

    def create_detailed_technique_report(self, person_tracker, technique_analysis, output_path="tennis_detailed_report.txt"):
        """Tạo báo cáo chi tiết về kỹ thuật tennis"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("           BÁO CÁO CHI TIẾT KỸ THUẬT TENNIS\n")
            f.write("=" * 80 + "\n\n")
            
            person_stats = technique_analysis['person_stats']
            court_accuracy = technique_analysis['court_accuracy']
            
            # Thống kê tổng hợp
            f.write("THỐNG KÊ TỔNG HỢP:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Tổng cú đánh: {court_accuracy['total_hits']}\n")
            f.write(f"Cú đánh trong sân: {court_accuracy['total_in_court']}\n")
            f.write(f"Cú đánh ngoài sân: {court_accuracy['total_out_court']}\n")
            f.write(f"Tỷ lệ chính xác tổng: {court_accuracy['overall_accuracy']:.1f}%\n\n")
            
            # Lọc chỉ những người có cú đánh
            active_persons = {pid: stats for pid, stats in person_stats.items() if stats['total_hits'] > 0}
            
            f.write(f"Tổng số người được track: {court_accuracy['total_persons_count']}\n")
            f.write(f"Số người có cú đánh: {court_accuracy['active_persons_count']}\n\n")
            
            if not active_persons:
                f.write("❌ Không có người chơi nào có cú đánh!\n\n")
            else:
                for person_id, stats in active_persons.items():
                    f.write(f"NGƯỜI CHƠI {person_id}:\n")
                    f.write("-" * 40 + "\n")
                    
                    # Thống kê cơ bản
                    f.write(f"Tổng cú đánh: {stats['total_hits']}\n")
                    f.write(f"Cú đánh trong sân: {stats['hits_in_court']}\n")
                    f.write(f"Cú đánh ngoài sân: {stats['hits_out_court']}\n")
                    f.write(f"Tỷ lệ chính xác: {stats['accuracy_percentage']:.1f}%\n")
                    
                    # Phân tích lỗi kỹ thuật
                    f.write(f"\nLỗi kỹ thuật phát hiện: {len(stats['technique_errors'])}\n")
                    
                    if stats['technique_errors']:
                        error_types = {}
                        for error in stats['technique_errors']:
                            error_type = error['type']
                            error_types[error_type] = error_types.get(error_type, 0) + 1
                        
                        for error_type, count in error_types.items():
                            f.write(f"  - {error_type}: {count} lần\n")
                            
                            # Mô tả lỗi
                            if error_type == 'insufficient_knee_bend':
                                f.write("    → Khụy gối không đủ sâu, ảnh hưởng đến lực đánh bóng\n")
                            elif error_type == 'stepping_on_line':
                                f.write("    → Dẫm vạch khi đánh bóng, vi phạm luật chơi\n")
                            elif error_type == 'poor_follow_through':
                                f.write("    → Tư thế sau khi đánh bóng không tốt\n")
                    else:
                        f.write("  - Không có lỗi kỹ thuật phát hiện\n")
                    
                    # Chi tiết từng cú đánh
                    if stats['hit_details']:
                        f.write(f"\nChi tiết {len(stats['hit_details'])} cú đánh:\n")
                        for i, hit_detail in enumerate(stats['hit_details'], 1):
                            status = "TRONG SÂN" if hit_detail['is_in_court'] else "NGOÀI SÂN"
                            f.write(f"  Cú đánh {i}:\n")
                            f.write(f"    - Frame: {hit_detail['frame']}\n")
                            f.write(f"    - Vị trí bóng: {hit_detail['ball_pos']}\n")
                            f.write(f"    - Trạng thái: {status}\n")
                            
                            if hit_detail['pose_analysis']:
                                pose_analysis = hit_detail['pose_analysis']
                                if 'shoulder_angle' in pose_analysis:
                                    f.write(f"    - Góc vai: {pose_analysis['shoulder_angle']:.1f}°\n")
                                if 'knee_bend' in pose_analysis:
                                    f.write(f"    - Góc khụy gối: {pose_analysis['knee_bend']:.1f}°\n")
                                if 'racket_position' in pose_analysis:
                                    f.write(f"    - Khoảng cách vợt-bóng: {pose_analysis['racket_position']:.1f} pixels\n")
                            
                            if hit_detail['technique_errors']:
                                f.write(f"    - Lỗi kỹ thuật: {len(hit_detail['technique_errors'])} lỗi\n")
                                for error in hit_detail['technique_errors']:
                                    f.write(f"      + {error['type']}: {error['description']}\n")
                    
                    f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"✅ Đã tạo báo cáo chi tiết: {output_path}")
