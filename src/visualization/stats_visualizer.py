# =============================================================================
# STATS VISUALIZER - VISUALIZATION CHO CHỈ SỐ NGƯỜI CHƠI TENNIS
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from typing import List, Tuple, Dict, Optional
import os


class StatsVisualizer:
    """
    Class tạo các visualization cho chỉ số người chơi tennis:
    - Heatmap phạm vi bao quát sân
    - Bảng thống kê
    - Bảng xếp hạng
    - Biểu đồ tốc độ
    """

    def __init__(self, court_points: List[Tuple[int, int]] = None):
        """
        Khởi tạo StatsVisualizer

        Args:
            court_points: 12 điểm tọa độ sân (optional)
        """
        self.court_points = court_points
        self.colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

    def create_heatmap(
        self,
        positions: List[Tuple[float, float]],
        court_image: np.ndarray = None,
        output_path: str = None,
        player_id: int = None
    ) -> np.ndarray:
        """
        Tạo heatmap phạm vi bao quát sân

        Args:
            positions: List các tọa độ (x, y) của người chơi
            court_image: Ảnh nền sân (optional)
            output_path: Đường dẫn lưu file (optional)
            player_id: ID người chơi để hiển thị

        Returns:
            Ảnh heatmap
        """
        if not positions:
            print(f"⚠️ Không có dữ liệu vị trí cho player {player_id}")
            return None

        # Xác định kích thước
        if court_image is not None:
            height, width = court_image.shape[:2]
        else:
            all_x = [p[0] for p in positions]
            all_y = [p[1] for p in positions]
            width = int(max(all_x) + 100)
            height = int(max(all_y) + 100)

        # Tạo heatmap bằng histogram 2D
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]

        # Tạo histogram
        heatmap, xedges, yedges = np.histogram2d(
            x_coords, y_coords,
            bins=[width // 20, height // 20],
            range=[[0, width], [0, height]]
        )

        # Chuẩn hóa và làm mượt
        heatmap = heatmap.T  # Transpose để khớp với ảnh
        heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (15, 15), 0)

        # Normalize về 0-255
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)

        # Áp dụng colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Resize về kích thước gốc
        heatmap_colored = cv2.resize(heatmap_colored, (width, height))

        # Overlay lên ảnh nền nếu có
        if court_image is not None:
            # Blend heatmap với ảnh nền
            result = cv2.addWeighted(court_image, 0.5, heatmap_colored, 0.5, 0)
        else:
            result = heatmap_colored

        # Vẽ đường viền sân nếu có court_points
        if self.court_points:
            pts = np.array(self.court_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(result, [pts], True, (255, 255, 255), 2)

        # Thêm title
        if player_id is not None:
            cv2.putText(
                result,
                f"Player {player_id} Coverage Heatmap",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

        # Lưu file nếu cần
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"✅ Saved heatmap: {output_path}")

        return result

    def create_stats_table(
        self,
        player_stats: dict,
        output_path: str = None
    ) -> np.ndarray:
        """
        Tạo bảng thống kê cho một người chơi

        Args:
            player_stats: Dict chứa thống kê người chơi
            output_path: Đường dẫn lưu file

        Returns:
            Ảnh bảng thống kê
        """
        # Tạo figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')

        player_id = player_stats.get("player_id", "N/A")

        # Chuẩn bị dữ liệu bảng
        table_data = []

        # 1. Độ chính xác
        acc = player_stats.get("accuracy", {})
        table_data.append(["ĐỘ CHÍNH XÁC", ""])
        table_data.append(["  Tổng cú đánh", str(acc.get("total_hits", 0))])
        table_data.append(["  Trong sân", str(acc.get("in_court", 0))])
        table_data.append(["  Ngoài sân", str(acc.get("out_court", 0))])
        table_data.append(["  Không qua lưới", str(acc.get("not_over_net", 0))])
        table_data.append(["  Tỷ lệ chính xác", f"{acc.get('accuracy_pct', 0):.1f}%"])

        # 2. Giao bóng
        serve = player_stats.get("serve", {})
        table_data.append(["", ""])
        table_data.append(["GIAO BÓNG", ""])
        table_data.append(["  Trong sân", str(serve.get("in_court", 0))])
        table_data.append(["  Ngoài sân", str(serve.get("out_court", 0))])
        table_data.append(["  Không qua lưới", str(serve.get("not_over_net", 0))])
        table_data.append(["  Tốc độ TB", f"{serve.get('avg_speed', 0):.1f} px/f"])
        table_data.append(["  Tốc độ max", f"{serve.get('max_speed', 0):.1f} px/f"])

        # 3. Trả bóng
        ret = player_stats.get("return", {})
        table_data.append(["", ""])
        table_data.append(["TRẢ BÓNG", ""])
        table_data.append(["  Trong sân", str(ret.get("in_court", 0))])
        table_data.append(["  Ngoài sân", str(ret.get("out_court", 0))])
        table_data.append(["  Không qua lưới", str(ret.get("not_over_net", 0))])

        # 4. Drive
        drive = player_stats.get("drive", {})
        table_data.append(["", ""])
        table_data.append(["DRIVE", ""])
        table_data.append(["  Tốc độ TB", f"{drive.get('avg_speed', 0):.1f} px/f"])
        table_data.append(["  Tốc độ max", f"{drive.get('max_speed', 0):.1f} px/f"])

        # 5. Mật độ bóng
        density = player_stats.get("shot_density", {})
        table_data.append(["", ""])
        table_data.append(["MẬT ĐỘ BÓNG", ""])
        table_data.append(["  Đường bóng dài", f"{density.get('long_pct', 0):.1f}%"])
        table_data.append(["  Đường bóng vừa", f"{density.get('medium_pct', 0):.1f}%"])
        table_data.append(["  Đường bóng ngắn", f"{density.get('short_pct', 0):.1f}%"])

        # Tạo bảng
        table = ax.table(
            cellText=table_data,
            colLabels=["Chỉ số", "Giá trị"],
            cellLoc='left',
            loc='center',
            colWidths=[0.6, 0.4]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style header
        for key, cell in table.get_celld().items():
            if key[0] == 0:  # Header row
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(color='white')

        plt.title(f"Thống kê người chơi {player_id}", fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Lưu file
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved stats table: {output_path}")

        # Convert to numpy array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return img

    def create_ranking_board(
        self,
        rankings: List[dict],
        player_images: Dict[int, np.ndarray] = None,
        output_path: str = None
    ) -> np.ndarray:
        """
        Tạo bảng xếp hạng người chơi

        Args:
            rankings: List các ranking data
            player_images: Dict {player_id: image} (optional)
            output_path: Đường dẫn lưu file

        Returns:
            Ảnh bảng xếp hạng
        """
        # Tạo figure
        fig, ax = plt.subplots(figsize=(12, len(rankings) * 1.5 + 2))
        ax.axis('off')

        # Chuẩn bị dữ liệu
        headers = ["Hạng", "Player", "Điểm", "Tỷ lệ In/Out", "Tốc độ TB", "Tổng cú đánh"]
        table_data = []

        for r in rankings:
            row = [
                f"#{r['rank']}",
                f"Player {r['player_id']}",
                f"{r['score']:.2f}",
                f"{r['in_out_ratio']*100:.1f}%",
                f"{r['avg_speed']:.1f}",
                str(r['total_hits'])
            ]
            table_data.append(row)

        # Tạo bảng
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colWidths=[0.1, 0.15, 0.15, 0.2, 0.2, 0.2]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)

        # Style
        for key, cell in table.get_celld().items():
            if key[0] == 0:  # Header
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#2196F3')
                cell.set_text_props(color='white')
            elif key[0] == 1:  # First place
                cell.set_facecolor('#FFD700')  # Gold
            elif key[0] == 2:  # Second place
                cell.set_facecolor('#C0C0C0')  # Silver
            elif key[0] == 3:  # Third place
                cell.set_facecolor('#CD7F32')  # Bronze

        plt.title("BẢNG XẾP HẠNG NGƯỜI CHƠI", fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Lưu file
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved ranking board: {output_path}")

        # Convert to numpy array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return img

    def create_speed_chart(
        self,
        serve_speeds: Dict[int, List[float]],
        drive_speeds: Dict[int, List[float]],
        output_path: str = None
    ) -> np.ndarray:
        """
        Tạo biểu đồ tốc độ bóng

        Args:
            serve_speeds: {player_id: [speeds...]}
            drive_speeds: {player_id: [speeds...]}
            output_path: Đường dẫn lưu file

        Returns:
            Ảnh biểu đồ
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Serve speeds
        ax1 = axes[0]
        for i, (player_id, speeds) in enumerate(serve_speeds.items()):
            if speeds:
                ax1.bar(
                    f"P{player_id}",
                    [np.mean(speeds), np.max(speeds)],
                    label=f"Player {player_id}",
                    alpha=0.7
                )
        ax1.set_title("Tốc độ giao bóng (pixels/frame)")
        ax1.set_ylabel("Tốc độ")
        ax1.legend()

        # Drive speeds
        ax2 = axes[1]
        for i, (player_id, speeds) in enumerate(drive_speeds.items()):
            if speeds:
                ax2.bar(
                    f"P{player_id}",
                    [np.mean(speeds), np.max(speeds)],
                    label=f"Player {player_id}",
                    alpha=0.7
                )
        ax2.set_title("Tốc độ drive (pixels/frame)")
        ax2.set_ylabel("Tốc độ")
        ax2.legend()

        plt.tight_layout()

        # Lưu file
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved speed chart: {output_path}")

        # Convert to numpy array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return img

    def create_shot_density_pie(
        self,
        shot_density: dict,
        player_id: int,
        output_path: str = None
    ) -> np.ndarray:
        """
        Tạo biểu đồ tròn mật độ bóng

        Args:
            shot_density: {long_pct, medium_pct, short_pct}
            player_id: ID người chơi
            output_path: Đường dẫn lưu file

        Returns:
            Ảnh biểu đồ
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        labels = ['Dài', 'Vừa', 'Ngắn']
        sizes = [
            shot_density.get('long_pct', 0),
            shot_density.get('medium_pct', 0),
            shot_density.get('short_pct', 0)
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        explode = (0.05, 0.05, 0.05)

        ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            shadow=True,
            startangle=90
        )
        ax.axis('equal')
        plt.title(f"Mật độ bóng - Player {player_id}", fontsize=14, fontweight='bold')

        # Lưu file
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved shot density chart: {output_path}")

        # Convert to numpy array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return img

    def create_full_report(
        self,
        all_stats: dict,
        rankings: List[dict],
        court_image: np.ndarray = None,
        output_dir: str = "outputs"
    ):
        """
        Tạo báo cáo đầy đủ cho tất cả người chơi

        Args:
            all_stats: {player_id: stats_dict}
            rankings: List ranking data
            court_image: Ảnh nền sân
            output_dir: Thư mục lưu output
        """
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("📊 TẠO BÁO CÁO PHÂN TÍCH NGƯỜI CHƠI")
        print("=" * 60)

        # 1. Tạo bảng xếp hạng
        self.create_ranking_board(
            rankings,
            output_path=os.path.join(output_dir, "ranking_board.png")
        )

        # 2. Tạo thống kê và heatmap cho từng người
        for player_id, stats in all_stats.items():
            print(f"\n🎾 Đang tạo báo cáo cho Player {player_id}...")

            # Stats table
            self.create_stats_table(
                stats,
                output_path=os.path.join(output_dir, f"player_{player_id}_stats.png")
            )

            # Heatmap
            positions = stats.get("heatmap_positions", [])
            if positions:
                self.create_heatmap(
                    positions,
                    court_image=court_image,
                    output_path=os.path.join(output_dir, f"player_{player_id}_heatmap.png"),
                    player_id=player_id
                )

            # Shot density pie
            density = stats.get("shot_density", {})
            if density:
                self.create_shot_density_pie(
                    density,
                    player_id,
                    output_path=os.path.join(output_dir, f"player_{player_id}_density.png")
                )

        # 3. Speed chart (all players)
        serve_speeds = {}
        drive_speeds = {}
        for player_id, stats in all_stats.items():
            serve = stats.get("serve", {})
            drive = stats.get("drive", {})
            serve_speeds[player_id] = serve.get("speeds", [])
            drive_speeds[player_id] = drive.get("speeds", [])

        self.create_speed_chart(
            serve_speeds,
            drive_speeds,
            output_path=os.path.join(output_dir, "speed_comparison.png")
        )

        print("\n✅ HOÀN THÀNH TẠO BÁO CÁO!")
        print(f"📁 Các file đã lưu tại: {output_dir}")

    def crop_player_image(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: int = 20
    ) -> np.ndarray:
        """
        Cắt ảnh người chơi từ frame

        Args:
            frame: Frame video
            bbox: (x1, y1, x2, y2)
            padding: Padding xung quanh

        Returns:
            Ảnh người chơi đã crop
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        return frame[y1:y2, x1:x2].copy()
