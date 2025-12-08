# =============================================================================
# STANDALONE TEST SCRIPT FOR TENNIS ANALYSIS
# =============================================================================

from src.core.tennis_analysis_module import TennisAnalysisModule
from config.settings import settings
import cv2
import os
import uuid
from datetime import datetime, timedelta
import traceback
import shutil
import time


# Initialize Tennis Analysis Module
print("Initializing Tennis Analysis Module...")
analyzer = TennisAnalysisModule(
    ball_model_path=settings.ball_model_path,
    person_model_path=settings.person_model_path,
    pose_model_path=settings.pose_model_path,
)


def save_cropped_image(image, output_folder, prefix, identifier):
    """Lưu ảnh crop và trả về đường dẫn"""
    if image is None:
        return None

    filename = f"{prefix}_{identifier}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(output_folder, filename)
    cv2.imwrite(filepath, image)
    return filename


def test_analyze_video():
    """
    Test video analysis với video cụ thể
    """
    try:
        # Video path - thay đổi path này nếu cần
        video_path = "1765199807.mp4"

        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found: {video_path}")
            print(f"Please make sure the video file exists in the current directory")
            return

        print(f"\n{'='*80}")
        print(f"Testing video analysis: {video_path}")
        print(f"{'='*80}\n")

        # Tạo thư mục output riêng cho test này
        request_id = uuid.uuid4().hex
        request_output_folder = os.path.join(settings.output_folder, request_id)
        os.makedirs(request_output_folder, exist_ok=True)

        # Parameters - hardcoded cho test
        ball_conf = 0.7
        person_conf = 0.6
        angle_threshold = 50
        intersection_threshold = 100
        court_bounds = (100, 100, 400, 500)

        print(f"Parameters:")
        print(f"  - ball_conf: {ball_conf}")
        print(f"  - person_conf: {person_conf}")
        print(f"  - angle_threshold: {angle_threshold}")
        print(f"  - intersection_threshold: {intersection_threshold}")
        print(f"  - court_bounds: {court_bounds}")
        print(f"  - output_folder: {request_output_folder}\n")

        # Phân tích video với timing
        print("Starting video analysis...")
        start_time = time.time()

        results = analyzer.analyze_video(
            video_path=video_path,
            ball_conf=ball_conf,
            person_conf=person_conf,
            angle_threshold=angle_threshold,
            intersection_threshold=intersection_threshold,
            court_bounds=court_bounds,
        )

        end_time = time.time()
        total_time = end_time - start_time

        print(
            f"\n✅ Analysis completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)!"
        )

        # Xử lý kết quả
        result = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "expires_at": (
                datetime.now() + timedelta(hours=settings.cleanup_hours)
            ).isoformat(),
            "highest_speed_info": {},
            "best_players": [],
            "match_statistics": {},
            "visualization_video_path": None,
        }

        # 1. Xử lý highest speed info
        print("\n📊 Processing highest speed info...")
        highest_speed = results["highest_speed_info"]
        cropped_filename = save_cropped_image(
            highest_speed["cropped_image"],
            request_output_folder,
            "highest_speed",
            "player",
        )

        result["highest_speed_info"] = {
            "frame": highest_speed["frame"],
            "time_seconds": round(highest_speed["time_seconds"], 2),
            "velocity": round(highest_speed["velocity"], 2),
            "person_id": highest_speed["person_id"],
            "shoulder_angle": round(highest_speed["shoulder_angle"], 2),
            "knee_bend_angle": round(highest_speed["knee_bend_angle"], 2),
            "cropped_image_file": cropped_filename if cropped_filename else None,
        }

        # 2. Xử lý best players
        print("👥 Processing best players...")
        for rank, player in enumerate(results["best_players"], 1):
            cropped_filename = save_cropped_image(
                player["cropped_image"],
                request_output_folder,
                f'player_{player["player_id"]}_rank_{rank}',
                "crop",
            )

            player_data = {
                "rank": rank,
                "player_id": player["player_id"],
                "score": round(player["score"], 2),
                "in_court_ratio": round(player["in_court_ratio"], 4),
                "avg_ball_speed": round(player["avg_ball_speed"], 2),
                "avg_shoulder_angle": round(player["avg_shoulder_angle"], 2),
                "avg_knee_bend_angle": round(player["avg_knee_bend_angle"], 2),
                "total_hits": player["total_hits"],
                "cropped_image_file": cropped_filename if cropped_filename else None,
            }
            result["best_players"].append(player_data)

        # 3. Xử lý match statistics
        print("📈 Processing match statistics...")
        stats = results["match_statistics"]
        result["match_statistics"] = {
            "rally_ratio": round(stats["rally_ratio"], 4),
            "in_court_ratio": round(stats["in_court_ratio"], 4),
            "out_court_ratio": round(stats["out_court_ratio"], 4),
            "total_hits": stats["total_hits"],
            "total_in_court": stats["total_in_court"],
            "total_out_court": stats["total_out_court"],
        }

        # 4. Xử lý visualization video
        print("🎥 Processing visualization video...")
        if results["visualization_video_path"] and os.path.exists(
            results["visualization_video_path"]
        ):
            # Copy video vào output folder
            video_filename = f"visualization_{request_id}.mp4"
            new_video_path = os.path.join(request_output_folder, video_filename)
            shutil.copy2(results["visualization_video_path"], new_video_path)
            result["visualization_video_path"] = new_video_path

        # Print results
        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print(f"{'='*80}\n")

        print(f"📁 Output folder: {request_output_folder}")
        print(f"\n🏆 Highest Speed Info:")
        print(f"  - Frame: {result['highest_speed_info']['frame']}")
        print(f"  - Velocity: {result['highest_speed_info']['velocity']} pixels/second")
        print(f"  - Person ID: {result['highest_speed_info']['person_id']}")
        print(f"  - Image saved: {result['highest_speed_info']['cropped_image_file']}")

        print(f"\n👥 Best Players ({len(result['best_players'])} players):")
        for player in result["best_players"]:
            print(f"  Rank #{player['rank']}: Player {player['player_id']}")
            print(f"    - Score: {player['score']}")
            print(f"    - In-court ratio: {player['in_court_ratio']:.2%}")
            print(f"    - Avg ball speed: {player['avg_ball_speed']:.2f}")
            print(f"    - Total hits: {player['total_hits']}")

        print(f"\n📊 Match Statistics:")
        print(f"  - Rally ratio: {result['match_statistics']['rally_ratio']:.2%}")
        print(f"  - In-court ratio: {result['match_statistics']['in_court_ratio']:.2%}")
        print(
            f"  - Out-court ratio: {result['match_statistics']['out_court_ratio']:.2%}"
        )
        print(f"  - Total hits: {result['match_statistics']['total_hits']}")

        if result["visualization_video_path"]:
            print(f"\n🎥 Visualization video: {result['visualization_video_path']}")

        # Print timing summary
        print(f"\n{'='*80}")
        print(
            f"⏱️  TOTAL  EXECUTION TIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
        )
        print(f"{'='*80}")
        print(f"\n✅ Test completed successfully!")

        return result

    except Exception as e:
        print(f"\n❌ Error during analysis:")
        print(f"{str(e)}")
        print(f"\nTraceback:")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_analyze_video()
