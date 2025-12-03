# =============================================================================
# SIMPLE TEST SCRIPT WITH TIMEOUT - Kiểm tra API với timeout cao
# =============================================================================

import requests
import json

# Cấu hình
API_URL = "http://localhost:5000"
VIDEO_PATH = "crop_video/part_000.mp4"  # Thay đổi đường dẫn video của bạn


def test_health_check():
    """Test health check endpoint"""
    print("=" * 80)
    print("TESTING HEALTH CHECK")
    print("=" * 80)
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=5)
        print(f"✅ Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}\n")
        return True
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False


def test_analyze_video():
    """Test video analysis endpoint với timeout cao"""
    print("=" * 80)
    print("TESTING VIDEO ANALYSIS")
    print("=" * 80)
    print("⏳ Đang upload và phân tích video... (có thể mất vài phút)")
    print("=" * 80)

    try:
        # Prepare files and data
        with open(VIDEO_PATH, "rb") as video_file:
            files = {"video": video_file}

            data = {
                "ball_conf": 0.7,
                "person_conf": 0.6,
                "angle_threshold": 50,
                "intersection_threshold": 100,
                "court_bounds": "100,100,400,500",
            }

            # Send request với timeout rất cao (30 phút)
            response = requests.post(
                f"{API_URL}/api/analyze",
                files=files,
                data=data,
                timeout=1800,  # 30 phút timeout
            )

        print(f"\n✅ Status: {response.status_code}\n")

        if response.status_code == 200:
            result = response.json()

            print("=" * 80)
            print("ANALYSIS RESULTS")
            print("=" * 80)

            # Print request info
            print(f"\n📋 Request ID: {result['request_id']}")
            print(f"⏰ Timestamp: {result['timestamp']}")

            # Print highest speed info
            print("\n1️⃣ HIGHEST SPEED INFO:")
            print("-" * 40)
            highest_speed = result["highest_speed_info"]
            print(f"Frame: {highest_speed['frame']}")
            print(f"Time: {highest_speed['time_seconds']} seconds")
            print(f"Velocity: {highest_speed['velocity']} pixels/second")
            print(f"Person ID: {highest_speed['person_id']}")
            print(f"Shoulder Angle: {highest_speed['shoulder_angle']}°")
            print(f"Knee Bend Angle: {highest_speed['knee_bend_angle']}°")
            if highest_speed["cropped_image_url"]:
                print(f"🖼️  Image URL: {highest_speed['cropped_image_url']}")

            # Print best players
            print("\n2️⃣ BEST PLAYERS:")
            print("-" * 40)
            for player in result["best_players"]:
                print(f"\n🏆 Rank #{player['rank']} - Player {player['player_id']}:")
                print(f"   Score: {player['score']}")
                print(f"   In Court Ratio: {player['in_court_ratio']*100:.2f}%")
                print(f"   Avg Ball Speed: {player['avg_ball_speed']} pixels/second")
                print(f"   Avg Shoulder Angle: {player['avg_shoulder_angle']}°")
                print(f"   Avg Knee Bend Angle: {player['avg_knee_bend_angle']}°")
                print(f"   Total Hits: {player['total_hits']}")
                if player["cropped_image_url"]:
                    print(f"   🖼️  Image URL: {player['cropped_image_url']}")

            # Print match statistics
            print("\n3️⃣ MATCH STATISTICS:")
            print("-" * 40)
            stats = result["match_statistics"]
            print(f"Rally Ratio: {stats['rally_ratio']*100:.2f}%")
            print(f"In Court Ratio: {stats['in_court_ratio']*100:.2f}%")
            print(f"Out Court Ratio: {stats['out_court_ratio']*100:.2f}%")
            print(f"Total Hits: {stats['total_hits']}")
            print(f"Total In Court: {stats['total_in_court']}")
            print(f"Total Out Court: {stats['total_out_court']}")

            # Print visualization video URL
            print("\n4️⃣ VISUALIZATION VIDEO:")
            print("-" * 40)
            if result["visualization_video_url"]:
                print(f"🎥 Video URL: {result['visualization_video_url']}")
                print("\n✅ Bạn có thể mở URL này trong trình duyệt để xem video!")

            print("\n" + "=" * 80)

            # Save full response to file
            with open("api_response.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print("💾 Full response saved to: api_response.json")
            print("=" * 80)

            return True

        else:
            print(f"❌ Error Response:")
            print(response.text)
            return False

    except requests.exceptions.Timeout:
        print("❌ Request timeout! Video quá lớn hoặc server xử lý quá lâu.")
        print("💡 Thử tăng timeout hoặc dùng video ngắn hơn.")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        print("💡 Kiểm tra:")
        print("   1. Flask server có đang chạy không?")
        print("   2. Port 5000 có bị chiếm không?")
        print("   3. Có lỗi gì trong terminal Flask server không?")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🎾 TENNIS ANALYSIS API TEST\n")

    # Test 1: Health check
    if not test_health_check():
        print("⚠️  Server không phản hồi. Hãy kiểm tra Flask server có đang chạy không.")
        exit(1)

    # Test 2: Video analysis
    print("\n")
    test_analyze_video()
