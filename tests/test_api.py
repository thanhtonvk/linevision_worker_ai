# =============================================================================
# TEST SCRIPT FOR TENNIS ANALYSIS API
# =============================================================================

import requests
import json

# Cấu hình
API_URL = "http://localhost:5000"
VIDEO_PATH = "crop_video/part_000.mp4"  # Thay đổi đường dẫn video của bạn


def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{API_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_analyze_video():
    """Test video analysis endpoint"""
    print("Testing video analysis...")

    # Prepare files and data
    files = {"video": open(VIDEO_PATH, "rb")}

    data = {
        "ball_conf": 0.7,
        "person_conf": 0.6,
        "angle_threshold": 50,
        "intersection_threshold": 100,
        "court_bounds": "100,100,400,500",
    }

    # Send request
    response = requests.post(f"{API_URL}/api/analyze", files=files, data=data)

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()  # Trực tiếp lấy result, không cần .data
        print("\n" + "=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)

        # Print highest speed info
        print("\n1. HIGHEST SPEED INFO:")
        print("-" * 40)
        highest_speed = result["highest_speed_info"]
        print(f"Frame: {highest_speed['frame']}")
        print(f"Time: {highest_speed['time_seconds']} seconds")
        print(f"Velocity: {highest_speed['velocity']} pixels/second")
        print(f"Person ID: {highest_speed['person_id']}")
        print(f"Shoulder Angle: {highest_speed['shoulder_angle']}°")
        print(f"Knee Bend Angle: {highest_speed['knee_bend_angle']}°")
        if highest_speed["cropped_image_url"]:
            print(f"Image URL: {highest_speed['cropped_image_url']}")

        # Print best players
        print("\n2. BEST PLAYERS:")
        print("-" * 40)
        for player in result["best_players"]:
            print(f"\nRank #{player['rank']} - Player {player['player_id']}:")
            print(f"  Score: {player['score']}")
            print(f"  In Court Ratio: {player['in_court_ratio']*100:.2f}%")
            print(f"  Avg Ball Speed: {player['avg_ball_speed']} pixels/second")
            print(f"  Avg Shoulder Angle: {player['avg_shoulder_angle']}°")
            print(f"  Avg Knee Bend Angle: {player['avg_knee_bend_angle']}°")
            print(f"  Total Hits: {player['total_hits']}")
            if player["cropped_image_url"]:
                print(f"  Image URL: {player['cropped_image_url']}")

        # Print match statistics
        print("\n3. MATCH STATISTICS:")
        print("-" * 40)
        stats = result["match_statistics"]
        print(f"Rally Ratio: {stats['rally_ratio']*100:.2f}%")
        print(f"In Court Ratio: {stats['in_court_ratio']*100:.2f}%")
        print(f"Out Court Ratio: {stats['out_court_ratio']*100:.2f}%")
        print(f"Total Hits: {stats['total_hits']}")
        print(f"Total In Court: {stats['total_in_court']}")
        print(f"Total Out Court: {stats['total_out_court']}")

        # Print visualization video URL
        print("\n4. VISUALIZATION VIDEO:")
        print("-" * 40)
        if result["visualization_video_url"]:
            print(f"Video URL: {result['visualization_video_url']}")
            print("\n✅ You can open this URL in your browser to view the video!")

        print("\n" + "=" * 80)

        # Save full response to file
        with open("api_response.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("\n✅ Full response saved to: api_response.json")

    else:
        print(f"Error: {response.text}")


def test_get_results(request_id):
    """Test get results endpoint"""
    print(f"\nTesting get results for request_id: {request_id}...")
    response = requests.get(f"{API_URL}/api/results/{request_id}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


if __name__ == "__main__":
    # Test health check
    test_health_check()

    # Test video analysis
    test_analyze_video()

    # Uncomment to test get results (replace with actual request_id)
    # test_get_results("your_request_id_here")
