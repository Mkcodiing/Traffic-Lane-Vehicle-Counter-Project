<<<<<<< HEAD
# Traffic-Lane-Vehicle-Counter-Project
This project is a Traffic Flow Counter that uses a pre-trained YOLO model with DeepSORT tracking to detect, classify, and count vehicles in multiple lanes from video footage. It outputs an annotated video and a CSV file containing vehicle ID, type, lane, frame, and timestamp for traffic analysis.
=======
# Traffic Flow Analysis - YOLOv8 + DeepSORT

## Files
- traffic_count.py  -> main script
- download_video.py -> download YouTube video to input_video.mp4 (optional)
- requirements.txt
- counts.csv         -> generated after running
- output.mp4         -> generated overlay video

## Setup
1. python -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt

## Run
# Use local video:
python traffic_count.py --video input_video.mp4

# Or download & run (YouTube):
python traffic_count.py --url "https://www.youtube.com/watch?v=VIDEO_ID"

## Outputs
- output.mp4 : overlays of bounding boxes, IDs and lane counters
- counts.csv : [vehicle_id, lane, frame, timestamp]

## Notes
- Edit lane boundaries in traffic_count.py (x1, x2) if needed.
- Use yolov8n for speed, yolov8s/m for higher accuracy.
>>>>>>> d088bd1 (1st commit)
