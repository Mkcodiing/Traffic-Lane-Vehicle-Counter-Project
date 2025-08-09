import logging
# Disable Ultralytics logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

import argparse
import time
import math
import cv2
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def download_youtube(url, out_path="input_video.mp4"):
    """Download video from YouTube using pytube"""
    try:
        from pytube import YouTube
    except Exception as e:
        raise RuntimeError("pytube not installed or failed: " + str(e))
    print(f"[INFO] Downloading {url} -> {out_path}")
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if stream is None:
        raise RuntimeError("No mp4 stream available; download manually.")
    stream.download(filename=out_path)
    return out_path


def get_lane_from_x(x, x1, x2):
    """Determine lane number based on x position"""
    if x < x1:
        return 1
    elif x < x2:
        return 2
    else:
        return 3


def main(args):
    # Choose input source
    if args.url and not args.video:
        video_path = download_youtube(args.url, out_path=args.input_name)
    elif args.video:
        video_path = args.video
    else:
        raise ValueError("Provide --video or --url")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[INFO] Opened {video_path} {width}x{height} @ {fps}fps, frames={total_frames}")

    # Process only 10 seconds for testing
    max_frames = int(fps * 10)
    print(f"[INFO] Will process {max_frames} frames (~10 seconds of video playback)")

    # Lane boundaries
    x1 = width // 3
    x2 = 2 * width // 3
    print(f"[INFO] Lane x boundaries: {x1}, {x2}")

    # Load YOLO model (COCO-pretrained)
    print("[INFO] Loading YOLO (COCO-pretrained) model for vehicle detection...")
    model = YOLO(args.model)
    tracker = DeepSort(max_age=30)

    # Vehicle classes from COCO dataset
    vehicle_classes = set(["car", "truck", "bus", "motorcycle", "bicycle", "train"])

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    lane_counts = [0, 0, 0]
    seen_pairs = set()
    records = []

    frame_idx = 0
    t0 = time.time()
    start_frame_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Stop after 20 seconds worth of frames
        if frame_idx > max_frames:
            break

        infer_size = args.infer_size
        scaled = cv2.resize(frame, (infer_size, math.ceil(infer_size * height / width)))
        scale_x = width / scaled.shape[1]
        scale_y = height / scaled.shape[0]

        # Vehicle detection with YOLO (COCO model)
        results = model(scaled, verbose=False)[0]
        boxes_xyxy = []
        scores = []
        class_ids = []
        class_names = []

        if len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy().flatten()
            cls = results.boxes.cls.cpu().numpy().astype(int).flatten()
            for i in range(len(confs)):
                name = model.names.get(cls[i], str(cls[i])).lower()
                if name in vehicle_classes:  # Only keep vehicle detections
                    x1b = int(xyxy[i, 0] * scale_x)
                    y1b = int(xyxy[i, 1] * scale_y)
                    x2b = int(xyxy[i, 2] * scale_x)
                    y2b = int(xyxy[i, 3] * scale_y)
                    boxes_xyxy.append((x1b, y1b, x2b, y2b))
                    scores.append(float(confs[i]))
                    class_ids.append(int(cls[i]))
                    class_names.append(name)

        # Prepare detections for tracker
        dets = []
        for (x1b, y1b, x2b, y2b), score, name in zip(boxes_xyxy, scores, class_names):
            w = x2b - x1b
            h = y2b - y1b
            if w > 4 and h > 4:
                dets.append(([x1b, y1b, w, h], score, name))

        tracks = tracker.update_tracks(dets, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = int(track.track_id)
            l, t_, r, b = [int(x) for x in track.to_ltrb()]
            cx = int((l + r) / 2)
            cy = int((t_ + b) / 2)

            # Get vehicle type from tracker
            vehicle_name = getattr(track, "det_class", None) or "vehicle"
            if isinstance(vehicle_name, list) and len(vehicle_name) > 0:
                vehicle_name = vehicle_name[0]
            vehicle_name = str(vehicle_name)

            lane_no = get_lane_from_x(cx, x1, x2)
            if (track_id, lane_no) not in seen_pairs:
                seen_pairs.add((track_id, lane_no))
                lane_counts[lane_no - 1] += 1
                timestamp = frame_idx / fps
                records.append({
                    "vehicle_id": track_id,
                    "vehicle_type": vehicle_name,
                    "lane of road": lane_no,
                    "frame ": frame_idx,
                    "    timestamp": round(timestamp, 3)
                })

            # Draw detection
            cv2.rectangle(frame, (l, t_), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"{vehicle_name} ID {track_id}", (l, max(15, t_ - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        # Draw lane lines and counts
        cv2.line(frame, (x1, 0), (x1, height), (255, 0, 0), 2)
        cv2.line(frame, (x2, 0), (x2, height), (255, 0, 0), 2)
        cv2.putText(frame, f"Lane1: {lane_counts[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Lane2: {lane_counts[1]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Lane3: {lane_counts[2]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        out.write(frame)

        if args.display:
            cv2.imshow("Traffic Counter", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Progress logging
        if frame_idx % 50 == 0:
            elapsed = time.time() - start_frame_time
            avg_per_frame = elapsed / frame_idx
            est_total = avg_per_frame * max_frames
            print(f"[INFO] Processed {frame_idx}/{max_frames} frames, elapsed {elapsed:.2f}s, est total time {est_total:.2f}s")

    cap.release()
    out.release()
    if args.display:
        cv2.destroyAllWindows()

    # Save results with custom tab spacing
    df = pd.DataFrame(records)
    with open(args.output_csv, "w") as f:
        # Header with 2 tabs
        f.write("\t\t".join(df.columns) + "\n")
        # Data rows with 3 tabs
        for _, row in df.iterrows():
            f.write("\t\t\t\t\t  ".join(map(str, row.tolist())) + "\n")

    print(f"[DONE] Video saved: {args.output_video}")
    print(f"[DONE] CSV saved: {args.output_csv}")
    print("Counts per lane:", lane_counts)
    print("Elapsed:", round(time.time() - t0, 2), "s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="YouTube URL to download (optional)")
    parser.add_argument("--video", type=str, help="Local video path (optional)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="COCO-pretrained YOLO model")
    parser.add_argument("--output_video", type=str, default="output.mp4", help="Filename for overlay video")
    parser.add_argument("--output_csv", type=str, default="counts.csv", help="CSV output filename")
    parser.add_argument("--input_name", type=str, default="input_video.mp4", help="Downloaded input filename")
    parser.add_argument("--infer_size", type=int, default=640, help="Size for model inference (smaller=faster)")
    parser.add_argument("--display", action="store_true", help="Show live display window")
    args = parser.parse_args()
    main(args)
