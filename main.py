import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Make sure sort.py and kalman_filter.py are in the same folder

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can change this to yolov8s.pt etc.

# Initialize SORT tracker
tracker = Sort()

# Use webcam (change to "video.mp4" to use a video file)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame)[0]

    detections = []

    for result in results.boxes.data:
        x1, y1, x2, y2, score, class_id = result.tolist()
        if score > 0.4:
            detections.append([x1, y1, x2, y2, score])

    #  Ensure proper shape for SORT
    if len(detections) > 0:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))

    # Track objects
    tracked_objects = tracker.update(detections)

    # Draw bounding boxes and tracking IDs
    for *bbox, obj_id in tracked_objects:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(obj_id)}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show the frame with detections
    cv2.putText(frame, "By - Parth Ingole", (frame.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 Object Detection and Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
