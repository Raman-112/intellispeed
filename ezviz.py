# import cv2
# from ultralytics import YOLO
# import time

# # Load the YOLOv8 model
# model = YOLO("yolov8n.pt")  # Load trained YOLOv8 model

# # Replace this with your RTSP URL
# rtsp_url = "rtsp://admin:GKWOKT@192.168.87.126:554/H.264"

# def connect_to_camera(rtsp_url):
#     """Attempts to connect to the RTSP camera with retries."""
#     cap = cv2.VideoCapture(rtsp_url)
#     while not cap.isOpened():
#         print("Retrying connection to the RTSP camera...")
#         time.sleep(2)  # Wait before retrying
#         cap = cv2.VideoCapture(rtsp_url)
#     return cap

# # Connect to the RTSP camera
# cap = connect_to_camera(rtsp_url)

# previous_positions = {}  # Dictionary to store the previous positions of detected objects for speed calculation

# # Read frames from the camera
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to retrieve frame. Reconnecting...")
#         cap.release()
#         cap = connect_to_camera(rtsp_url)
#         continue

#     # Run YOLOv8 on the current frame
#     results = model.predict(frame, conf=0.5)  # Confidence threshold of 0.5

#     # Extract bounding boxes and class IDs from YOLO output
#     for result in results:
#         boxes = result.boxes.xyxy  # Bounding box coordinates
#         class_ids = result.boxes.cls  # Class IDs
#         confidences = result.boxes.conf  # Confidence scores

#         for i in range(len(boxes)):
#             box = boxes[i].cpu().numpy()  # Get bounding box for each detection
#             class_id = int(class_ids[i].cpu().numpy())
#             confidence = confidences[i].cpu().numpy()

#             # Only focus on 'Car' detections (class_id based on your training)
#             if class_id == 0:  # Assuming 'Car' is labeled as class 0 in your model
#                 x1, y1, x2, y2 = box  # Coordinates of bounding box
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

#                 # Label the car with confidence score
#                 cv2.putText(frame, f"Car: {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#                 # Speed calculation (basic, depending on FPS and distance tracking)
#                 # Add your logic here to track object speed based on frames
#                 current_position = ((x1 + x2) / 2, (y1 + y2) / 2)  # Center of the bounding box
#                 object_id = f"car_{i}"  # Example ID for each car detection

#                 if object_id in previous_positions:
#                     prev_position = previous_positions[object_id]
#                     distance_moved = ((current_position[0] - prev_position[0]) ** 2 + (current_position[1] - prev_position[1]) ** 2) ** 0.5
#                     speed = distance_moved / (1 / cap.get(cv2.CAP_PROP_FPS))  # Speed estimation based on FPS
#                     cv2.putText(frame, f"Speed: {speed:.2f} px/sec", (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#                 previous_positions[object_id] = current_position  # Update position

#     # Display the frame with YOLO detections
#     cv2.imshow("RTSP Live Feed - YOLOv8 Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
#         break

# # Release the capture and close windows
# cap.release()
# cv2.destroyAllWindows()




















# import argparse
# import cv2
# from ultralytics import YOLO
# import supervision as sv
# import numpy as np
# from collections import defaultdict, deque
# import os
# import time

# # Define polygon area for zone filtering (adjust coordinates as needed)
# SOURCE = np.array([[800, 210], [1030, 210], [2000, 900], [-340, 900]])  # Adjust these points based on video resolution

# TARGET_WIDTH = 25
# TARGET_HEIGHT = 250

# TARGET = np.array(
#     [
#         [0, 0],
#         [TARGET_WIDTH - 1, 0],
#         [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
#         [0, TARGET_HEIGHT - 1],
#     ]
# )

# class ViewTransformer:
#     def __init__(self, source: np.ndarray, target: np.ndarray):
#         source = source.astype(np.float32)
#         target = target.astype(np.float32)
#         self.m = cv2.getPerspectiveTransform(source, target)
#         print("Transformation matrix:", self.m)  # Debug: print transformation matrix

#     def transform_points(self, points: np.ndarray) -> np.ndarray:
#         if points is None or points.size == 0:
#             print("No points to transform.")  # Debug: handle case when no points
#             return None
#         reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
#         transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
#         return transformed_points.reshape(-1, 2)

# # Function to connect to RTSP stream
# def connect_to_camera(rtsp_url):
#     """Attempts to connect to the RTSP camera with retries."""
#     cap = cv2.VideoCapture(rtsp_url)
#     while not cap.isOpened():
#         print("Retrying connection to the RTSP camera...")
#         time.sleep(2)  # Wait before retrying
#         cap = cv2.VideoCapture(rtsp_url)
#     return cap

# # Main program
# if __name__ == "__main__":
#     # Load YOLOv8 model
#     model = YOLO("yolov8m.pt")

#     # Connect to the RTSP stream
#     rtsp_url = "rtsp://admin:GKWOKT@192.168.87.126:554/H.264"  # Update with your RTSP URL
#     cap = connect_to_camera(rtsp_url)

#     # Get frame rate (FPS) from the camera
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

#     # Initialize ByteTrack for tracking
#     byte_track = sv.ByteTrack(frame_rate=fps)

#     # Set manual thickness and text scale
#     thickness = 1  # Thinner bounding boxes
#     text_scale = 0.5  # Reduce text size

#     # Create annotators for bounding boxes and labels
#     bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
#     label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

#     # Define a polygon zone (optional)
#     polygon_zone = sv.PolygonZone(SOURCE, frame_resolution_wh=frame_size)
#     view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

#     # Initialize the dictionary to store coordinates with deque
#     coordinates = defaultdict(lambda: deque(maxlen=int(fps)))

#     # Vehicle class IDs (from the COCO dataset)
#     VEHICLE_CLASS_IDS = [2, 5, 7]  # Car: 2, Bus: 5, Truck: 7

#     # Process frames from the RTSP stream
#     previous_positions = {}

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to retrieve frame. Reconnecting...")
#             cap.release()
#             cap = connect_to_camera(rtsp_url)
#             continue

#         # Run YOLO detection on the current frame
#         result = model(frame, conf=0.3)[0]

#         # Convert YOLO results to detections
#         detections = sv.Detections.from_ultralytics(result)

#         # Debug: print out detected class IDs before filtering
#         print("Detected class IDs:", detections.class_id)

#         # Filter for vehicles (car, bus, truck)
#         detections = detections[(detections.class_id == 2) | (detections.class_id == 5) | (detections.class_id == 7)]

#         # Check if any vehicles are detected
#         if len(detections) == 0:
#             print("No vehicles detected in the frame.")
#             continue  # Skip the rest of the loop if no vehicles are detected

#         # Filter detections inside the polygon zone
#         detections = detections[polygon_zone.trigger(detections)]

#         # Track the detections
#         tracked_detections = byte_track.update_with_detections(detections=detections)

#         # Print tracker IDs
#         print(f"Tracker IDs: {tracked_detections.tracker_id}")

#         # Create labels for each vehicle detection
#         points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

#         # Check if points are valid before transformation
#         print("Original points:", points)  # Debug: print original points
#         if points.size > 0:
#             points = view_transformer.transform_points(points=points)
#             if points is not None:
#                 points = points.astype(int)
#                 print("Transformed points:", points)  # Debug: print transformed points
#         else:
#             print("No valid points to transform.")
#             continue

#         labels = []
#         for tracker_id, [_, y] in zip(tracked_detections.tracker_id, points):
#             coordinates[tracker_id].append(y)
#             if len(coordinates[tracker_id]) < fps / 2:
#                 labels.append(f"#{tracker_id}")
#             else:
#                 coordinate_start = coordinates[tracker_id][-1]
#                 coordinate_end = coordinates[tracker_id][0]
#                 distance = abs(coordinate_start - coordinate_end)
#                 time_elapsed = len(coordinates[tracker_id]) / fps
#                 speed = distance / time_elapsed * 3.6  # Convert to km/h
#                 labels.append(f"#{tracker_id}  {int(speed)} km/h")

#         # Annotate the frame with bounding boxes and polygon zone
#         annotated_frame = frame.copy()
#         annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)

#         # Annotate bounding boxes and labels
#         annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
#         annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)

#         # Display the annotated frame in a window
#         cv2.imshow("RTSP Live Stream - YOLOv8 Vehicle Detection", annotated_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()



















import cv2
from ultralytics import YOLO
import time
import easyocr  # For OCR
import requests  # To send data to web API or Firebase (optional)

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Load trained YOLOv8 model

# Replace this with your RTSP URL
rtsp_url = "rtsp://admin:GKWOKT@192.168.87.126:554/H.264"

# Speed limit (in px/sec, needs calibration)
SPEED_LIMIT = 100

def connect_to_camera(rtsp_url):
    """Attempts to connect to the RTSP camera with retries."""
    cap = cv2.VideoCapture(rtsp_url)
    while not cap.isOpened():
        print("Retrying connection to the RTSP camera...")
        time.sleep(2)  # Wait before retrying
        cap = cv2.VideoCapture(rtsp_url)
    return cap

# Connect to the RTSP camera
cap = connect_to_camera(rtsp_url)

previous_positions = {}  # Dictionary to store the previous positions of detected objects for speed calculation

# Initialize OCR reader
ocr_reader = easyocr.Reader(['en'])

# Read frames from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame. Reconnecting...")
        cap.release()
        cap = connect_to_camera(rtsp_url)
        continue

    # Run YOLOv8 on the current frame
    results = model.predict(frame, conf=0.5)  # Confidence threshold of 0.5

    # Extract bounding boxes and class IDs from YOLO output
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates
        class_ids = result.boxes.cls  # Class IDs
        confidences = result.boxes.conf  # Confidence scores

        for i in range(len(boxes)):
            box = boxes[i].cpu().numpy()  # Get bounding box for each detection
            class_id = int(class_ids[i].cpu().numpy())
            confidence = confidences[i].cpu().numpy()

            # Only focus on 'Car' detections (class_id based on your training)
            if class_id == 0:  # Assuming 'Car' is labeled as class 0 in your model
                x1, y1, x2, y2 = box  # Coordinates of bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                # Label the car with confidence score
                cv2.putText(frame, f"Car: {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Speed calculation (basic, depending on FPS and distance tracking)
                current_position = ((x1 + x2) / 2, (y1 + y2) / 2)  # Center of the bounding box
                object_id = f"car_{i}"  # Example ID for each car detection

                if object_id in previous_positions:
                    prev_position = previous_positions[object_id]
                    distance_moved = ((current_position[0] - prev_position[0]) ** 2 + (current_position[1] - prev_position[1]) ** 2) ** 0.5
                    speed = distance_moved / (1 / cap.get(cv2.CAP_PROP_FPS))  # Speed estimation based on FPS
                    cv2.putText(frame, f"Speed: {speed:.2f} px/sec", (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Check if speed exceeds the limit
                    if speed > SPEED_LIMIT:
                        cv2.putText(frame, "SPEEDING!", (int(x1), int(y1) - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        # Send alert (Example: Send data to the website)
                        data = {
                            "number_plate": "UNKNOWN",  # Placeholder for now
                            "speed": speed,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        print(f"Speeding detected: {data}")

                        # Uncomment this if you want to send it to your website
                        # requests.post("YOUR_API_URL", json=data)

                # Run OCR on the detected car region to identify the number plate
                car_region = frame[int(y1):int(y2), int(x1):int(x2)]
                ocr_results = ocr_reader.readtext(car_region)

                if ocr_results:
                    number_plate = ocr_results[0][-2]  # The recognized text
                    print(f"Detected number plate: {number_plate}")
                    cv2.putText(frame, f"Plate: {number_plate}", (int(x1), int(y2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                    # Optionally, update the speeding data with the number plate
                    data["number_plate"] = number_plate

                previous_positions[object_id] = current_position  # Update position

    # Display the frame with YOLO detections
    cv2.imshow("RTSP Live Feed - YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
