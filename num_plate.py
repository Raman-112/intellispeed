

# # import argparse
# # import cv2
# # from ultralytics import YOLO
# # import supervision as sv
# # import numpy as np
# # from collections import defaultdict, deque
# # import os
# # import easyocr  # For OCR to read the number plates

# # # Function to parse command-line arguments
# # def parse_arguments() -> argparse.Namespace:
# #     parser = argparse.ArgumentParser(
# #         description="Vehicle speed estimation with number plate detection using YOLOv8 and Supervision"
# #     )
# #     parser.add_argument(
# #         "--source_video_path",
# #         default="C:/Users/User/Desktop/vid/highway.mp4",
# #         help="Path to the source video file",
# #         type=str
# #     )
# #     return parser.parse_args()

# # # Set up the source and target for perspective transform
# # SOURCE = np.array([[810, 210], [1030, 210], [1840, 658], [-230, 658]])  # Adjust these points based on video resolution
# # TARGET_WIDTH = 25
# # TARGET_HEIGHT = 250

# # TARGET = np.array(
# #     [
# #         [0, 0],
# #         [TARGET_WIDTH - 1, 0],
# #         [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
# #         [0, TARGET_HEIGHT - 1],
# #     ]
# # )

# # class ViewTransformer:
# #     def __init__(self, source: np.ndarray, target: np.ndarray):
# #         source = source.astype(np.float32)
# #         target = target.astype(np.float32)
# #         self.m = cv2.getPerspectiveTransform(source, target)

# #     def transform_points(self, points: np.ndarray) -> np.ndarray:
# #         reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
# #         transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
# #         return transformed_points.reshape(-1, 2)

# # # Main program
# # if __name__ == "__main__":
# #     args = parse_arguments()

# #     # Get video information
# #     video_info = sv.VideoInfo.from_video_path(args.source_video_path)

# #     # Load YOLOv8 model for vehicle detection
# #     vehicle_model = YOLO("yolov8m.pt")

# #     # Load another YOLOv8 or any model trained for number plate detection
# #     number_plate_model = YOLO("yolov8n-license-plate.pt")  # Example model for number plates

# #     # Initialize ByteTrack for tracking
# #     byte_track = sv.ByteTrack(frame_rate=video_info.fps)

# #     # Set manual thickness and text scale
# #     thickness = 1
# #     text_scale = 0.8

# #     # Create annotators for bounding boxes and labels
# #     bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
# #     label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

# #     # Initialize OCR reader
# #     ocr_reader = easyocr.Reader(['en'])  # English language model for OCR

# #     # Process video frames
# #     frame_generator = sv.get_video_frames_generator(args.source_video_path)

# #     # Define a polygon zone (optional)
# #     polygon_zone = sv.PolygonZone(SOURCE)
# #     view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

# #     # Initialize the dictionary to store coordinates with deque
# #     coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

# #     # Vehicle class IDs (from the COCO dataset)
# #     VEHICLE_CLASS_IDS = [2, 5, 7]  # Car: 2, Bus: 5, Truck: 7

# #     # Prepare video writer
# #     output_path = os.path.expanduser("C:/Users/User/Desktop/vidi_with_number_plate.mp4")  
# #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# #     fps = video_info.fps
# #     frame_size = (video_info.resolution_wh[0], video_info.resolution_wh[1])
# #     video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# #     # Check if video writer is opened correctly
# #     if not video_writer.isOpened():
# #         print("Error: VideoWriter not opened!")

# #     # Dictionary to keep track of labels by tracker ID
# #     label_dict = {}

# #     for frame in frame_generator:
# #         # Run YOLO detection on the current frame for vehicles
# #         result = vehicle_model(frame, conf=0.3)[0]

# #         # Convert YOLO results to detections
# #         detections = sv.Detections.from_ultralytics(result)

# #         # Filter for vehicles (car, bus, truck) based on COCO class IDs
# #         detections = detections[(detections.class_id == 2) | (detections.class_id == 5) | (detections.class_id == 7)]

# #         # Filter detections inside the polygon zone
# #         detections = detections[polygon_zone.trigger(detections)]

# #         # Track the detections
# #         tracked_detections = byte_track.update_with_detections(detections=detections)

# #         # Process number plate detection within vehicle bounding boxes
# #         for box in tracked_detections.xyxy:
# #             x1, y1, x2, y2 = map(int, box)
# #             vehicle_crop = frame[y1:y2, x1:x2]

# #             # Run number plate detection on cropped vehicle frame
# #             plate_result = number_plate_model(vehicle_crop, conf=0.3)[0]
# #             plate_detections = sv.Detections.from_ultralytics(plate_result)

# #             # If a number plate is detected
# #             for plate_box in plate_detections.xyxy:
# #                 px1, py1, px2, py2 = map(int, plate_box)
# #                 number_plate_crop = vehicle_crop[py1:py2, px1:px2]

# #                 # Use OCR to read the number plate
# #                 ocr_results = ocr_reader.readtext(number_plate_crop)

# #                 # Get the number plate text
# #                 for res in ocr_results:
# #                     print(f"Detected number plate: {res[1]}")  # Log detected number plate

# #         # Annotate the frame with bounding boxes and polygon zone
# #         annotated_frame = frame.copy()
# #         annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)

# #         # Annotate bounding boxes and labels on the frame
# #         annotated_frame = bounding_box_annotator.annotate(
# #             scene=annotated_frame, detections=tracked_detections
# #         )

# #         # Write the annotated frame to the output video
# #         video_writer.write(annotated_frame)
# #         print(f"Writing frame to video...")

# #     # Release resources
# #     video_writer.release()
# #     print(f"Video saved at {output_path}")



# import argparse
# import cv2
# from ultralytics import YOLO  # YOLOv8 model loading
# import supervision as sv
# import numpy as np
# from collections import defaultdict, deque
# import os
# import easyocr  # For OCR to read the number plates

# # Function to parse command-line arguments
# def parse_arguments() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Vehicle speed estimation with number plate OCR using YOLOv8 and EasyOCR"
#     )
#     parser.add_argument(
#         "--source_video_path",
#         default="C:/Users/User/Desktop/vid/car.mp4",
#         help="Path to the source video file",
#         type=str
#     )
#     return parser.parse_args()

# # Set up the source and target for perspective transform
# SOURCE = np.array([[810, 210], [1030, 210], [1840, 658], [-230, 658]])  # Adjust these points based on video resolution
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

#     def transform_points(self, points: np.ndarray) -> np.ndarray:
#         reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
#         transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
#         return transformed_points.reshape(-1, 2)

# # Main program
# if __name__ == "__main__":
#     args = parse_arguments()

#     # Get video information
#     video_info = sv.VideoInfo.from_video_path(args.source_video_path)

#     # Load YOLOv8 model for vehicle detection
#     vehicle_model = YOLO("yolov8m.pt")  # You can use yolov8n.pt or yolov8s.pt

#     # Initialize ByteTrack for tracking
#     byte_track = sv.ByteTrack(frame_rate=video_info.fps)

#     # Set manual thickness and text scale
#     thickness = 1
#     text_scale = 0.8

#     # Create annotators for bounding boxes and labels
#     bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
#     label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

#     # Initialize OCR reader (EasyOCR)
#     ocr_reader = easyocr.Reader(['en'])  # English language model for OCR

#     # Process video frames
#     frame_generator = sv.get_video_frames_generator(args.source_video_path)

#     # Define a polygon zone (optional)
#     polygon_zone = sv.PolygonZone(SOURCE)
#     view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

#     # Initialize the dictionary to store coordinates with deque
#     coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

#     # Vehicle class IDs (from the COCO dataset)
#     VEHICLE_CLASS_IDS = [2, 5, 7]  # Car: 2, Bus: 5, Truck: 7

#     # Prepare video writer
#     output_path = os.path.expanduser("C:/Users/User/Desktop/w_plate.mp4")  
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     fps = video_info.fps
#     frame_size = (video_info.resolution_wh[0], video_info.resolution_wh[1])
#     video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

#     # Check if video writer is opened correctly
#     if not video_writer.isOpened():
#         print("Error: VideoWriter not opened!")

#     for frame in frame_generator:
#         # Run YOLO detection on the current frame for vehicles
#         result = vehicle_model(frame, conf=0.3)[0]

#         # Convert YOLO results to detections
#         detections = sv.Detections.from_ultralytics(result)

#         # Filter for vehicles (car, bus, truck) based on COCO class IDs
#         detections = detections[(detections.class_id == 2) | (detections.class_id == 5) | (detections.class_id == 7)]

#         # Filter detections inside the polygon zone
#         detections = detections[polygon_zone.trigger(detections)]

#         # Track the detections
#         tracked_detections = byte_track.update_with_detections(detections=detections)

#         # Process number plate OCR within vehicle bounding boxes
#         for box in tracked_detections.xyxy:
#             x1, y1, x2, y2 = map(int, box)
#             vehicle_crop = frame[y1:y2, x1:x2]

#             # Use OCR to attempt reading the number plate directly from the vehicle crop
#             ocr_results = ocr_reader.readtext(vehicle_crop)

#             # Get the number plate text if available
#             for res in ocr_results:
#                 print(f"Detected number plate: {res[1]}")  # Log detected number plate

#         # Annotate the frame with bounding boxes and polygon zone
#         annotated_frame = frame.copy()
#         annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)

#         # Annotate bounding boxes and labels on the frame
#         annotated_frame = bounding_box_annotator.annotate(
#             scene=annotated_frame, detections=tracked_detections
#         )

#         # Write the annotated frame to the output video
#         video_writer.write(annotated_frame)
#         print(f"Writing frame to video...")

#     # Release resources
#     video_writer.release()
#     print(f"Video saved at {output_path}")

































import argparse
import cv2
from ultralytics import YOLO  # YOLOv8 model loading
import supervision as sv
import numpy as np
from collections import defaultdict, deque
import os
import easyocr  # For OCR to read the number plates

# Function to parse command-line arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle speed estimation with number plate OCR using YOLOv8 and EasyOCR"
    )
    parser.add_argument(
        "--source_video_path",
        default="C:/Users/User/Desktop/vid/car.mp4",
        help="Path to the source video file",
        type=str
    )
    return parser.parse_args()

# Set up the source and target for perspective transform
SOURCE = np.array([[810, 210], [1030, 210], [1840, 658], [-230, 658]])  # Adjust these points based on video resolution
TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# Main program
if __name__ == "__main__":
    args = parse_arguments()

    # Get video information
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)

    # Load YOLOv8 model for vehicle detection
    vehicle_model = YOLO("yolov8m.pt")  # You can use yolov8n.pt or yolov8s.pt

    # Initialize ByteTrack for tracking
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    # Set manual thickness and text scale
    thickness = 1
    text_scale = 0.8

    # Create annotators for bounding boxes and labels
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    # Initialize OCR reader (EasyOCR)
    ocr_reader = easyocr.Reader(['en'])  # English language model for OCR

    # Process video frames
    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    # Define a polygon zone (optional)
    polygon_zone = sv.PolygonZone(SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    # Initialize the dictionary to store coordinates with deque
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    # Vehicle class IDs (from the COCO dataset)
    VEHICLE_CLASS_IDS = [2, 5, 7]  # Car: 2, Bus: 5, Truck: 7

    # Prepare video writer
    output_path = os.path.expanduser("C:/Users/User/Desktop/weis_plate.mp4")  
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = video_info.fps
    frame_size = (video_info.resolution_wh[0], video_info.resolution_wh[1])
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Check if video writer is opened correctly
    if not video_writer.isOpened():
        print("Error: VideoWriter not opened!")

    for frame in frame_generator:
        # Run YOLO detection on the current frame for vehicles
        result = vehicle_model(frame, conf=0.3)[0]

        # Convert YOLO results to detections
        detections = sv.Detections.from_ultralytics(result)

        # Filter for vehicles (car, bus, truck) based on COCO class IDs
        detections = detections[(detections.class_id == 2) | (detections.class_id == 5) | (detections.class_id == 7)]

        # Filter detections inside the polygon zone
        detections = detections[polygon_zone.trigger(detections)]

        # Track the detections
        tracked_detections = byte_track.update_with_detections(detections=detections)

        # Process number plate OCR within vehicle bounding boxes
        for i, box in enumerate(tracked_detections.xyxy):
            x1, y1, x2, y2 = map(int, box)
            vehicle_crop = frame[y1:y2, x1:x2]

            # Use OCR to attempt reading the number plate directly from the vehicle crop
            ocr_results = ocr_reader.readtext(vehicle_crop)

            # Get the number plate text if available and annotate it on the frame
            number_plate_text = ""
            if ocr_results:
                number_plate_text = ocr_results[0][1]  # Take the first detected text
                print(f"Detected number plate: {number_plate_text}")  # Log detected number plate

            # Annotate the detected number plate text on the frame
            cv2.putText(frame, number_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Annotate the frame with bounding boxes and polygon zone
        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)

        # Annotate bounding boxes and labels on the frame
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=tracked_detections
        )
        # Write the annotated frame to the output video
        video_writer.write(annotated_frame)
        print(f"Writing frame to video...")      
    # Release resources
    video_writer.release()
    print(f"Video saved at {output_path}")
