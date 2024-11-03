
# import argparse
# import cv2
# from ultralytics import YOLO
# import supervision as sv
# import numpy as np
# from collections import defaultdict, deque
# import os
# import torch
# from torchvision import models, transforms

# def parse_arguments() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Vehicle speed estimation using YOLOv8 and Supervision"
#     )
#     parser.add_argument(
#         "--source_video_path",
#         default="C:/Users/User/Desktop/vid/highway.mp4",
#         help="Path to the source video file",
#         type=str
#     )
#     return parser.parse_args()

# def segment_road(frame: np.ndarray) -> np.ndarray:
#     preprocess = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     input_tensor = preprocess(frame).unsqueeze(0)
#     with torch.no_grad():
#         output = road_model(input_tensor)['out'][0]
#     output_predictions = output.argmax(0).byte().cpu().numpy()
#     return output_predictions

# def get_road_bbox(segmentation: np.ndarray) -> np.ndarray:
#     road_mask = segmentation == 1  # Assuming class 1 is the road
#     y_indices, x_indices = np.where(road_mask)
#     if len(y_indices) == 0 or len(x_indices) == 0:
#         return np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
#     top_left = (x_indices.min(), y_indices.min())
#     bottom_right = (x_indices.max(), y_indices.max())
#     return np.array([
#         [top_left[0], top_left[1]],
#         [bottom_right[0], top_left[1]],
#         [bottom_right[0], bottom_right[1]],
#         [top_left[0], bottom_right[1]],
#     ])

# # Load the road segmentation model
# road_model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# # SOURCE and TARGET points for perspective transform
# # SOURCE = np.array([[380, 150], [580, 150], [1810, 658], [-620, 658]])

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
#         if transformed_points is None:
#             print("Perspective transform failed. Check your source and target points.")
#             return points  # Return original points if transformation fails
#         return transformed_points.reshape(-1, 2)

# # Main program
# if __name__ == "__main__":
#     args = parse_arguments()

#     # Get video information using OpenCV
#     cap = cv2.VideoCapture(args.source_video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     class VideoInfo:
#         def __init__(self, fps, resolution_wh):
#             self.fps = fps
#             self.resolution_wh = resolution_wh

#     video_info = VideoInfo(fps=fps, resolution_wh=(width, height))

#     # Load YOLOv8 model
#     model = YOLO("yolov8n.pt")

#     # Initialize ByteTrack for tracking
#     byte_track = sv.ByteTrack(frame_rate=video_info.fps)

#     # Set manual thickness and text scale (reduce thickness to avoid hiding vehicles)
#     thickness = 1  # Set to 1 for thinner bounding boxes
#     text_scale = 0.8  # Increased text scale for better visibility

#     # Create annotators for bounding boxes and labels
#     bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)

#     # Create the label annotator
#     label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

#     # Process video frames
#     frame_generator = sv.get_video_frames_generator(args.source_video_path)

#     # Initialize the dictionary to store coordinates with deque
#     coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

#     # Vehicle class IDs (from the COCO dataset)
#     VEHICLE_CLASS_IDS = [2, 5, 7]  # Car: 2, Bus: 5, Truck: 7

#     # Prepare video writer
#     output_path = os.path.expanduser("C:/Users/User/Desktop/highe.mp4")  # Save video to desktop
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     fps = video_info.fps
#     frame_size = (video_info.resolution_wh[0], video_info.resolution_wh[1])  # Ensure correct order: (width, height)
#     video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

#     # Check if video writer is opened correctly
#     if not video_writer.isOpened():
#         print("Error: VideoWriter not opened!")

#     # Dictionary to keep track of labels by tracker ID
#     label_dict = {}

#     for frame in frame_generator:
#         # Perform road segmentation
#         road_segmentation = segment_road(frame)
#         SOURCE = get_road_bbox(road_segmentation)

#         # Define a polygon zone (optional)
#         polygon_zone = sv.PolygonZone(SOURCE)  # Adjusted to match the latest library version
#         view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

#         # Run YOLO detection on the current frame with a lower confidence threshold
#         result = model(frame, conf=0.3)[0]
        
#         # Convert YOLO results to detections
#         detections = sv.Detections.from_ultralytics(result)
        
#         # Filter for vehicles (car, bus, truck) based on COCO class IDs
#         detections = detections[(detections.class_id == 2) | (detections.class_id == 5) | (detections.class_id == 7)]
        
#         # Filter detections inside the polygon zone
#         detections = detections[polygon_zone.trigger(detections)]
        
#         # Track the detections
#         tracked_detections = byte_track.update_with_detections(detections=detections)

#         # Print tracker ID to check if they are being assigned correctly
#         print(f"Tracker IDs: {tracked_detections.tracker_id}")

#         # Create labels for each vehicle detection (showing tracker ID)
#         points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
#         points = view_transformer.transform_points(points=points).astype(int)

#         labels = []
#         label_colors = []  # List to store colors corresponding to each label
#         for tracker_id, [_, y], class_id in zip(tracked_detections.tracker_id, points, detections.class_id):
#             coordinates[tracker_id].append(y)
#             if len(coordinates[tracker_id]) < video_info.fps / 2:
#                 labels.append(f"#{tracker_id}")
#             else:
#                 coordinate_start = coordinates[tracker_id][-1]
#                 coordinate_end = coordinates[tracker_id][0]
#                 distance = abs(coordinate_start - coordinate_end)
#                 time = len(coordinates[tracker_id]) / video_info.fps
#                 speed = distance / time * 3.6
#                 labels.append(f"#{tracker_id}  {int(speed)} km/h")

#             # Set color based on vehicle class (blue for cars, red for trucks)
#             if class_id == 2:  # Car
#                 label_colors.append(sv.Color.BLUE)
#             elif class_id == 7:  # Truck
#                 label_colors.append(sv.Color.RED)
#             else:
#                 label_colors.append(sv.Color.WHITE)  # Default to white for other vehicles

#         # Annotate the frame with bounding boxes and polygon zone
#         annotated_frame = frame.copy()
#         annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)

#         # Annotate bounding boxes and labels on the frame
#         annotated_frame = bounding_box_annotator.annotate(
#             scene=annotated_frame, detections=tracked_detections
#         )
        
#         # Annotate tracker ID labels with different colors (blue for cars, red for trucks)
#         for label, color in zip(labels, label_colors):
#             annotated_frame = label_annotator.annotate(
#                 scene=annotated_frame, detections=tracked_detections, labels=labels
#             )

#         # Write the annotated frame to the output video
#         video_writer.write(annotated_frame)
#         print(f"Writing frame to video...")  # Print a message for each frame written

#     # Release resources
#     video_writer.release()
#     print(f"Video saved at {output_path}")
































# # import argparse
# # import cv2
# # from ultralytics import YOLO
# # import supervision as sv
# # import numpy as np
# # from collections import defaultdict, deque
# # import os
# # import torch
# # from torchvision import models, transforms

# # def parse_arguments() -> argparse.Namespace:
# #     parser = argparse.ArgumentParser(
# #         description="Vehicle speed estimation using YOLOv8 and Supervision"
# #     )
# #     parser.add_argument(
# #         "--source_video_path",
# #         default="C:/Users/User/Desktop/vid/highway.mp4",
# #         help="Path to the source video file",
# #         type=str
# #     )
# #     return parser.parse_args()

# # def segment_road(frame: np.ndarray) -> np.ndarray:
# #     preprocess = transforms.Compose([
# #         transforms.ToPILImage(),
# #         transforms.Resize((256, 256)),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #     ])
# #     input_tensor = preprocess(frame).unsqueeze(0)
# #     with torch.no_grad():
# #         output = road_model(input_tensor)['out'][0]
# #     output_predictions = output.argmax(0).byte().cpu().numpy()
# #     return output_predictions

# # def get_road_bbox(segmentation: np.ndarray) -> np.ndarray:
# #     road_mask = segmentation == 1  # Assuming class 1 is the road
# #     y_indices, x_indices = np.where(road_mask)
# #     if len(y_indices) == 0 or len(x_indices) == 0:
# #         # Fallback coordinates if no road is detected
# #         print("Warning: No road detected, using fallback coordinates.")
# #         return np.array([[380, 150], [580, 150], [1810, 658], [-620, 658]])
# #     top_left = (x_indices.min(), y_indices.min())
# #     bottom_right = (x_indices.max(), y_indices.max())
# #     return np.array([
# #         [top_left[0], top_left[1]],
# #         [bottom_right[0], top_left[1]],
# #         [bottom_right[0], bottom_right[1]],
# #         [top_left[0], bottom_right[1]],
# #     ])

# # # Load the road segmentation model
# # road_model = models.segmentation.deeplabv3_resnet101(weights="DeepLabV3_ResNet101_Weights.DEFAULT").eval()

# # # TARGET points for perspective transform (static)
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
# #     def __init__(self, target: np.ndarray):
# #         self.target = target.astype(np.float32)
# #         self.m = None  # Initialize m as None

# #     def set_source(self, source: np.ndarray):
# #         source = source.astype(np.float32)
# #         self.m = cv2.getPerspectiveTransform(source, self.target)

# #     def transform_points(self, points: np.ndarray) -> np.ndarray:
# #         if self.m is None:
# #             print("Warning: Perspective transform matrix 'm' is not set. Returning original points.")
# #             return points  # Return original points if transformation matrix is not set
# #         reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
# #         transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
# #         return transformed_points.reshape(-1, 2)

# # # Define VideoInfo class
# # class VideoInfo:
# #     def __init__(self, fps, resolution_wh):
# #         self.fps = fps
# #         self.resolution_wh = resolution_wh

# # # Main program
# # if __name__ == "__main__":
# #     args = parse_arguments()
# #     cap = cv2.VideoCapture(args.source_video_path)
# #     fps = int(cap.get(cv2.CAP_PROP_FPS))
# #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# #     video_info = VideoInfo(fps=fps, resolution_wh=(width, height))

# #     # Load YOLOv8 model
# #     model = YOLO("yolov8n.pt")
# #     byte_track = sv.ByteTrack(frame_rate=video_info.fps)
# #     thickness = 1
# #     text_scale = 0.8
# #     bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
# #     label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

# #     frame_generator = sv.get_video_frames_generator(args.source_video_path)
# #     coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
# #     VEHICLE_CLASS_IDS = [2, 5, 7]
# #     output_path = os.path.expanduser("C:/Users/User/Desktop/highwa.mp4")
# #     video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
# #     view_transformer = ViewTransformer(target=TARGET)

# #     for frame in frame_generator:
# #         road_segmentation = segment_road(frame)
# #         SOURCE = get_road_bbox(road_segmentation)
        
# #         if np.any(SOURCE):
# #             view_transformer.set_source(SOURCE)
        
# #         polygon_zone = sv.PolygonZone(SOURCE)
# #         result = model(frame, conf=0.3)[0]
# #         detections = sv.Detections.from_ultralytics(result)
# #         detections = detections[(detections.class_id == 2) | (detections.class_id == 5) | (detections.class_id == 7)]
# #         detections = detections[polygon_zone.trigger(detections)]
# #         tracked_detections = byte_track.update_with_detections(detections=detections)
        
# #         points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
# #         points = view_transformer.transform_points(points=points).astype(int)
        
# #         labels, label_colors = [], []
# #         for tracker_id, [_, y], class_id in zip(tracked_detections.tracker_id, points, detections.class_id):
# #             coordinates[tracker_id].append(y)
# #             if len(coordinates[tracker_id]) < video_info.fps / 2:
# #                 labels.append(f"#{tracker_id}")
# #             else:
# #                 distance = abs(coordinates[tracker_id][-1] - coordinates[tracker_id][0])
# #                 time = len(coordinates[tracker_id]) / video_info.fps
# #                 speed = distance / time * 3.6
# #                 labels.append(f"#{tracker_id}  {int(speed)} km/h")
# #             label_colors.append(sv.Color.BLUE if class_id == 2 else sv.Color.RED if class_id == 7 else sv.Color.WHITE)

# #         annotated_frame = frame.copy()
# #         annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)
# #         annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
        
# #         for label, color in zip(labels, label_colors):
# #             annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)
        
# #         video_writer.write(annotated_frame)

# #     video_writer.release()
# #     print(f"Video saved at {output_path}")

























# # import argparse
# # import cv2
# # import torch
# # from torchvision import models, transforms
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from ultralytics import YOLO
# # import supervision as sv
# # from collections import defaultdict, deque
# # import os

# # def parse_arguments() -> argparse.Namespace:
# #     parser = argparse.ArgumentParser(
# #         description="Vehicle speed estimation using YOLOv8 and Supervision with automatic road detection"
# #     )
# #     parser.add_argument(
# #         "--source_video_path",
# #         default="C:/Users/User/Desktop/vid/highway.mp4",
# #         help="Path to the source video file",
# #         type=str
# #     )
# #     return parser.parse_args()

# # def segment_road(frame: np.ndarray) -> np.ndarray:
# #     preprocess = transforms.Compose([
# #         transforms.ToPILImage(),
# #         transforms.Resize((256, 256)),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #     ])
# #     input_tensor = preprocess(frame).unsqueeze(0)
# #     with torch.no_grad():
# #         output = road_model(input_tensor)['out'][0]
# #     output_predictions = output.argmax(0).byte().cpu().numpy()

# #     # Visualize the segmentation result
# #     plt.figure(figsize=(10, 5))
# #     plt.subplot(1, 2, 1)
# #     plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# #     plt.title("Original Frame")
# #     plt.axis("off")

# #     plt.subplot(1, 2, 2)
# #     plt.imshow(output_predictions, cmap="gray")
# #     plt.title("Road Segmentation Output")
# #     plt.axis("off")
# #     plt.show()

# #     return output_predictions

# # def get_road_bbox(segmentation: np.ndarray) -> np.ndarray:
# #     road_mask = segmentation == 1  # Assuming class 1 is the road
# #     y_indices, x_indices = np.where(road_mask)
# #     if len(y_indices) == 0 or len(x_indices) == 0:
# #         print("Warning: No road detected, using fallback coordinates.")
# #         return np.array([[380, 150], [580, 150], [1810, 658], [-620, 658]])
# #     top_left = (x_indices.min(), y_indices.min())
# #     bottom_right = (x_indices.max(), y_indices.max())
# #     return np.array([
# #         [top_left[0], top_left[1]],
# #         [bottom_right[0], top_left[1]],
# #         [bottom_right[0], bottom_right[1]],
# #         [top_left[0], bottom_right[1]],
# #     ])

# # # Load the road segmentation model
# # road_model = models.segmentation.deeplabv3_resnet101(weights="DeepLabV3_ResNet101_Weights.DEFAULT").eval()

# # # TARGET points for perspective transform (static)
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
# #     def __init__(self, target: np.ndarray):
# #         self.target = target.astype(np.float32)
# #         self.m = None

# #     def set_source(self, source: np.ndarray):
# #         source = source.astype(np.float32)
# #         self.m = cv2.getPerspectiveTransform(source, self.target)

# #     def transform_points(self, points: np.ndarray) -> np.ndarray:
# #         if self.m is None:
# #             print("Warning: Perspective transform matrix 'm' is not set. Returning original points.")
# #             return points
# #         reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
# #         transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
# #         return transformed_points.reshape(-1, 2)

# # class VideoInfo:
# #     def __init__(self, fps, resolution_wh):
# #         self.fps = fps
# #         self.resolution_wh = resolution_wh

# # # Main program
# # if __name__ == "__main__":
# #     args = parse_arguments()
# #     cap = cv2.VideoCapture(args.source_video_path)
# #     fps = int(cap.get(cv2.CAP_PROP_FPS))
# #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# #     video_info = VideoInfo(fps=fps, resolution_wh=(width, height))

# #     # Load YOLOv8 model
# #     model = YOLO("yolov8n.pt")
# #     byte_track = sv.ByteTrack(frame_rate=video_info.fps)
# #     thickness = 1
# #     text_scale = 0.8
# #     bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
# #     label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

# #     frame_generator = sv.get_video_frames_generator(args.source_video_path)
# #     coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
# #     VEHICLE_CLASS_IDS = [2, 5, 7]
# #     output_path = os.path.expanduser("C:/Users/User/Desktop/highway_output.mp4")
# #     video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
# #     view_transformer = ViewTransformer(target=TARGET)

# #     for frame in frame_generator:
# #         road_segmentation = segment_road(frame)
# #         SOURCE = get_road_bbox(road_segmentation)
        
# #         if np.any(SOURCE):
# #             view_transformer.set_source(SOURCE)
        
# #         polygon_zone = sv.PolygonZone(SOURCE)
# #         result = model(frame, conf=0.3)[0]
# #         detections = sv.Detections.from_ultralytics(result)
# #         detections = detections[(detections.class_id == 2) | (detections.class_id == 5) | (detections.class_id == 7)]
# #         detections = detections[polygon_zone.trigger(detections)]
# #         tracked_detections = byte_track.update_with_detections(detections=detections)
        
# #         points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
# #         points = view_transformer.transform_points(points=points).astype(int)
        
# #         labels, label_colors = [], []
# #         for tracker_id, [_, y], class_id in zip(tracked_detections.tracker_id, points, detections.class_id):
# #             coordinates[tracker_id].append(y)
# #             if len(coordinates[tracker_id]) < video_info.fps / 2:
# #                 labels.append(f"#{tracker_id}")
# #             else:
# #                 distance = abs(coordinates[tracker_id][-1] - coordinates[tracker_id][0])
# #                 time = len(coordinates[tracker_id]) / video_info.fps
# #                 speed = distance / time * 3.6
# #                 labels.append(f"#{tracker_id}  {int(speed)} km/h")
# #             label_colors.append(sv.Color.BLUE if class_id == 2 else sv.Color.RED if class_id == 7 else sv.Color.WHITE)

# #         annotated_frame = frame.copy()
# #         annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)
# #         annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
        
# #         for label, color in zip(labels, label_colors):
# #             annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)
        
# #         video_writer.write(annotated_frame)

# #     video_writer.release()
# #     print(f"Video saved at {output_path}")








































# #LEGACY CODE

# import argparse
# import cv2
# from ultralytics import YOLO
# import supervision as sv
# import numpy as np
# from collections import defaultdict, deque
# import os

# # Function to parse command-line arguments
# def parse_arguments() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Vehicle speed estimation using YOLOv8 and Supervision"
#     )
#     parser.add_argument(
#         "--source_video_path",
#         default="C:/Users/User/Desktop/vid/highway.mp4",
#         help="Path to the source video file",
#         type=str
#     )
#     return parser.parse_args()

# # Set up the source and target for perspective transform
# # SOURCE = np.array([[380, 150], [580, 150], [1810, 658], [-620, 658]])  # Adjust these points based on video resolution
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

#     # Load YOLOv8 model
#     model = YOLO("yolov8m.pt")

#     # Initialize ByteTrack for tracking
#     byte_track = sv.ByteTrack(frame_rate=video_info.fps)

#     # Set manual thickness and text scale (reduce thickness to avoid hiding vehicles)
#     thickness = 1  # Set to 1 for thinner bounding boxes
#     text_scale = 0.8  # Increased text scale for better visibility

#     # Create annotators for bounding boxes and labels
#     bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)

#     # Create the label annotator
#     label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

#     # Process video frames
#     frame_generator = sv.get_video_frames_generator(args.source_video_path)

# # Define a polygon zone (optional)
# polygon_zone = sv.PolygonZone(SOURCE)  # Adjusted to match the latest library version
# view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

# # Initialize the dictionary to store coordinates with deque
# coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

# # Vehicle class IDs (from the COCO dataset)
# VEHICLE_CLASS_IDS = [2, 5, 7]  # Car: 2, Bus: 5, Truck: 7

# # Prepare video writer
# output_path = os.path.expanduser("C:/Users/User/Desktop/vidi.mp4")  # Save video to desktop
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# fps = video_info.fps
# frame_size = (video_info.resolution_wh[0], video_info.resolution_wh[1])  # Ensure correct order: (width, height)
# video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

# # Check if video writer is opened correctly
# if not video_writer.isOpened():
#     print("Error: VideoWriter not opened!")

# # Dictionary to keep track of labels by tracker ID
# label_dict = {}

# for frame in frame_generator:
#     # Run YOLO detection on the current frame with a lower confidence threshold
#     result = model(frame, conf=0.3)[0]

#     # Convert YOLO results to detections
#     detections = sv.Detections.from_ultralytics(result)

#     # Filter for vehicles (car, bus, truck) based on COCO class IDs
#     detections = detections[(detections.class_id == 2) | (detections.class_id == 5) | (detections.class_id == 7)]

#     # Filter detections inside the polygon zone
#     detections = detections[polygon_zone.trigger(detections)]

#     # Track the detections
#     tracked_detections = byte_track.update_with_detections(detections=detections)

#     # Print tracker ID to check if they are being assigned correctly
#     print(f"Tracker IDs: {tracked_detections.tracker_id}")

#     # Create labels for each vehicle detection (showing tracker ID)
#     points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
#     points = view_transformer.transform_points(points=points).astype(int)

#     labels = []
#     label_colors = []  # List to store colors corresponding to each label
#     for tracker_id, [_, y], class_id in zip(tracked_detections.tracker_id, points, detections.class_id):
#         coordinates[tracker_id].append(y)
#         if len(coordinates[tracker_id]) < video_info.fps / 2:
#             labels.append(f"#{tracker_id}")
#         else:
#             coordinate_start = coordinates[tracker_id][-1]
#             coordinate_end = coordinates[tracker_id][0]
#             distance = abs(coordinate_start - coordinate_end)
#             time = len(coordinates[tracker_id]) / video_info.fps
#             speed = distance / time * 3.6
#             labels.append(f"#{tracker_id}  {int(speed)} km/h")

#         # Set color based on vehicle class (blue for cars, red for trucks)
#         if class_id == 2:  # Car
#             label_colors.append(sv.Color.BLUE)
#         elif class_id == 7:  # Truck
#             label_colors.append(sv.Color.RED)
#         else:
#             label_colors.append(sv.Color.WHITE)  # Default to white for other vehicles

#     # Annotate the frame with bounding boxes and polygon zone
#     annotated_frame = frame.copy()
#     annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)

#     # Annotate bounding boxes and labels on the frame
#     annotated_frame = bounding_box_annotator.annotate(
#         scene=annotated_frame, detections=tracked_detections
#     )

#     # Annotate tracker ID labels with different colors (blue for cars, red for trucks)
#     for label, color in zip(labels, label_colors):
#         annotated_frame = label_annotator.annotate(
#             scene=annotated_frame, detections=tracked_detections, labels=labels
#         )

#     # Write the annotated frame to the output video
#     video_writer.write(annotated_frame)
#     print(f"Writing frame to video...")  # Print a message for each frame written

# # Release resources
# video_writer.release()
# print(f"Video saved at {output_path}")



# # Track the detections
# tracked_detections = byte_track.update_with_detections(detections=detections)

# # Print tracker ID to check if they are being assigned correctly
# print(f"Tracker IDs: {tracked_detections.tracker_id}")

# # Create labels for each vehicle detection (showing tracker ID)
# points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
# points = view_transformer.transform_points(points=points).astype(int)

# labels = []
# label_colors = []  # List to store colors corresponding to each label
# for tracker_id, [_, y], class_id in zip(tracked_detections.tracker_id, points, detections.class_id):
#     coordinates[tracker_id].append(y)
#     if len(coordinates[tracker_id]) < video_info.fps / 2:
#         labels.append(f"#{tracker_id}")
#     else:
#         coordinate_start = coordinates[tracker_id][-1]
#         coordinate_end = coordinates[tracker_id][0]
#         distance = abs(coordinate_start - coordinate_end)
#         time = len(coordinates[tracker_id]) / video_info.fps
#         speed = distance / time * 3.6
#         labels.append(f"#{tracker_id}  {int(speed)} km/h")

#     # Set color based on vehicle class (blue for cars, red for trucks)
#     if class_id == 2:  # Car
#         label_colors.append(sv.Color.BLUE)
#     elif class_id == 7:  # Truck
#         label_colors.append(sv.Color.RED)
#     else:
#         label_colors.append(sv.Color.WHITE)  # Default to white for other vehicles

# # Annotate the frame with bounding boxes and polygon zone
# annotated_frame = frame.copy()
# annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)

# # Annotate bounding boxes and labels on the frame
# annotated_frame = bounding_box_annotator.annotate(
#     scene=annotated_frame, detections=tracked_detections
# )

# # Annotate tracker ID labels with different colors (blue for cars, red for trucks)
# for label, color in zip(labels, label_colors):
#     annotated_frame = label_annotator.annotate(
#         scene=annotated_frame, detections=tracked_detections, labels=labels
#     )

# # Write the annotated frame to the output video
# video_writer.write(annotated_frame)
# print(f"Writing frame to video...")  # Print a message for each frame written

# # Release resources
# video_writer.release()
# print(f"Video saved at {output_path}")



















































# import argparse
# import cv2
# from ultralytics import YOLO
# import supervision as sv
# import numpy as np
# from collections import defaultdict, deque
# import os
# import boto3

# # Function to send SNS notification
# def send_sns_notification(speed, license_plate):
#     sns_client = boto3.client('sns', region_name='your-region')  # E.g., 'us-west-2'
#     topic_arn = 'arn:aws:sns:your-region:your-account-id:VehicleAlert'  # Replace with your SNS topic ARN

#     message = f"Speed: {speed} km/h, License Plate: {license_plate}"
#     response = sns_client.publish(
#         TopicArn=topic_arn,
#         Message=message,
#         Subject='Vehicle Speed Alert'
#     )
#     print("Notification sent:", response)

# # Function to parse command-line arguments
# def parse_arguments() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Vehicle speed estimation using YOLOv8 and Supervision"
#     )
#     parser.add_argument(
#         "--source_video_path",
#         default="C:/Users/User/Desktop/vid/highway.mp4",
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

#     # Load YOLOv8 model
#     model = YOLO("yolov8m.pt")

#     # Initialize ByteTrack for tracking
#     byte_track = sv.ByteTrack(frame_rate=video_info.fps)

#     # Set manual thickness and text scale (reduce thickness to avoid hiding vehicles)
#     thickness = 1  # Set to 1 for thinner bounding boxes
#     text_scale = 0.8  # Increased text scale for better visibility

#     # Create annotators for bounding boxes and labels
#     bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)

#     # Create the label annotator
#     label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

#     # Process video frames
#     frame_generator = sv.get_video_frames_generator(args.source_video_path)

#     # Define a polygon zone (optional)
#     polygon_zone = sv.PolygonZone(SOURCE)  # Adjusted to match the latest library version
#     view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

#     # Initialize the dictionary to store coordinates with deque
#     coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

#     # Vehicle class IDs (from the COCO dataset)
#     VEHICLE_CLASS_IDS = [2, 5, 7]  # Car: 2, Bus: 5, Truck: 7

#     # Prepare video writer
#     output_path = os.path.expanduser("C:/Users/User/Desktop/vidi.mp4")  # Save video to desktop
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     fps = video_info.fps
#     frame_size = (video_info.resolution_wh[0], video_info.resolution_wh[1])  # Ensure correct order: (width, height)
#     video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

#     # Check if video writer is opened correctly
#     if not video_writer.isOpened():
#         print("Error: VideoWriter not opened!")

#     # Dictionary to keep track of labels by tracker ID
#     label_dict = {}

#     for frame in frame_generator:
#         # Run YOLO detection on the current frame with a lower confidence threshold
#         result = model(frame, conf=0.3)[0]

#         # Convert YOLO results to detections
#         detections = sv.Detections.from_ultralytics(result)

#         # Filter for vehicles (car, bus, truck) based on COCO class IDs
#         detections = detections[(detections.class_id == 2) | (detections.class_id == 5) | (detections.class_id == 7)]

#         # Filter detections inside the polygon zone
#         detections = detections[polygon_zone.trigger(detections)]

#         # Track the detections
#         tracked_detections = byte_track.update_with_detections(detections=detections)

#         # Print tracker ID to check if they are being assigned correctly
#         print(f"Tracker IDs: {tracked_detections.tracker_id}")

#         # Create labels for each vehicle detection (showing tracker ID)
#         points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
#         points = view_transformer.transform_points(points=points).astype(int)

#         labels = []
#         label_colors = []  # List to store colors corresponding to each label
#         for tracker_id, [_, y], class_id in zip(tracked_detections.tracker_id, points, detections.class_id):
#             coordinates[tracker_id].append(y)
#             if len(coordinates[tracker_id]) < video_info.fps / 2:
#                 labels.append(f"#{tracker_id}")
#             else:
#                 coordinate_start = coordinates[tracker_id][-1]
#                 coordinate_end = coordinates[tracker_id][0]
#                 distance = abs(coordinate_start - coordinate_end)
#                 time = len(coordinates[tracker_id]) / video_info.fps
#                 speed = distance / time * 3.6
#                 labels.append(f"#{tracker_id}  {int(speed)} km/h")

#                 # **Send SNS notification if speed exceeds the threshold**
#                 if speed > 80:  # Assuming speed threshold is 80 km/h
#                     license_plate = "Unknown"  # You need to add your license plate detection here
#                     send_sns_notification(speed, license_plate)

#             # Set color based on vehicle class (blue for cars, red for trucks)
#             if class_id == 2:  # Car
#                 label_colors.append(sv.Color.BLUE)
#             elif class_id == 7:  # Truck
#                 label_colors.append(sv.Color.RED)
#             else:
#                 label_colors.append(sv.Color.WHITE)  # Default to white for other vehicles

#         # Annotate the frame with bounding boxes and polygon zone
#         annotated_frame = frame.copy()
#         annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)

#         # Annotate bounding boxes and labels on the frame
#         annotated_frame = bounding_box_annotator.annotate(
#             scene=annotated_frame, detections=tracked_detections
#         )

#         # Annotate tracker ID labels with different colors (blue for cars, red for trucks)
#         for label, color in zip(labels, label_colors):
#             annotated_frame = label_annotator.annotate(
#                 scene=annotated_frame, detections=tracked_detections, labels=labels
#             )

#         # Write the annotated frame to the output video
#         video_writer.write(annotated_frame)
#         print(f"Writing frame to video...")  # Print a message for each frame written

#     # Release resources
#     video_writer.release()
#     print(f"Video saved at {output_path}")

































import argparse
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from collections import defaultdict, deque
import os
import easyocr  # For OCR to read the number plates

# Function to parse command-line arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle speed estimation with number plate detection using YOLOv8 and Supervision"
    )
    parser.add_argument(
        "--source_video_path",
        default="C:/Users/User/Desktop/vid/highway.mp4",
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
    vehicle_model = YOLO("yolov8m.pt")

    # Load another YOLOv8 or any model trained for number plate detection
    number_plate_model = YOLO("yolov8n-license-plate.pt")  # Example model for number plates

    # Initialize ByteTrack for tracking
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    # Set manual thickness and text scale
    thickness = 1
    text_scale = 0.8

    # Create annotators for bounding boxes and labels
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    # Initialize OCR reader
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
    output_path = os.path.expanduser("C:/Users/User/Desktop/vidi_with_number_plate.mp4")  
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = video_info.fps
    frame_size = (video_info.resolution_wh[0], video_info.resolution_wh[1])
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Check if video writer is opened correctly
    if not video_writer.isOpened():
        print("Error: VideoWriter not opened!")

    # Dictionary to keep track of labels by tracker ID
    label_dict = {}

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

        # Process number plate detection within vehicle bounding boxes
        for box in tracked_detections.xyxy:
            x1, y1, x2, y2 = map(int, box)
            vehicle_crop = frame[y1:y2, x1:x2]

            # Run number plate detection on cropped vehicle frame
            plate_result = number_plate_model(vehicle_crop, conf=0.3)[0]
            plate_detections = sv.Detections.from_ultralytics(plate_result)

            # If a number plate is detected
            for plate_box in plate_detections.xyxy:
                px1, py1, px2, py2 = map(int, plate_box)
                number_plate_crop = vehicle_crop[py1:py2, px1:px2]

                # Use OCR to read the number plate
                ocr_results = ocr_reader.readtext(number_plate_crop)

                # Get the number plate text
                for res in ocr_results:
                    print(f"Detected number plate: {res[1]}")  # Log detected number plate

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


