import cv2
import numpy as np

# Function to capture mouse click events
points = []

def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point selected: ({x}, {y})")

# Load the video
video_path = "C:/Users/User/Desktop/vid/car.mp4"  # Path to your video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame from the video
ret, frame = cap.read()

if not ret:
    print("Error: Could not read the first frame.")
    exit()

# Create a window and set a mouse callback to select points
cv2.namedWindow("Select Points")
cv2.setMouseCallback("Select Points", select_points)

# Show the frame and wait for user to select 4 points
while len(points) < 4:
    cv2.imshow("Select Points", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit early
        break

cv2.destroyAllWindows()

if len(points) == 4:
    print("Selected points:", points)
    # You can directly use these points as your new `SOURCE` array
    SOURCE = np.array(points)
    print(f"Updated SOURCE array: {SOURCE}")
else:
    print("Error: 4 points were not selected.")
