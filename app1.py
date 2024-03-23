import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import supervision as sv

# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Function to check if a human detection is within a seat detection
def is_within(human_detection, seat_detection, threshold=0.8):
    iou = calculate_iou(human_detection, seat_detection)
    return iou > threshold



# Function to perform object recognition on the selected video file
def perform_object_recognition(uploaded_file):
    # Model initialization
    model = YOLO("best.pt")

    # Get file path from uploaded file object
    video_file_path = uploaded_file.name

    webcam_resolution = [800, 600]
    frame_width, frame_height = webcam_resolution
    cap = cv2.VideoCapture(video_file_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Supervision setup
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )
    ZONE_POLYGON = np.array([
        [0, 0],
        [frame_width, 0],
        [frame_width, frame_height],
        [0, frame_height]
    ])
    zone_polygon = (ZONE_POLYGON * np.array([frame_width, frame_height])).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.red(),
        thickness=2,
        text_color=sv.Color.black(),
        text_thickness=2,
        text_scale=1
    )

    # Object recognition loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection using YOLO
        results = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Extracting human and seat boxes
        seat_boxes = [box for box, _, _, clas, _, _ in detections if clas == 1]
        human_boxes = [box for box, _, _, clas, _, _ in detections if clas == 0]

        # Counting occupied seats
        num_occupied = 0
        for human_box in human_boxes:
            for seat_box in seat_boxes:
                if is_within(human_box, seat_box, 0.1):
                    num_occupied += 1

        # Displaying annotations on the frame
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=[f"#{model.model.names[class_id]} {confidence:0.2f}"
                    for _, _, confidence, class_id, _, _
                    in detections]
        )
        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        # Displaying seat count information on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        frame = cv2.putText(frame, f'Number of Occupied Seats: {num_occupied}', (0, 100), font,
                            fontScale, color, thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, f'Number of Free Seats: {64 - num_occupied}', (0, 150), font,
                            fontScale, color, thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, f'Detected Seats: {sum(detections.class_id)}', (0, 200), font,
                            fontScale, color, thickness, cv2.LINE_AA)

        # Displaying the frame
        st.image(frame, channels="BGR", use_column_width=True)

    cap.release()


# Main function to define the Streamlit app
def main():
    st.title("Railway Seat Occupancy Detection")

    # File uploader widget to select video file
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov"])

    if uploaded_file is not None:
        st.write("Processing video...")
        perform_object_recognition(uploaded_file)

# Run the Streamlit app
if __name__ == "__main__":
    main()
