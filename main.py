import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time


st.set_page_config(page_title="Lost & Found Monitoring", layout="wide")
st.title("🎒 Lost & Found Monitoring")
st.markdown("""
Detect **backpack, handbag, suitcase** only when **no person is interacting with it**.  
Street scenario friendly: pedestrians walking by are ignored.
""")


ALONE_TIME = 5            # seconds a bag must stay unattended
IOU_THRESHOLD = 0.08       # only consider person really near the bag
PERSON_CLASS = 0
TARGET_CLASSES = [24, 26, 28]
PERSON_SIZE_THRESHOLD = 1000  # ignore very small people far away

model = YOLO("yolov8s.pt")  


def iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def detect_image(frame):
    """Detect bags without tracking (for image upload)."""
    results = model(frame, conf=0.3)[0]  # lower confidence for small objects
    det_frame = frame.copy()
    persons, boxes = [], []

    if results.boxes is not None:
        xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes.xyxy is not None else []
        cls_list = results.boxes.cls.cpu().numpy() if results.boxes.cls is not None else []

        for box, cls in zip(xyxy, cls_list):
            cls = int(cls)
            x1, y1, x2, y2 = [int(v) for v in box]
            if cls == PERSON_CLASS:
                persons.append((x1, y1, x2, y2))
            elif cls in TARGET_CLASSES:
                boxes.append((x1, y1, x2, y2, results.names[cls]))

    for x1, y1, x2, y2, name in boxes:
        overlap = False
        for px1, py1, px2, py2 in persons:
            area = (px2 - px1) * (py2 - py1)
            if area < PERSON_SIZE_THRESHOLD:
                continue
            if iou((x1, y1, x2, y2), (px1, py1, px2, py2)) > IOU_THRESHOLD:
                overlap = True
                break
        if not overlap:
            cv2.rectangle(det_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(det_frame, f"{name}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return det_frame


mode = st.sidebar.selectbox(
    "Select Mode",
    ("Choose an option", "Real-time Camera", "Image Upload")
)


if mode == "Real-time Camera":
    st.write(f"Press **Start Camera**. Bags unattended > {ALONE_TIME}s will be marked.")
    start = st.checkbox("Start Camera")
    frame_window = st.image([], channels="RGB")
    alone_tracker = {}  

    if start:
        cap = cv2.VideoCapture(0)
        while start:
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera not found.")
                break

            results = model.track(frame, persist=True, verbose=False, conf=0.3)[0]

            persons, bags = [], []

            if results.boxes is not None:
                xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes.xyxy is not None else []
                cls_list = results.boxes.cls.cpu().numpy() if results.boxes.cls is not None else []
                id_list = results.boxes.id.cpu().numpy() if results.boxes.id is not None else []

                for box, cls, tid in zip(xyxy, cls_list, id_list):
                    cls, tid = int(cls), int(tid)
                    if cls == PERSON_CLASS:
                        persons.append(box)
                    elif cls in TARGET_CLASSES:
                        bags.append((tid, box, cls))

            for bag_id, bag_box, cls in bags:
                bx1, by1, bx2, by2 = bag_box
                alone = True
                for p_box in persons:
                    px1, py1, px2, py2 = p_box
                    area = (px2 - px1) * (py2 - py1)
                    if area < PERSON_SIZE_THRESHOLD:
                        continue
                    if iou(bag_box, p_box) > IOU_THRESHOLD:
                        alone = False
                        break

                label = results.names[cls]
                color = (0, 255, 0)

                if alone:
                    if bag_id not in alone_tracker:
                        alone_tracker[bag_id] = time.time()
                        color = (0, 165, 255)
                        cv2.putText(frame, f"ALONE {label}", (int(bx1), int(by1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    else:
                        elapsed = time.time() - alone_tracker[bag_id]
                        if elapsed > ALONE_TIME:
                            color = (0, 0, 255)
                            cv2.putText(frame, f"UNATTENDED {label}", (int(bx1), int(by1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        else:
                            color = (0, 165, 255)
                            cv2.putText(frame, f"ALONE {label}", (int(bx1), int(by1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                else:
                    alone_tracker.pop(bag_id, None)

                cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), color, 2)

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

elif mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        result = detect_image(frame)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Detection Result")

else:
    st.info("⬅️ Select a mode from the sidebar to begin.")
