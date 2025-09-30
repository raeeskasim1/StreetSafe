# StreetSafe 🛡️

Detect unattended bags in real-time for street and public monitoring.

StreetSafe uses **YOLOv8** to detect **backpacks, handbags, and suitcases**, alerting when they are left unattended. It is optimized for street scenarios, ignoring pedestrians walking nearby while monitoring bags that may be left unattended.

---

## Features

- **Real-time Camera Detection**: Monitor live video feed for unattended bags.  
- **Image Upload**: Test detection on uploaded images.  
- **Street-Friendly Logic**: Ignores pedestrians walking past; detects bags only when unattended.  
- ** ISOLATED→ UNATTENDED Alerts**: Bags are first marked as **ISOLATED**, and if left for more than N seconds (default 5s), marked as **UNATTENDED**.  
- **Clean Display**: No unnecessary bounding boxes for the monitoring region; only detected objects are highlighted.  

---

## Configuration

- **ISOLATED_TIME**: Time in seconds before a bag is marked as unattended.  
- **IOU_THRESHOLD**: Determines how close a person must be to the bag to reset the timer.  
- **PERSON_SIZE_THRESHOLD**: Ignores very small distant people for street detection.  
- **TARGET_CLASSES**: Backpack, handbag, suitcase.  

---

## Demo Images

| Isolated Detection | Unattended Detection |
|-----------------|--------------------|
| ![Alone](img/2.png) | ![Unattended](img/1.png) |

---

## Tools & Technologies Used

- **Python 3.8+**  
- **Streamlit** – Web interface for real-time monitoring  
- **Ultralytics YOLOv8** – Object detection and tracking  
- **OpenCV** – Video and image processing  
- **NumPy** – Numerical operations  
- **Pillow (PIL)** – Image handling  

---

