import os
import pathlib
import cv2
import torch

# ✅ Fix: Force Windows-style paths to prevent 'PosixPath' errors
pathlib.PosixPath = pathlib.WindowsPath

# ✅ Load YOLOv5 model from local directory with force_reload to avoid cache issues
model = torch.hub.load(r"D:\Python 3.13.2\yolov5", 'custom',
                       path=r"D:\Python 3.13.2\yolov5\best (2).pt",
                       source='local', force_reload=True)

# ✅ Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Run YOLOv5 model on the frame
    results = model(frame)

    # ✅ Process detections
    for *xyxy, conf, cls in results.xyxy[0]:  # Bounding box coordinates
        x1, y1, x2, y2 = map(int, xyxy)
        conf = float(conf)
        cls = int(cls)

        # ✅ Draw bounding box if confidence is high
        if conf > 0.6:  # Adjust confidence threshold if needed
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"QR_Code ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ✅ Display the frame
    cv2.imshow("YOLOv5 Object Detection", frame)

    # ✅ Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release resources
cap.release()
cv2.destroyAllWindows()
