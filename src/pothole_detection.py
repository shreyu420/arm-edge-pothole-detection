import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

# ================= CONFIG =================
MODEL_PATH = "/home/pi/pothole_project/best-int8.tflite"
CONF_THRESHOLD = 0.7
NMS_THRESHOLD = 0.4
LOG_COOLDOWN = 5

LOG_FILE = "output/detections.csv"
SNAPSHOT_DIR = "output/snapshots"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ================= LOAD MODEL =================
interpreter = Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_H = input_details[0]['shape'][1]
INPUT_W = input_details[0]['shape'][2]

print("Model Loaded")
print("Input Shape:", INPUT_W, "x", INPUT_H)

# ================= LOG FILE =================
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Type", "Confidence", "Snapshot"])

last_log_time = 0

# ================= CAMERA =================
picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(main={"size": (480, 320)})
)
picam2.start()

print("Camera Started")

prev_time = 0

# ================= MAIN LOOP =================
while True:
    frame = picam2.capture_array()
    h, w, _ = frame.shape
    scale_x = w / INPUT_W
    scale_y = h / INPUT_H
    # Preprocess
    img = cv2.resize(frame, (INPUT_W, INPUT_H))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img_rgb, axis=0)

    if input_details[0]['dtype'] == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        input_data = input_data.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    detections = output[0]   # (N, 85)

    boxes = []
    scores = []

    # Proper YOLO Confidence Calculation
    for det in detections:
        obj_score = det[4]
        class_scores = det[5:]
        class_id = np.argmax(class_scores)
        class_score = class_scores[class_id]

        confidence = float(obj_score * class_score)

        if confidence > CONF_THRESHOLD and class_id == 0:

            x, y, bw, bh = det[0:4]

            # Convert normalized ? model pixel space
            x_center = x * INPUT_W
            y_center = y * INPUT_H
            box_width = bw * INPUT_W
            box_height = bh * INPUT_H

            # Convert to original frame scale
            x_center *= scale_x
            y_center *= scale_y
            box_width *= scale_x
            box_height *= scale_y

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(confidence)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    # Draw Final Boxes
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            conf = scores[i]

            cv2.rectangle(frame,
                          (x, y),
                          (x+bw, y+bh),
                          (0, 255, 0),
                          2)

            cv2.putText(frame,
                        f"POTHOLE {conf:.2f}",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

            # Logging
            if time.time() - last_log_time > LOG_COOLDOWN:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                snap_name = f"pothole_{timestamp}.jpg"
                snap_path = os.path.join(SNAPSHOT_DIR, snap_name)

                cv2.imwrite(snap_path, frame)

                with open(LOG_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, "pothole", conf, snap_name])

                print("Logged:", snap_name)
                last_log_time = time.time()

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(frame,
                f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)

    cv2.imshow("ARM Edge AI - Pothole Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()
print("Stopped Successfully.")