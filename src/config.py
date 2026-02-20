# ================= Model Config =================
MODEL_PATH = "models/best-int8.tflite"
INPUT_WIDTH = 320
INPUT_HEIGHT = 320
CONF_THRESHOLD = 0.6
NMS_THRESHOLD = 0.4

# ================= Camera Config =================
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 320

# ================= Logging Config =================
LOG_COOLDOWN = 5
LOG_FILE = "output/detections.csv"
SNAPSHOT_DIR = "output/snapshots"

# ================= Performance Config =================
NUM_THREADS = 4