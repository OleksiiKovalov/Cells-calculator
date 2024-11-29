pip install -r  .\requirements.txt
from model.tracker import Tracker

tracker = Tracker("model/yolov8n-seg.pt", 512)

tracker.track("test_images")
