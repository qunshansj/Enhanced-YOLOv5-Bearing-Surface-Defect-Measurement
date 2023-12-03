python
import detect
import os
import shutil

class YOLOv5v6Detector:
    def __init__(self, model_path, test_folder, temp_folder):
        self.model_path = model_path
        self.test_folder = test_folder
        self.temp_folder = temp_folder

    def detect_objects(self):
        result = detect.det_yolov5v6(self.model_path, self.test_folder, self.temp_folder)
        return result

# Example usage
detector = YOLOv5v6Detector('./best.pt', './test', './Temporary_folder')
result = detector.detect_objects()
