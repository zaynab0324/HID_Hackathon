"""Download the offline face models (run once while online):  python get_models.py"""
import os
import urllib.request

BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
FILES = {
    "face_detection_yunet_2023mar.onnx": BASE + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": BASE + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(folder, exist_ok=True)
for name, url in FILES.items():
    path = os.path.join(folder, name)
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        print("already have", name)
        continue
    print("downloading", name, "...")
    urllib.request.urlretrieve(url, path)
    print("  ok", os.path.getsize(path) // 1024, "KB")
print("Offline face models ready.")
