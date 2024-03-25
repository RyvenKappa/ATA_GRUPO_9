from ultralytics import YOLO

"""
    Convierte de modelo pytorch a modelo onnx
"""

model = YOLO('yolov8m-seg.pt')
model.export(format='onnx')
