from ultralytics import YOLO

model = YOLO('yolov8m-seg.onnx')

validation_results = model.val(data='data.yaml',imgsz=416,batch=16)
validation_results.box.map    # map50-95
validation_results.box.map50  # map50
validation_results.box.map75  # map75
validation_results.box.maps   # a list contains map50-95 of each category