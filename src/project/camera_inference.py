import cv2
from ultralytics import YOLO

"""
    Realiza inferencia en tiempo real sobre la cámara
"""
model = YOLO('yolov8m-seg.onnx')
capture = cv2.VideoCapture(0)

ok = False

while(True):
    ok, img = capture.read()

    cv2.imshow('camara',img)
    predictions = model.predict(img, imgsz=416, show=False, save=False)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()