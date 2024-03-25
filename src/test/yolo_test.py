from ultralytics import YOLO

#El modelo se puede descargar de https://huggingface.co/turhancan97/yolov8-segment-trash-detection/blob/main/yolov8m-seg.pt, y metemos el nombre del fichero .pt
model = YOLO('yolov8m-seg.pt')

#Hacemos inferencia sobre la imagen de prueba y con guardado, que crea una carpeta runs
prediction = model.predict("prueba.jpg", imgsz=640, show=False, save=True)
