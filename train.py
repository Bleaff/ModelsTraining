from ultralytics import YOLO
 
# Load the model.
model = YOLO('/home/bleaf/train_people/yolo_model/yolov8m.pt')
 
# Training.
results = model.train(
   data='/home/bleaf/train_people/filtered_dataset/meta.yaml',
   imgsz=640,
   epochs=200,
   batch=16,
   name='yolov8m_200e'
)
