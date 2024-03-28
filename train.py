from ultralytics import YOLO
 
# Load the model.
model = YOLO('/home/bleaf/train_people/yolo_model/yolov8m.pt')
 
# Training.
results = model.train(
   data='/home/bleaf/train_people/merged_dataset_people/meta.yaml',
   imgsz=640,
   epochs=50,
   batch=16,
   name='yolov8n_v8_50e'
)
