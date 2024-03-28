from ultralytics import YOLO
import argparse
# Load the model.
model = YOLO('/home/bleaf/Desktop/leanring/yolov8-face/runs/detect/yolov8s_v8_50e_augmented3/weights/last.pt')
 
# Training.
results = model.train(
   data='/home/bleaf/train_people/merged_dataset_people/meta.yaml',
   imgsz=640,
   epochs=50,
   batch=8,
   name='yolov8s_v8_50e_augmented',
   augment=True,
   nbs=64,
   hsv_h=0.015,
   hsv_s=0.7,
   hsv_v=0.4, 
   degrees=0.0,
   translate=0.2,
   scale=0.3,
   shear=0.0,
   perspective=0.0,
   flipud=0.0,
   fliplr=0.5,
   mosaic=0.0,
)

