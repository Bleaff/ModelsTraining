from ultralytics import YOLO
import argparse
# Load the model.
model = YOLO('/home/bleaf/train_people/yolo_model/yolov8m.pt')
 
# Training.
results = model.train(
   data='/mnt/datasets/bleaf/Datasets/merged_dataset_people/meta.yaml',
   imgsz=640,
   epochs=50,
   batch=4,
   name='yolov8m_v8_50e_augmented',
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
