import os
import cv2
import torch
from ultralytics import YOLO

model_path = "person_detection.pt"
source_image_path = "data/4.jpg"

model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

if __name__ == "__main__":
    img = cv2.imread(source_image_path)
    results = model(img)
    print(results)
    result = results.pandas().xyxy[0]
    person_count = 0
    
    for _, row in result.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        class_name = model.names[int(row['class'])]
        if class_name == "person":
            person_count+=1
 
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(img, f'{class_name}', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        
    cv2.putText(img, f'Person Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow('Object Detection', img)
    
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
