import os
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from matplotlib import pyplot as plt





model_path = "person_detection.pt"
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s') ##for yolov5 pretrained weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)



## For Video Stream--------------

# def detect_persons_in_video(video_path, save_dir):
#     """
#     Process an video with YOLOv5 object detection.
#     Model Is Loaded globally
    
#     Parameters:
#     - video_path (str): path of video.
#     - save_path (str): Directory path where detected frames will be saved.
    
#     """
    
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print("Error: Couldn't open video.")
#         return

#     frame_count = 0
    

#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_interval = int(1 / fps)

#     while True:
#         person_count = 0
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Can't receive frame (stream end?). Exiting...")
#             break

#         annotated_frame = frame.copy()
#         results = model(frame)
#         result = results.pandas().xyxy[0]

        
#         for _, row in result.iterrows():
#             xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
#             class_name = model.names[int(row['class'])]  
#             if class_name=="person":
#                 person_count+=1
#                 cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
#                 cv2.putText(annotated_frame, f'{class_name}', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
#         cv2.putText(annotated_frame, f'Total Person: {person_count}', (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
#         frame_name = os.path.join(save_dir, f'frame_{frame_count}.jpg')
#         cv2.imwrite(frame_name, annotated_frame)

#         frame_count += 1
#         person_count = 0
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * fps)
#         # cv2.imshow('video Stream with Detection', annotated_frame)
#     #     if cv2.waitKey(1) & 0xFF == ord('q'):
#     #         break

#     # cap.release()
#     # cv2.destroyAllWindows()




# if __name__ == "__main__":
#     dest_dir = "results_vid"
#     if not os.path.exists(dest_dir): 
#         os.makedirs(dest_dir, exist_ok=True)
#     Vid_path = "data\persons.mp4"
#     detect_persons_in_video(Vid_path, dest_dir)






##For RTSP url--------------------
# def process_rtsp_stream(rtsp_url, save_dir):
#     """
#     Process an RTSP stream with YOLOv5 object detection.
#     Model Is Loaded globally
    
#     Parameters:
#     - rtsp_url (str): RTSP stream URL.
#     - save_path (str): Directory path where detected frames will be saved.
    
#     """
#     cap = cv2.VideoCapture(rtsp_url, save_dir)

#     if not cap.isOpened():
#         print("Error: Couldn't open Camera.")
#         return

#     frame_count = 0
#     person_count = 0

#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_interval = int(1 / fps)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Can't receive frame (stream end?). Exiting...")
#             break

#         annotated_frame = frame.copy()
#         result = results.pandas().xyxy[0]

        
#         for _, row in result.iterrows():
#             xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
#             class_name = model.names[int(row['class'])]  
            
#             if class_name=="person":
#                 person_count+=1
#                 cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 0, 0), 2)
#                 cv2.putText(annotated_frame, f'{class_name}', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
#         cv2.putText(annotated_frame, f'Total Person: {person_count}', (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
#         frame_name = os.path.join(save_dir, f'frame_{frame_count}.jpg')
#         cv2.imwrite(frame_name, annotated_frame)

#         cv2.imshow('RTSP Stream with Detection', annotated_frame)

#         frame_count += 1
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count * fps)

#         # cv2.imshow('RTSP Stream with Detection', annotated_frame)   
#     #     if cv2.waitKey(1) & 0xFF == ord('q'):
#     #         break

#     # cap.release()
#     # cv2.destroyAllWindows()


# if __name__ == "__main__":
#     dest_dir = "results_rtsp"
#     if not os.path.exists(dest_dir): 
#         os.makedirs(dest_dir, exist_ok=True)
#     rtsp_url = "rtsp://admin:rmt@2022@192.168.0.216" #'rtsp://username:password@camera_ip_address/stream' {Format of rtsp url to be given}
#     process_rtsp_stream(rtsp_url, dest_dir)

    






## For Image Processing------------------------
def detection_first_phase(source_dir, dest_dir):
    """
    Process directory of images with YOLOv5 object detection.
    Model Is Loaded globally
    
    Parameters:
    - source_dir (str): path of image directory.
    - save_path (str): Directory path where detected frames will be saved.
    
    """

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for image_file in os.listdir(source_dir):
        if image_file.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(source_dir, image_file)
            img = cv2.imread(image_path)

            results = model(img)
            result = results.pandas().xyxy[0]
            
            person_count = 0
            for _, row in result.iterrows():
                xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                class_name = model.names[int(row['class'])]
                
                if class_name == "person":
                    person_count += 1
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 0), 2)
                    cv2.putText(img, f'{class_name}', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            cv2.putText(img, f'Total Person: {person_count}', (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            output_path = os.path.join(dest_dir, image_file)
            cv2.imwrite(output_path, img)


if __name__ == '__main__':
    Source_Directory = "data"
    Destination_directory = "results_img"
    
    detection_first_phase(Source_Directory, Destination_directory)

