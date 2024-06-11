from ultralytics import YOLO
from datetime import datetime
import cv2
import math
import os

def video_detection(path_x):
    video_capture = path_x
    
    # Create a Webcam Object
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    saveimgdir = 'ViolationCaptured'
    if not os.path.exists(saveimgdir):
        os.mkdir(saveimgdir)
    else:
        pass

    model = YOLO("D:\\Code And Stuff\\TA CODE THINGY\\WEBAPP-DETECTION\\best.pt")
    classNames = ['number plate', 'rider', 'with helmet', 'without helmet']
    while True:
        success, img = cap.read()
        if not success:
            break
        
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                check, frame = cap.read()
                if not check:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(x1, y1, x2, y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                if class_name == 'with helmet':
                    color = (51, 255, 0)
                elif class_name == "without helmet":
                    color = (0, 0, 255)
                    now = datetime.now()
                    current_time = now.strftime("%d-%m-%Y %H-%M-%S")
                    filename = f"Violation - {current_time}.jpg"
                    # Draw the bounding box and label on the frame before saving it
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                    cv2.imwrite(os.path.join(saveimgdir, filename), img=frame)
                    print("Processing image...")
                elif class_name == "number plate":
                    color = (0, 232, 252)
                else:
                    color = (255, 87, 51)
                
                if conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        yield img

    cap.release()
    cv2.destroyAllWindows()
