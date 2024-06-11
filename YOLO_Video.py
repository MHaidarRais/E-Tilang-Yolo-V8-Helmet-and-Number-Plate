from flask import Flask, request, render_template
from ultralytics import YOLO
from datetime import datetime
import cv2
import math
import os
import easyocr

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    camera = request.form['camera']
    video_detection(int(camera))
    return "Detection started"

def detect_and_print_text(image, detections, threshold=0.2):
    for bbox, text, score in detections:
        if score > threshold:
            # Print the detected text to the terminal
            print(f"Detected text: {text}")

def video_detection(camera_index):
    cap = cv2.VideoCapture(camera_index)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    saveimgdir = 'ViolationCaptured'
    if not os.path.exists(saveimgdir):
        os.mkdir(saveimgdir)

    model = YOLO("D:\\Code And Stuff\\TA CODE THINGY\\WEBAPP-DETECTION\\best.pt")
    classNames = ['number plate', 'rider', 'with helmet', 'without helmet']
    
    reader = easyocr.Reader(['en'], gpu=False)
    
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
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                color = (51, 255, 0) if class_name == 'with helmet' else (0, 0, 255) if class_name == 'without helmet' else (0, 232, 252) if class_name == 'number plate' else (255, 87, 51)
                
                if conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                    
                    if class_name == "without helmet":
                        # Date format naming system
                        now = datetime.now()
                        current_time = now.strftime("%d-%m-%Y %H-%M-%S")
                        filename = f"Violation - {current_time}.jpg"

                        # Bounding box inside the saved file
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)
                        cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                        cv2.imwrite(os.path.join(saveimgdir, filename), img=frame)

                        #Path to the latest saved image file
                        image_path = os.path.join(saveimgdir, filename)

                        # Read the image
                        img = cv2.imread(image_path)

                        # Check if the image was successfully loaded
                        if img is None:
                            raise ValueError("Error loading the image. Please check the file path.") 

                        # Perform text detection
                        text_detections = reader.readtext(img)
                        threshold = 0.2

                        # Detect and print text
                        detect_and_print_text(img, text_detections, threshold)

        yield img

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)
