from flask import Flask, request, render_template
from ultralytics import YOLO
from datetime import datetime
import cv2
import math
import os
import easyocr
import requests
import json
from firebaselib import db, TaskList, TaskListById, Task, storage

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    camera = request.form['camera']
    video_detection(int(camera))
    return "Detection started"

def detect_and_save_text(image, bbox, timestamp):
    x1, y1, x2, y2 = bbox
    cropped_img = image[y1:y2, x1:x2]
    reader = easyocr.Reader(['en'], gpu=False)
    text_detections = reader.readtext(cropped_img)
    
    detected_texts = [text[1] for text in text_detections]
    number_plate_text = detected_texts[0] if detected_texts else 'undetected'

    if not os.path.exists('NumberPlateCaptured'):
        os.makedirs('NumberPlateCaptured')

    number_plate_filename = f"numberplate_{timestamp}.jpg"
    cv2.imwrite(os.path.join('NumberPlateCaptured', number_plate_filename), cropped_img)

    return number_plate_text, number_plate_filename

def locationCoordinates():
    try:
        response = requests.get('https://ipinfo.io')
        data = response.json()
        loc = data['loc'].split(',')
        lat, long = float(loc[0]), float(loc[1])
        city = data.get('city', 'Unknown')
        state = data.get('region', 'Unknown')
        return lat, long, city, state
    except:
        print("Internet Not available")
        exit()
        return False

def save_violation_data_to_firestore(date_info, time_info, location, latitude, longitude):
    task = Task(
        day=date_info['day'],
        month=date_info['month'],
        year=date_info['year'],
        hour=time_info['hour'],
        minute=time_info['minute'],
        second=time_info['second'],
        city=location.split(", ")[0],
        state=location.split(", ")[1],
        lat=latitude,
        long=longitude
    )
    db.collection('ETLE').add(task.to_dict())

def save_violation_data(image_path, date_info, time_info, location, latitude, longitude):
    violation_data = {
        "image": image_path,
        "date": date_info,
        "time": time_info,
        "location": location,
        "latitude": latitude,
        "longitude": longitude
    }
    
    if os.path.exists("violation.json"):
        with open("violation.json", "r") as file:
            data = json.load(file)
        data.append(violation_data)
    else:
        data = [violation_data]
    
    with open("violation.json", "w") as file:
        json.dump(data, file, indent=4)

def is_inside_or_near(bbox1, bbox2, margin=20):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    if (x1_1 - margin < x1_2 < x2_1 + margin or x1_1 - margin < x2_2 < x2_1 + margin) and \
       (y1_1 - margin < y1_2 < y2_1 + margin or y1_1 - margin < y2_2 < y2_1 + margin):
        return True
    return False

def video_detection(camera_index):
    cap = cv2.VideoCapture(camera_index)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    saveimgdir = 'ViolationCaptured'
    if not os.path.exists(saveimgdir):
        os.mkdir(saveimgdir)

    model = YOLO("D:\\Code And Stuff\\TA CODE THINGY\\WEBAPP-DETECTION\\YOLOV8N-V9.pt")
    classNames = ['number plate', 'rider', 'with helmet', 'without helmet']
    
    reader = easyocr.Reader(['en'], gpu=False)
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        results = model(img, stream=True)
        rider_bbox = None
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
                    
                    if class_name == "rider":
                        rider_bbox = [x1, y1, x2, y2]
                    
                    if class_name == "without helmet" and rider_bbox:
                        without_helmet_bbox = [x1, y1, x2, y2]
                        
                        if is_inside_or_near(rider_bbox, without_helmet_bbox):
                            now = datetime.now()
                            lat, long, city, state = locationCoordinates()
                            timestamp = now.strftime("%d-%m-%Y %H-%M-%S")
                            filename = f"{timestamp}.jpg"
                            
                            date_info = {
                                "day": now.day,
                                "month": now.month,
                                "year": now.year
                            }
                            
                            time_info = {
                                "hour": now.hour,
                                "minute": now.minute,
                                "second": now.second
                            }
                            
                            for box2 in boxes:
                                if classNames[int(box2.cls[0])] == 'number plate':
                                    number_plate_bbox = [int(box2.xyxy[0][0]), int(box2.xyxy[0][1]), int(box2.xyxy[0][2]), int(box2.xyxy[0][3])]
                                    break

                            if number_plate_bbox:
                                number_plate_text, number_plate_filename = detect_and_save_text(img, number_plate_bbox, timestamp)
                            else:
                                number_plate_text = 'undetected'
                                number_plate_filename = None

                            save_violation_data_to_firestore(date_info, time_info, f"{city}, {state}", lat, long, number_plate_text)

                            cv2.imwrite(os.path.join(saveimgdir, filename), img=img)
                            storage_path = f"ViolationCaptured/{filename}"
                            storage.child(storage_path).put(os.path.join(saveimgdir, filename))

                            if number_plate_filename:
                                NPstorage_path = f"NumberPlateCaptured/{number_plate_filename}"
                                storage.child(NPstorage_path).put(os.path.join('NumberPlateCaptured', number_plate_filename))

        yield img

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)
