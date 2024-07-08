from flask import Flask, render_template, Response, request, session, jsonify
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
import cv2
from datetime import datetime
import requests
import json
import math
import easyocr
from ultralytics import YOLO
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ayush'
app.config['UPLOAD_FOLDER'] = 'static/files'
socketio = SocketIO(app)

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

def locationCoordinates():
    try:
        response = requests.get('https://ipinfo.io')
        data = response.json()
        loc = data['loc'].split(',')
        lat, long = float(loc[0]), float(loc[1])
        city = data.get('city', 'Unknown')
        state = data.get('region', 'Unknown')
        return lat, long, city, state
    except Exception as e:
        print("Error fetching location:", str(e))
        return None, None, None, None

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

def video_detection(path_x):
    cap = cv2.VideoCapture(path_x)
    if not cap.isOpened():
        print(f"Error opening video file: {path_x}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saveimgdir = 'ViolationCaptured'
    if not os.path.exists(saveimgdir):
        os.mkdir(saveimgdir)

    model = YOLO("D:\\Code And Stuff\\TA CODE THINGY\\WEBAPP-DETECTION\\YOLOV8N-V5.pt")
    classNames = ['number plate', 'rider', 'with helmet', 'without helmet']
    
    reader = easyocr.Reader(['en'], gpu=False)
    frame_count = 0
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
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
                        now = datetime.now()
                        lat, long, city, state = locationCoordinates()
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
                        
                        filename = f"Violation - {now.strftime('%d-%m-%Y %H-%M-%S')}-{city,state}-{lat,long}.jpg"
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                        cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                        cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                        cv2.imwrite(os.path.join(saveimgdir, filename), img=img)

                        image_path = os.path.join(saveimgdir, filename)
                        save_violation_data(image_path, date_info, time_info, f"{city}, {state}", lat, long)

        frame_count += 1
        progress = int((frame_count / total_frames) * 100)
        socketio.emit('progress', {'progress': progress})
        yield img

    cap.release()
    cv2.destroyAllWindows()

def generate_frames(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('indexproject.html')

@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('webcam.html')

@app.route('/FrontPage', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)
        session['video_path'] = file_path
    return render_template('uploadvideo.html', form=form)

@app.route('/video')
def video():
    return Response(generate_frames(path_x=session.get('video_path', None)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webapp')
def webapp():
    return Response(generate_frames_web(path_x=0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/progress')
def progress():
    return render_template('progress.html')

@app.route('/violations')
def violations():
    # Read violations from violation.json
    with open('violation.json', 'r') as file:
        violations = json.load(file)
    return render_template('violationList.html', violations=violations)

@app.route('/violation/<int:index>')
def violation_details(index):
    try:
        with open('violation.json', 'r') as file:
            violations_data = json.load(file)
        
        if index < 0 or index >= len(violations_data):
            return "Invalid violation index."
        
        violation = violations_data[index]
    except FileNotFoundError:
        violation = None
    
    return render_template('violation_details.html', violation=violation)

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

if __name__ == "__main__":
    socketio.run(app, debug=True)
