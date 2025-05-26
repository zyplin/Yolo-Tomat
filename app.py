from http.client import REQUEST_ENTITY_TOO_LARGE
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, send_file
import cv2
import os
import time
import csv
from collections import Counter
from ultralytics import YOLO
from itertools import zip_longest

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CSV_EXPORT'] = 'exported.csv'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_PATH = "my_model.pt"
MODEL_URL = "https://drive.google.com/file/d/1BjexTUhxkNMg7-DO8I75LQjBBobH2Lv3/view?usp=drive_link"

if not os.path.exists(MODEL_PATH):
    print("📥 Mengunduh model...")
    r = REQUEST_ENTITY_TOO_LARGE.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

model = YOLO(MODEL_PATH)

camera = None
camera_active = False
latest_uploaded_result = None
latest_stats = {}

label_descriptions = {
    "Mentah": "Tomat masih hijau. Belum cocok untuk dikonsumsi langsung.",
    "Setengah Matang": "Tomat mulai jingga. Cocok disimpan sebentar atau dimasak.",
    "Matang": "Tomat merah cerah dan empuk. Siap dikonsumsi atau dijual."
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deteksi')
def deteksi():
    detected_labels = request.args.get('detected_labels')
    descriptions = request.args.get('descriptions')
    result = request.args.get('latest_result')

    zipped_data = list(zip(
        detected_labels.split(',') if detected_labels else [],
        descriptions.split('||') if descriptions else []
    ))

    return render_template('deteksi.html',
                           camera_active=camera_active,
                           latest_result=result,
                           zipped_data=zipped_data,
                           stats=latest_stats)

@app.route('/start_camera')
def start_camera():
    global camera, camera_active
    if not camera_active:
        camera = cv2.VideoCapture(0)
        camera_active = True
    return redirect(url_for('deteksi'))

@app.route('/stop_camera')
def stop_camera():
    global camera, camera_active
    camera_active = False
    if camera:
        camera.release()
    return redirect(url_for('deteksi'))

def generate_frames():
    global camera
    while camera_active and camera is not None and camera.isOpened():
        success, frame = camera.read()
        if not success:
            break
        results = model(frame)
        annotated = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    global latest_uploaded_result, latest_stats

    file = request.files['image']
    if not file or file.filename == '':
        return redirect(url_for('deteksi'))

    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    results = model(path)
    img = cv2.imread(path)

    predicted_classes = results[0].boxes.cls.cpu().numpy()
    names = results[0].names
    labels = [names[int(cls)] for cls in predicted_classes]
    detected_labels = list(set(labels))
    label_descriptions_found = [label_descriptions.get(label, "Tidak ada deskripsi.") for label in detected_labels]
    latest_stats = dict(Counter(labels))

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{names[cls_id]} {conf:.2f}"
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    result_name = 'result_' + file.filename
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_name)
    cv2.imwrite(result_path, img)
    latest_uploaded_result = result_name

    with open(app.config['CSV_EXPORT'], 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Label', 'Deskripsi'])
        for label, desc in zip(detected_labels, label_descriptions_found):
            writer.writerow([label, desc])

    return redirect(url_for('deteksi',
                            detected_labels=','.join(detected_labels),
                            descriptions='||'.join(label_descriptions_found),
                            latest_result=result_name))

@app.route('/download_csv')
def download_csv():
    return send_file(app.config['CSV_EXPORT'], as_attachment=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/info_tomat')
def info_tomat():
    return render_template('info_tomat.html')

@app.route('/tentang')
def tentang():
    return render_template('tentang.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

