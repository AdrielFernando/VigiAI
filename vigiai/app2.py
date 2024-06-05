from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

def resize_video(frame, width=10, height=20, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = frame.shape[:2]

    if width is None and height is None:
        return frame

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(frame, dim, interpolation=inter)
    return resized

def detect_person(frame):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, rects

def check_crossed_line(current_rects, line_positionX):
    crossed = False
    for (x, y, w, h) in current_rects:
        right_x = x + w

        if x < line_positionX < right_x:
            crossed = True
            break

    return crossed

def generate_frames():
    video_capture = cv2.VideoCapture('testoficial.mp4')
    line_positionX = 100

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = resize_video(frame, width=640)
        frame = cv2.flip(frame, 1)
        frame, current_rects = detect_person(frame)
        cv2.line(frame, (line_positionX, 0), (line_positionX, frame.shape[0]), (0, 0, 255), 2)

        crossed = check_crossed_line(current_rects, line_positionX)

        # Convertendo crossed para um valor que possa ser transmitido no cabeçalho HTTP
        crossed_str = "1" if crossed else "0"

        # Codificar o frame em JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Concatenar o frame e o valor de crossed no cabeçalho HTTP
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n' +
               b'Content-Type: text/plain\r\n\r\n' + crossed_str.encode() + b'\r\n')

    video_capture.release()
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
