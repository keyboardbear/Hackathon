import cv2
import numpy as np
import time
from datetime import datetime
import os
import face_recognition
import speech_recognition as sr
import threading
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename
from collections import deque

app = Flask(__name__)

# 전역 변수
cap = None
monitor = None


def gen_frames():
    global cap, monitor
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = monitor.process_frame(frame)

        # 이벤트 클립 처리
        monitor.frame_buffer.append(frame.copy())
        if monitor.event_triggered and len(monitor.future_frames) < monitor.future_frames_count:
            monitor.future_frames.append(frame.copy())
        if monitor.event_triggered and len(monitor.future_frames) == monitor.future_frames_count:
            monitor.save_event_video()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


class AdvancedMonitor:
    def __init__(self):
        try:
            # 필요한 모델 및 분류기 초기화
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise IOError("Haar Cascade 파일을 로드할 수 없습니다.")

            self.net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
            self.classes = []
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]

            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

            # 얼굴 인식 관련 변수
            self.known_face_encodings = []
            self.known_face_names = []
            self.load_known_faces()

            # 폴더 생성
            self.log_folder = "logs"
            if not os.path.exists(self.log_folder):
                os.makedirs(self.log_folder)

            self.video_folder = "video"
            if not os.path.exists(self.video_folder):
                os.makedirs(self.video_folder)

            # 음성 인식 초기화
            self.speech_recognizer = sr.Recognizer()
            self.audio_source = None
            self.recognized_text = ""
            self.audio_thread = None
            self.is_listening = False

            # 이벤트 클립 저장용 버퍼
            self.frame_rate = 20
            self.buffer_seconds = 1
            self.frame_buffer = deque(maxlen=self.frame_rate * self.buffer_seconds)
            self.event_triggered = False
            self.future_frames = deque()
            self.future_frames_count = self.frame_rate * self.buffer_seconds

            # 회전 감지 관련 변수
            self.orb = cv2.ORB_create()
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self.prev_keypoints = None
            self.prev_descriptors = None
            self.total_rotation = 0.0
            self.rotation_recording = False
            self.rotation_video_writer = None

        except Exception as e:
            print(f"초기화 중 오류 발생: {e}")
            raise

        # 통계 변수 초기화
        self.start_time = time.time()
        self.total_frames = 0
        self.absent_frames = 0
        self.multiple_faces_frames = 0
        self.phone_detected_frames = 0
        self.prev_frame = None
        self.movement_threshold = 1000
        self.recording = False
        self.video_writer = None

    def start_speech_recognition(self):
        try:
            self.audio_source = sr.Microphone()
            self.is_listening = True
            self.audio_thread = threading.Thread(target=self._listen_audio)
            self.audio_thread.daemon = True
            self.audio_thread.start()
        except Exception as e:
            print(f"음성 인식 초기화 오류: {e}")

    def _listen_audio(self):
        with self.audio_source as source:
            self.speech_recognizer.adjust_for_ambient_noise(source)
            while self.is_listening:
                try:
                    audio = self.speech_recognizer.listen(source, timeout=1)
                    text = self.speech_recognizer.recognize_google(audio, language='ko-KR')
                    self.recognized_text = text
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"음성 인식 오류 상세: {str(e)}")

    def get_recognized_text(self):
        return self.recognized_text

    def calculate_rotation_angle(self, matches, keypoints1, keypoints2):
        """특징점 매칭으로 회전 각도 계산"""
        try:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            if matrix is not None:
                angle = np.arctan2(matrix[0, 1], matrix[0, 0]) * (180 / np.pi)
                return angle
            return 0.0
        except Exception as e:
            print(f"회전 각도 계산 오류: {e}")
            return 0.0

    def detect_rotation(self, frame):
        """회전 감지 처리"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)

            if descriptors is None:
                self.prev_keypoints = None
                self.prev_descriptors = None
                return frame, 0, False

            if self.prev_descriptors is not None:
                matches = self.bf.match(self.prev_descriptors, descriptors)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) > 5:
                    rotation_angle = self.calculate_rotation_angle(matches, self.prev_keypoints, keypoints)
                    self.total_rotation += rotation_angle

                    # 화면에 회전 정보 표시
                    cv2.putText(frame, f"Rotation: {abs(self.total_rotation):.1f}/360.0", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # 360도 이상 회전 감지되면 녹화 종료
                    if abs(self.total_rotation) >= 30.0 and self.rotation_recording:
                        # 회전 완료 메시지 표시
                        cv2.putText(frame, "360 Rotation Complete!", (frame.shape[1] // 2 - 150, frame.shape[0] // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        return frame, self.total_rotation, True

            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return frame, self.total_rotation, False
        except Exception as e:
            print(f"회전 감지 오류: {e}")
            return frame, 0, False

    def start_rotation_recording(self, frame):
        """회전 감지 녹화 시작"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.video_folder, f"rotation_video_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.rotation_video_writer = cv2.VideoWriter(filename, fourcc, 20.0,
                                                         (frame.shape[1], frame.shape[0]))
            self.rotation_recording = True
            self.total_rotation = 0.0
            self.prev_keypoints = None
            self.prev_descriptors = None
            print(f"회전 감지 녹화 시작: {filename}")
            return True
        except Exception as e:
            print(f"회전 녹화 시작 오류: {e}")
            return False

    def stop_rotation_recording(self):
        """회전 감지 녹화 종료"""
        try:
            if self.rotation_video_writer:
                self.rotation_video_writer.release()
            self.rotation_recording = False
            self.total_rotation = 0.0
            print("회전 감지 녹화 종료")
            return True
        except Exception as e:
            print(f"회전 녹화 종료 오류: {e}")
            return False

    def trigger_event(self, event_type):
        if self.event_triggered:
            return

        self.event_triggered = True
        self.future_frames.clear()
        self.event_type = event_type
        print(f"이벤트 발생! - {event_type} (전후 1초 클립 저장 준비)")

    def save_event_video(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        event_name = "face" if self.event_type == "face" else "phone"
        filename = os.path.join(self.video_folder, f"event_{event_name}_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        if len(self.frame_buffer) == 0:
            print("이벤트를 저장할 버퍼가 없습니다.")
            self.event_triggered = False
            return

        height, width, _ = self.frame_buffer[0].shape
        out = cv2.VideoWriter(filename, fourcc, self.frame_rate, (width, height))

        for frame in self.frame_buffer:
            out.write(frame)

        for frame in self.future_frames:
            out.write(frame)

        out.release()
        print(f"이벤트 클립 저장 완료: {filename}")
        self.event_triggered = False
        self.event_type = None

    def recognize_faces(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model="small")

            # 얼굴이 감지되지 않으면 부재 프레임 증가
            if len(face_locations) == 0:
                self.absent_frames += 1
                cv2.putText(frame, "No Face Detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return frame, 0

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            return frame, len(face_locations)
        except Exception as e:
            print(f"얼굴 인식 오류: {e}")
            return frame, 0

    def detect_phone(self, frame):
        try:
            height, width, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            phone_detected = False
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.25 and self.classes[class_id] in ["cell phone", "mobile phone"]:
                        phone_detected = True
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, f"Phone: {confidence:.2f}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if phone_detected:
                self.phone_detected_frames += 1
            return phone_detected

        except Exception as e:
            print(f"휴대폰 감지 오류: {e}")
            return False

    def detect_movement(self, frame):
        try:
            if self.prev_frame is None:
                self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return False

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(self.prev_frame, gray)
            self.prev_frame = gray

            movement = np.sum(frame_diff > 30)
            if movement > self.movement_threshold:
                cv2.putText(frame, "Movement Detected!", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                return True
            return False
        except Exception as e:
            print(f"움직임 감지 오류: {e}")
            return False

    def process_frame(self, frame):
        try:
            frame, face_count = self.recognize_faces(frame)

            # 이벤트1: 사람 두 명 이상 감지
            if face_count >= 2:
                self.trigger_event("face")
                self.multiple_faces_frames += 1
                cv2.putText(frame, f"WARNING: {face_count} Faces Detected!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 이벤트2: 핸드폰 감지
            phone_detected = self.detect_phone(frame)
            if phone_detected:
                self.trigger_event("phone")

            # 회전 감지 처리
            if self.rotation_recording:
                frame, rotation_angle, rotation_complete = self.detect_rotation(frame)
                if rotation_complete:
                    self.stop_rotation_recording()
                elif self.rotation_video_writer:
                    self.rotation_video_writer.write(frame)

            self.detect_movement(frame)
            self.total_frames += 1
            self.display_stats(frame)

            if self.recording and self.video_writer:
                self.video_writer.write(frame)

            return frame
        except Exception as e:
            print(f"프레임 처리 오류: {e}")
            return frame

    def display_stats(self, frame):
        elapsed_time = int(time.time() - self.start_time)
        if self.total_frames > 0:
            presence_rate = ((self.total_frames - self.absent_frames) / self.total_frames) * 100
            phone_rate = (self.phone_detected_frames / self.total_frames) * 100
            multiple_faces_rate = (self.multiple_faces_frames / self.total_frames) * 100
        else:
            presence_rate = phone_rate = multiple_faces_rate = 100.0

        cv2.putText(frame, f"Time: {elapsed_time}s", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Presence: {presence_rate:.1f}%", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Phone Detection: {phone_rate:.1f}%", (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def toggle_recording(self, frame):
        try:
            if not os.path.exists(self.video_folder):
                os.makedirs(self.video_folder)

            if not self.recording:
                filename = os.path.join(self.video_folder,
                                        f"exam_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0,
                                                    (frame.shape[1], frame.shape[0]))
                if not self.video_writer.isOpened():
                    print("비디오 라이터를 열 수 없습니다")
                    return

                self.recording = True
                self.start_speech_recognition()
                print(f"녹화 시작: {filename}")
            else:
                if self.video_writer:
                    self.video_writer.release()
                self.recording = False
                self.is_listening = False
                if self.audio_thread:
                    self.audio_thread.join()
                log_filename = f"monitoring_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                self.save_log(log_filename)
                print("녹화 중지 및 로그 저장 완료")
        except Exception as e:
            print(f"녹화 오류: {e}")
            if self.video_writer:
                self.video_writer.release()

    def load_known_faces(self):
        known_faces_dir = "known_faces"
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            return

        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith((".jpg", ".png")):
                filepath = os.path.join(known_faces_dir, filename)
                try:
                    image = face_recognition.load_image_file(filepath)
                    encodings = face_recognition.face_encodings(image, model="small")
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(os.path.splitext(filename)[0])
                except Exception as e:
                    print(f"이미지 처리 오류: {filename} - {e}")

    def save_log(self, filename):
        try:
            elapsed_time = int(time.time() - self.start_time)
            if self.total_frames > 0:
                presence_rate = ((self.total_frames - self.absent_frames) / self.total_frames) * 100
                phone_rate = (self.phone_detected_frames / self.total_frames) * 100
                multiple_faces_rate = (self.multiple_faces_frames / self.total_frames) * 100
            else:
                presence_rate = phone_rate = multiple_faces_rate = 100.0

            log_filepath = os.path.join(self.log_folder, filename)
            with open(log_filepath, 'w', encoding='utf-8') as f:
                f.write(f"모니터링 보고서 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"총 모니터링 시간: {elapsed_time} 초\n")
                f.write(f"총 프레임: {self.total_frames}\n")
                f.write(f"부재 프레임: {self.absent_frames}\n")
                f.write(f"출석률: {presence_rate:.1f}%\n")
                f.write(f"휴대폰 감지율: {phone_rate:.1f}%\n")
                f.write(f"다중 얼굴 감지율: {multiple_faces_rate:.1f}%\n")
                if self.recognized_text:
                    f.write(f"\n음성 인식 결과:\n{self.recognized_text}\n")
        except Exception as e:
            print(f"로그 저장 오류: {e}")


# Flask 라우트
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_recording', methods=['POST'])
def toggle_recording():
    global monitor, cap
    if monitor:
        ret, frame = cap.read()
        if ret:
            monitor.toggle_recording(frame)
            return jsonify({'recording': monitor.recording})
    return jsonify({'error': '녹화 상태를 변경할 수 없습니다.'})


@app.route('/start_rotation_recording', methods=['POST'])
def start_rotation_recording():
    global monitor, cap
    if monitor:
        ret, frame = cap.read()
        if ret:
            success = monitor.start_rotation_recording(frame)
            return jsonify({'success': success})
    return jsonify({'success': False, 'error': '회전 감지 녹화를 시작할 수 없습니다.'})


@app.route('/stop_rotation_recording', methods=['POST'])
def stop_rotation_recording():
    global monitor
    if monitor:
        success = monitor.stop_rotation_recording()
        return jsonify({'success': success})
    return jsonify({'success': False, 'error': '회전 감지 녹화를 종료할 수 없습니다.'})


@app.route('/get_stats')
def get_stats():
    global monitor
    if monitor and monitor.total_frames > 0:
        presence_rate = ((monitor.total_frames - monitor.absent_frames) / monitor.total_frames) * 100
        phone_rate = (monitor.phone_detected_frames / monitor.total_frames) * 100
        return jsonify({
            'presence_rate': presence_rate,
            'phone_rate': phone_rate
        })
    return jsonify({
        'presence_rate': 100.0,
        'phone_rate': 0.0
    })


@app.route('/get_speech_text')
def get_speech_text():
    global monitor
    if monitor:
        return jsonify({'text': monitor.get_recognized_text()})
    return jsonify({'text': ''})


@app.route('/get_rotation_status')
def get_rotation_status():
    global monitor
    if monitor and monitor.rotation_recording:
        return jsonify({
            'angle': abs(monitor.total_rotation),
            'complete': abs(monitor.total_rotation) >= 360.0
        })
    return jsonify({
        'angle': 0,
        'complete': False
    })


@app.route('/upload_face', methods=['POST'])
def upload_face():
    if 'face' not in request.files:
        return jsonify({'error': '파일이 업로드되지 않았습니다.'})

    file = request.files['face']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('known_faces', filename)
        file.save(filepath)
        monitor.load_known_faces()
        return jsonify({'message': '얼굴이 등록되었습니다.'})


if __name__ == "__main__":
    try:
        cap = cv2.VideoCapture(0)
        monitor = AdvancedMonitor()
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        if monitor:
            monitor.save_log(f"monitoring_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        if cap:
            cap.release()
        cv2.destroyAllWindows()