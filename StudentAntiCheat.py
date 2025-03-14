import cv2
import numpy as np
import time
from datetime import datetime
import os
import face_recognition
from collections import deque

class AdvancedMonitor:
    def __init__(self):
        try:
            # Haar Cascade 분류기 초기화 (사용 안 해도 무방)
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise IOError("Haar Cascade 파일을 로드할 수 없습니다.")

            # YOLO 모델 로드 (휴대폰 감지용)
            self.net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
            self.classes = []
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]

            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

            # 얼굴 데이터 로드
            self.known_face_encodings = []
            self.known_face_names = []
            self.load_known_faces()

            # 로그 저장 폴더 & 비디오 저장 폴더 생성
            self.log_folder = "logs"
            if not os.path.exists(self.log_folder):
                os.makedirs(self.log_folder)

            self.video_folder = "video"
            if not os.path.exists(self.video_folder):
                os.makedirs(self.video_folder)

            # ------------------------ 이벤트 클립 저장용 버퍼 ------------------------
            self.frame_rate = 20          # 초당 프레임 수 (카메라 FPS와 맞춰야 전후 1초 정확도 ↑)
            self.buffer_seconds = 1       # 이벤트 발생 전후로 저장할 초 단위(1초로 변경)
            self.frame_buffer = deque(maxlen=self.frame_rate * self.buffer_seconds)  # 과거(1초) 프레임을 저장
            self.event_triggered = False
            self.future_frames = deque()  # 이벤트 발생 후 미래(1초) 프레임을 저장
            self.future_frames_count = self.frame_rate * self.buffer_seconds
            # ----------------------------------------------------------------------

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
        self.movement_threshold = 1000  # 움직임 감지 임계값

        # **연속 녹화(Full video) 관련 변수**
        self.full_video_writer = None   # 전체 영상을 저장할 VideoWriter

    # ------------------------------------------------------
    #  1) 이벤트(전후 1초) 클립 저장 관련 메서드
    # ------------------------------------------------------
    def trigger_event(self):
        """
        이벤트(사람 2명 이상) 발생 시:
         - 이미 이벤트 처리 중이라면 무시
         - 아니라면 event_triggered = True 로 만들어 미래 프레임 수집 시작
        """
        if self.event_triggered:
            # 이미 이벤트 진행 중이면 중복 방지
            return

        self.event_triggered = True
        self.future_frames.clear()
        print("이벤트1 발생! (전후 1초 클립 저장 준비)")

    def save_event_video(self):
        """
        버퍼(과거 1초) + 미래(1초) 프레임을 합쳐
        event1_video_(날짜시간).avi 로 저장
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.video_folder, f"event1_video_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # 버퍼가 비어있지 않은 경우에만 저장 진행
        if len(self.frame_buffer) == 0:
            print("이벤트를 저장할 버퍼가 없습니다.")
            self.event_triggered = False
            return

        height, width, _ = self.frame_buffer[0].shape
        out = cv2.VideoWriter(filename, fourcc, self.frame_rate, (width, height))

        # 1) 과거(1초) 프레임 저장
        for frame in self.frame_buffer:
            out.write(frame)

        # 2) 미래(1초) 프레임 저장
        for frame in self.future_frames:
            out.write(frame)

        out.release()
        print(f"이벤트1 클립 저장 완료: {filename}")
        self.event_triggered = False

    # ------------------------------------------------------
    #  2) 얼굴 인식
    # ------------------------------------------------------
    def load_known_faces(self):
        """등록된 얼굴 이미지(known_faces 폴더)를 읽어 얼굴 인코딩 저장"""
        known_faces_dir = "known_faces"
        if not os.path.exists(known_faces_dir):
            print(f"등록된 얼굴 폴더가 존재하지 않습니다: {known_faces_dir}")
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
                        print(f"얼굴 데이터 추가됨: {filename}")
                    else:
                        print(f"얼굴이 감지되지 않았습니다: {filename}")
                except Exception as e:
                    print(f"이미지 처리 중 오류 발생: {filename} - {e}")

    def recognize_faces(self, frame):
        """
        얼굴 인식 후, 얼굴 개수도 함께 반환
        return: (처리된 frame, 감지된 얼굴 개수)
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model="small")

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]

                # 얼굴 박스 및 이름 표시
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            # 얼굴 개수 반환
            return frame, len(face_locations)

        except Exception as e:
            print(f"얼굴 인식 중 오류 발생: {e}")
            return frame, 0

    # ------------------------------------------------------
    #  3) 휴대폰 감지
    # ------------------------------------------------------
    def detect_phone(self, frame):
        """
        휴대폰 감지 후, 감지 여부(True/False) 반환
        """
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

                        # 감지 결과 시각화
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, f"Phone: {confidence:.2f}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        print(f"휴대폰 감지 - 신뢰도: {confidence:.2f}, 위치: ({x}, {y}, {w}, {h})")

            if phone_detected:
                self.phone_detected_frames += 1
            return phone_detected

        except Exception as e:
            print(f"휴대폰 감지 중 오류 발생: {e}")
            return False

    # ------------------------------------------------------
    #  4) 움직임 감지
    # ------------------------------------------------------
    def detect_movement(self, frame):
        """움직임 감지"""
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
            print(f"움직임 감지 중 오류 발생: {e}")
            return False

    # ------------------------------------------------------
    #  5) 메인 프로세스 (이벤트1: 사람2명, 이벤트2: 휴대폰 감지 등)
    # ------------------------------------------------------
    def process_frame(self, frame):
        """
        각 프레임 처리:
         - 얼굴 인식 -> 사람 2명 이상인지 확인(이벤트1)
         - 휴대폰 감지(이벤트2)
         - (필요 시) 움직임 감지
         - 이벤트1 발생 시 trigger_event()
        """
        try:
            # 얼굴 인식
            frame, face_count = self.recognize_faces(frame)

            # (이벤트1) 사람 두 명 이상 감지
            if face_count >= 2:
                # 전후 1초 클립 저장을 위한 이벤트 트리거
                self.trigger_event()

            # (이벤트2) 휴대폰 감지 (별도 이벤트 클립은 저장하지 않음, 메시지만)
            phone_found = self.detect_phone(frame)
            if phone_found:
                print("이벤트2 발생! (휴대폰 감지)")

            # 움직임 감지 (원하는 경우 추가 로직)
            self.detect_movement(frame)

            # 통계 업데이트 및 표시
            self.total_frames += 1
            self.display_stats(frame)

            return frame
        except Exception as e:
            print(f"프레임 처리 중 오류 발생: {e}")
            return frame

    def display_stats(self, frame):
        """통계 정보 표시"""
        elapsed_time = int(time.time() - self.start_time)
        if self.total_frames > 0:
            presence_rate = ((self.total_frames - self.absent_frames) / self.total_frames) * 100
            phone_rate = (self.phone_detected_frames / self.total_frames) * 100
            multiple_faces_rate = (self.multiple_faces_frames / self.total_frames) * 100
        else:
            presence_rate = phone_rate = multiple_faces_rate = 100.0

        # 상태 정보 표시
        cv2.putText(frame, f"Time: {elapsed_time}s", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Presence: {presence_rate:.1f}%", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Phone Detection: {phone_rate:.1f}%", (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ------------------------------------------------------
    #  6) 로그 저장
    # ------------------------------------------------------
    def save_log(self, filename):
        """모니터링 로그 저장"""
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
        except Exception as e:
            print(f"로그 저장 중 오류 발생: {e}")

    # ------------------------------------------------------
    #  7) 메인 루프 - 카메라 열고, 상시 녹화 + 이벤트 처리
    # ------------------------------------------------------
    def start_monitoring(self, camera_index=0):
        """실시간 모니터링 시작 (상시 녹화, q 키로 종료)"""
        print("모니터링 시작...")
        print("조작법:")
        print("- 'q': 종료 (프로그램 종료와 함께 상시 녹화 중단)")

        # 전체 영상(Full video) 파일명 설정 (video 폴더 아래)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        full_video_filename = os.path.join(self.video_folder, f"full_video_{timestamp}.avi")
        log_filename = f"monitoring_log_{timestamp}.txt"

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return

        # 전체 영상 녹화용 VideoWriter 생성
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        ret, frame = cap.read()
        if not ret:
            print("첫 프레임을 읽을 수 없습니다.")
            return

        # 영상 크기 정보를 첫 프레임 기준으로 가져옴
        height, width, _ = frame.shape
        self.full_video_writer = cv2.VideoWriter(full_video_filename, fourcc, 20.0, (width, height))
        print(f"[Full Video] 녹화를 시작합니다: {full_video_filename}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("카메라에서 프레임을 읽을 수 없습니다.")
                    break

                # -------------------- 상시 녹화 --------------------
                self.full_video_writer.write(frame)

                # -------------------- 이벤트 클립 처리 --------------------
                # 1) 과거(1초) 버퍼에 현재 프레임 추가
                self.frame_buffer.append(frame.copy())

                # 2) 만약 이벤트가 트리거된 상태라면, 미래(1초) 프레임 쌓기
                if self.event_triggered and len(self.future_frames) < self.future_frames_count:
                    self.future_frames.append(frame.copy())

                # 3) 미래 프레임 1초치가 모두 쌓이면 이벤트 영상 저장
                if self.event_triggered and len(self.future_frames) == self.future_frames_count:
                    self.save_event_video()
                    # save_event_video() 내부에서 self.event_triggered = False로 종료

                # -------------------- 메인 처리(사람2명·휴대폰·움직임 등) --------------------
                processed_frame = self.process_frame(frame)

                # 화면에 출력
                cv2.imshow('Exam Monitoring', processed_frame)

                # 종료 키 체크
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        finally:
            # 종료 처리
            print("\n모니터링 종료")
            # 1) 상시 녹화 종료
            if self.full_video_writer is not None:
                self.full_video_writer.release()
                print(f"[Full Video] 녹화를 종료합니다: {full_video_filename}")

            # 2) 카메라 및 윈도우 해제
            cap.release()
            cv2.destroyAllWindows()

            # 3) 로그 저장
            self.save_log(log_filename)
            print(f"로그 파일이 저장되었습니다: {os.path.join(self.log_folder, log_filename)}")


if __name__ == "__main__":
    monitor = AdvancedMonitor()
    monitor.start_monitoring()
