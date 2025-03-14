import cv2
import numpy as np
# 현재 제일 좋음음
class FullRotationDetector:
    def __init__(self, output_video_file="rotation_output.avi"):
        self.orb = cv2.ORB_create()  # ORB 특징점 추출기
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # BF 매처
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.total_rotation = 0.0  # 누적 회전 각도
        
        # 비디오 저장 설정
        self.output_video_file = output_video_file
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 비디오 코덱
        self.out = None

    def calculate_rotation_angle(self, matches, keypoints1, keypoints2):
        """특징점 매칭으로 회전 각도 계산"""
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 변환 행렬 추정
        matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if matrix is not None:
            angle = np.arctan2(matrix[0, 1], matrix[0, 0]) * (180 / np.pi)
            return angle
        return 0.0

    def process_frame(self, frame):
        """프레임 처리 및 회전 계산"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        # 현재 프레임에서 특징점이 감지되지 않으면 바로 반환
        if descriptors is None:
            self.prev_keypoints = None
            self.prev_descriptors = None
            return frame

        if self.prev_descriptors is not None:
            # 이전 프레임과 현재 프레임에서 특징점 매칭
            matches = self.bf.match(self.prev_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > 5:  # 충분한 매칭점이 있을 때만 계산
                rotation_angle = self.calculate_rotation_angle(matches, self.prev_keypoints, keypoints)
                self.total_rotation += rotation_angle
                print(f"회전 각도: {rotation_angle:.2f}° | 누적 회전: {self.total_rotation:.2f}°")

                # 360도 회전 감지
                if abs(self.total_rotation) >= 60.0:
                    print("한 바퀴 회전 완료!")
                    self.total_rotation = 0.0  # 초기화
                    return

        # 현재 프레임 데이터를 다음 계산에 사용
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        return frame

    def start(self, video_source=0):
        """카메라에서 프레임 읽기 시작"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("카메라를 열 수 없습니다.")
            return

        # 비디오 출력 파일 설정 (해상도 및 프레임 속도 설정)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.out is None:
            self.out = cv2.VideoWriter(self.output_video_file, self.fourcc, 20.0, (frame_width, frame_height))

        print("카메라 회전 감지 시작...")
        print("종료하려면 'q'를 누르세요.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = self.process_frame(frame)
                cv2.imshow("Rotation Detector", frame)

                # 비디오 녹화
                self.out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            self.out.release()  # 비디오 파일 저장 종료
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = FullRotationDetector(output_video_file="rotation_output.avi")
    detector.start()
