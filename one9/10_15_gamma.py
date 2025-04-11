import cv2
import numpy as np
import os
import time
from collections import deque

class LaneDetector:
    def __init__(self, image_dir):
        # 이미지 디렉토리 불러오기
        if not os.path.exists(image_dir):
            raise ValueError("이미지 디렉토리가 잘못되었습니다.")
        
        self.image_dir = image_dir
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        if len(self.image_files) == 0:
            raise ValueError("이미지 파일이 없습니다.")
        
        test_image = cv2.imread(self.image_files[0])
        if test_image is None:
            raise ValueError("이미지를 열 수 없습니다.")
        
        # 이미지 너비와 높이 설정
        self.width = test_image.shape[1]
        self.height = test_image.shape[0]

        # 좌우 무게 중심 별도로 큐에 저장
        self.left_centers = deque(maxlen=20)
        self.right_centers = deque(maxlen=20)
        
        # 이전 조향 각도를 저장할 변수
        self.previous_angle = 0

        # 출력 디렉토리 설정
        self.output_dir = 'output_st/1_curve_yuv_resize_edge_blur'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    # 카메라 파라미터 반환
    def load_camera_params(self):
        camera_matrix = np.array([[294.9, 0, 630.9], [0, 680.1, -1415.6], [0, 0, 1]])
        dist_coeffs = np.array([-0.0058, 0.0003, 0, 0])
        return camera_matrix, dist_coeffs

    # YUV 변환 및 Y 채널 보정
    def apply_yuv_transform(self, roi_image):
        frame_resized = cv2.resize(roi_image, (roi_image.shape[1] // 2, roi_image.shape[0] // 2))
        YUV = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)
        Y, U, V = cv2.split(YUV)

        Y_max = np.max(Y)
        Y_max_3_5 = Y_max * (3 / 5)

        Y_trans = np.where(
            (Y > 0) & (Y < Y_max_3_5), 
            Y / 3, 
            np.where(
                (Y >= Y_max_3_5) & (Y < Y_max), 
                (Y * 2) - Y_max, 
                Y
            )
        )

        Y_trans_uint8 = Y_trans.astype(np.uint8)
        transformed_YUV = cv2.merge([Y_trans_uint8, U, V])
        transformed_frame = cv2.cvtColor(transformed_YUV, cv2.COLOR_YUV2BGR)
        original_size_frame = cv2.resize(transformed_frame, (roi_image.shape[1], roi_image.shape[0]))

        return original_size_frame

    # ROI 영역 자르기
    def crop_roi(self, frame, roi_corners):
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, roi_corners, (255, 255, 255))
        roi_image = cv2.bitwise_and(frame, mask)
        return roi_image

    # HSV 필터 적용
    def apply_hsv_filter(self, frame, lower_color, upper_color):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_color = cv2.inRange(hsv, lower_color, upper_color)
        filtered_image = cv2.bitwise_and(frame, frame, mask=mask_color)
        return filtered_image

    # Canny 에지 검출
    def apply_canny_edge(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 2)
        edges = cv2.Canny(blur, 150, 200)
        return edges

    # 무게중심 계산
    def calculate_centroid(self, mask):
        moments = cv2.moments(mask)
        if moments['m00'] != 0:
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
            return (centroid_x, centroid_y)
        else:
            return None

    # ROI 시각화
    def draw_roi(self, frame, roi_corners):
        cv2.polylines(frame, roi_corners, isClosed=True, color=(0, 255, 255), thickness=2)

    # 차선 검출 및 중심점 반환
    def detect_lane_combined(self, frame, lower_color, upper_color, roi_corners, min_slope_degrees=10):
        roi_image = self.crop_roi(frame, roi_corners)
        roi_image_yuv = self.apply_yuv_transform(roi_image)

        filtered_image = self.apply_hsv_filter(roi_image_yuv, lower_color, upper_color)
        edges = self.apply_canny_edge(filtered_image)

        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        combined_mask = cv2.bitwise_and(filtered_image, edges)

        lane_centroid = self.calculate_centroid(combined_mask)
        if lane_centroid is None:
            return np.zeros_like(frame), [], None

        lines = cv2.HoughLinesP(combined_mask, rho=1, theta=np.pi/180, threshold=80, lines=np.array([]), minLineLength=80, maxLineGap=50)

        line_image = np.zeros_like(frame)
        line_centroids = []
        min_slope_radians = np.deg2rad(min_slope_degrees)

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x2 - x1 == 0:
                        slope = float('inf')
                    else:
                        slope = (y2 - y1) / (x2 - x1)

                    angle = np.arctan(slope)
                    if np.abs(angle) < min_slope_radians:
                        continue

                    centroid_x = (x1 + x2) // 2
                    centroid_y = (y1 + y2) // 2
                    distance = np.sqrt((centroid_x - lane_centroid[0]) ** 2 + (centroid_y - lane_centroid[1]) ** 2)

                    if distance <= 30:
                        cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        line_centroids.append((centroid_x, centroid_y))

        if not line_centroids:
            line_centroids.append(lane_centroid)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(line_image, contours, -1, (255, 255, 255), 2)

        return line_image, line_centroids, lane_centroid

    # 조향각 계산
    def calculate_avg_steering_angle(self):
        if self.left_centers and self.right_centers:
            avg_left_center = np.mean(self.left_centers, axis=0).astype(int)
            avg_right_center = np.mean(self.right_centers, axis=0).astype(int)

            avg_center_x = (avg_left_center[0] + avg_right_center[0]) // 2
            avg_center_y = (avg_left_center[1] + avg_right_center[1]) // 2

            delta_x = avg_center_x - (self.width // 2)
            delta_y = self.height - avg_center_y
            avg_angle = np.degrees(np.arctan2(delta_x, delta_y))

            self.previous_angle = avg_angle
        else:
            avg_angle = self.previous_angle

        return avg_angle

    # 조향각 시각화
    def draw_steering_angle(self, frame, angle):
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height

        length = 100
        angle_rad = np.deg2rad(angle)

        end_x = int(center_x + length * np.sin(angle_rad))
        end_y = int(center_y - length * np.cos(angle_rad))

        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3)
        cv2.putText(frame, f'Steering Angle: {angle:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    def process_images(self):
        # ROI 설정
        roi_corners_left = np.array([[(50, 220), (0, self.height - 300), (self.width // 2 - 50, self.height - 300), (self.width // 2 - 50, 220)]], dtype=np.int32)
        roi_corners_right = np.array([[(self.width // 2 + 50, 220), (self.width // 2 + 50, self.height - 300), (self.width, self.height - 300), (self.width - 50, 220)]], dtype=np.int32)

        # HSV 범위 설정 (차선 색상 필터링)
        lower_white = np.array([0, 0, 220])
        upper_white = np.array([179, 25, 255])

        for idx, image_file in enumerate(self.image_files):
            frame = cv2.imread(image_file)

            start = time.time()

            # 렌즈 왜곡 보정
            camera_matrix, dist_coeffs = self.load_camera_params()
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

            # 좌, 우측 차선 검출 및 중심점 계산
            line_image_left, centroids_left, left_lane_centroid = self.detect_lane_combined(frame, lower_white, upper_white, [roi_corners_left])
            line_image_right, centroids_right, right_lane_centroid = self.detect_lane_combined(frame, lower_white, upper_white, [roi_corners_right])

            # 중심점들을 deque에 저장
            if centroids_left:
                self.left_centers.extend(centroids_left)
            if centroids_right:
                self.right_centers.extend(centroids_right)

            # 스티어링 각도 계산
            avg_angle = self.calculate_avg_steering_angle()
            self.draw_steering_angle(frame, avg_angle)

            # YUV 보정된 이미지에 ROI 그리기
            yuv_with_roi = self.apply_yuv_transform(frame)
            self.draw_roi(yuv_with_roi, [roi_corners_left])
            self.draw_roi(yuv_with_roi, [roi_corners_right])

            # **ROI가 그려진 이미지에 중심점 표시**
            if left_lane_centroid is not None:
                # 왼쪽 차선 중심점 그리기
                cv2.circle(yuv_with_roi, left_lane_centroid, 5, (0, 0, 255), -1)  # 초록색 원으로 표시
            if right_lane_centroid is not None:
                # 오른쪽 차선 중심점 그리기
                cv2.circle(yuv_with_roi, right_lane_centroid, 5, (0, 0, 255), -1)  # 초록색 원으로 표시

            # 좌우 차선 이미지를 더해서 합치기
            hough_contour_image = line_image_left + line_image_right

            # 처리된 이미지 두 개를 이어 붙이기
            combined_image = np.hstack((yuv_with_roi, hough_contour_image))

            end = time.time()
            print(f"Processing Time for {image_file} = {end - start:.5f} seconds")

            # 결과 이미지 저장
            output_image_path = os.path.join(self.output_dir, f"processed_{idx}.png")
            cv2.imwrite(output_image_path, combined_image)


if __name__ == "__main__":
    image_dir = '/home/ubuntu/catkin_ws/src/lane_detection/bag/hw_curve_3'
    detector = LaneDetector(image_dir)
    detector.process_images()
