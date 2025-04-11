import cv2
import numpy as np
import os
import time
from collections import deque

class LaneDetector:
    def __init__(self, image_dir):
        if not os.path.exists(image_dir):
            raise ValueError("이미지 디렉토리가 잘못되었습니다.")
        
        self.image_dir = image_dir
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        if len(self.image_files) == 0:
            raise ValueError("이미지 파일이 없습니다.")
        
        test_image = cv2.imread(self.image_files[0])
        if test_image is None:
            raise ValueError("이미지를 열 수 없습니다.")
        
        self.width = test_image.shape[1]
        self.height = test_image.shape[0]

        # 좌우 무게 중심을 별도로 저장할 큐
        self.left_centers = deque(maxlen=20)
        self.right_centers = deque(maxlen=20)
        
        # 이전 조향 각도를 저장할 변수
        self.previous_angle = 0

        # 출력 디렉토리 설정
        self.output_dir = 'output_contour_re3_09'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_camera_params(self):
        camera_matrix = np.array([[294.9, 0, 630.9], [0, 680.1, -1415.6], [0, 0, 1]])
        dist_coeffs = np.array([-0.0058, 0.0003, 0, 0])
        return camera_matrix, dist_coeffs
         
    def crop_roi(self, frame, roi_corners):
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, roi_corners, (255, 255, 255))
        roi_image = cv2.bitwise_and(frame, mask)
        return roi_image

    def apply_hsv_filter(self, frame, lower_color, upper_color):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_color = cv2.inRange(hsv, lower_color, upper_color)
        filtered_image = cv2.bitwise_and(frame, frame, mask=mask_color)   
        return filtered_image

    def apply_canny_edge(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Canny(blur, 100, 170)
        return edges

    def calculate_centroid(self, mask):
        """ cv2.moments를 이용하여 차선 중심 계산 """
        moments = cv2.moments(mask)
        if moments['m00'] != 0:
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
            return (centroid_x, centroid_y)
        else:
            return None

    def detect_lane_combined(self, frame, lower_color, upper_color, roi_corners, min_slope_degrees=10):
        """ 차선을 감지하고 Contour와 Hough Line을 결합하여 최종 차선을 검출 """
        roi_image = self.crop_roi(frame, roi_corners)
        filtered_image = self.apply_hsv_filter(roi_image, lower_color, upper_color)
        edges = self.apply_canny_edge(filtered_image)

        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
        combined_mask = cv2.bitwise_and(filtered_image, edges)

        # Contour 계산
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = np.zeros_like(frame)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)

        # Hough Line Transform 계산
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

                    distance = np.sqrt((centroid_x - self.width // 2) ** 2 + (centroid_y - self.height) ** 2)

                    if distance <= 30:
                        cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        line_centroids.append((centroid_x, centroid_y))

        # Hough Line과 Contour를 결합
        final_image = cv2.addWeighted(line_image, 0.7, contour_image, 0.3, 0)

        # 무게 중심 계산
        lane_centroid = self.calculate_centroid(combined_mask)
        if lane_centroid is None:
            return final_image, []

        if not line_centroids:
            line_centroids.append(lane_centroid)

        return final_image, line_centroids

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
        # 좌우로 나눈 ROI 설정
        roi_corners_left = np.array([[(50, 220), (0, self.height-300), (self.width//2-50, self.height-300), (self.width//2-50, 220)]], dtype=np.int32)
        roi_corners_right = np.array([[(self.width//2+50, 220), (self.width//2+50, self.height-300), (self.width, self.height-300), (self.width-50, 220)]], dtype=np.int32)

        lower_white = np.array([0, 0, 220])
        upper_white = np.array([179, 25, 255])

        for idx, image_file in enumerate(self.image_files):
            frame = cv2.imread(image_file)

            start = time.time()

            camera_matrix, dist_coeffs = self.load_camera_params()

            # 왜곡 보정 적용
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

            # 왼쪽, 오른쪽 차선 각각 처리
            line_image_left, centroids_left = self.detect_lane_combined(frame, lower_white, upper_white, roi_corners_left)
            line_image_right, centroids_right = self.detect_lane_combined(frame, lower_white, upper_white, roi_corners_right)

            combined_image = cv2.addWeighted(line_image_left, 0.5, line_image_right, 0.5, 0)

            end = time.time()
            print(f"Processing Time for {image_file} = {end - start:.5f} seconds")

            # 결과 저장
            output_image_path = os.path.join(self.output_dir, f"processed_{idx}.png")
            cv2.imwrite(output_image_path, combined_image)

if __name__ == "__main__":
    image_dir = '/home/ubuntu/catkin_ws/src/lane_detection/bag/re3'  # 이미지 폴더 경로 설정
    detector = LaneDetector(image_dir)
    detector.process_images()
