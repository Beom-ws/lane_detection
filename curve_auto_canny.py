# # # 이미지 파일을 처리하는 코드 : Auto Canny 이미지에 대해서 Hough Transform 진행 + hough에 중점을 원으로 그리는 코드 

# import cv2
# import numpy as np
# import os
# import math
# import time
# from collections import deque


# class LaneDetector:
#     def __init__(self, image_dir):
#         # 이미지 디렉토리 불러오기
#         if not os.path.exists(image_dir):
#             raise ValueError("이미지 디렉토리가 잘못되었습니다.")
        
#         self.image_dir = image_dir
#         self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

#         if len(self.image_files) == 0:
#             raise ValueError("이미지 파일이 없습니다.")

#         test_image = cv2.imread(self.image_files[0])
#         if test_image is None:
#             raise ValueError("이미지를 열 수 없습니다.")
        
#         # 이미지 너비와 높이 설정
#         self.width = test_image.shape[1]
#         self.height = test_image.shape[0]
#         self.angle_data = []
#         self.previous_x = []
#         self.angle = 0
#         self.Y_max = 0
#         self.Y_max_3_5 = 0
#         self.angle = 0
#         self.angle_list = []
#         self.angle_add = []


#     # Y에 대한 감마 보정을, 1차식으로 변경하여 적용 : YUV_transform
#     def YUV_transform(self, Y):
#         self.Y_max = np.max(Y)
#         self.Y_max_3_5 = self.Y_max * (3 / 5)

#         Y_trans = np.where(
#             (Y > 0) & (Y < self.Y_max_3_5),
#             Y / 3,
#             np.where(
#                 (Y >= self.Y_max_3_5) & (Y < self.Y_max),
#                 (Y * 2) - self.Y_max,
#                 Y
#             )
#         )
#         Y_trans_uint8 = Y_trans.astype(np.uint8)
#         gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (13, 13), 2)
#         return gaussian_image
    

#     # 중앙값의 1.33/2.33 배 canny의 하향/상향 적용 
#     def yuv_auto_canny(self, image, sigma=0.33):
#         global v
#         v = np.median(image)
#         lower2 = int(min(255, (1.0 + sigma) * v))
#         lower = int(max(50, lower2))
#         upper = int(min(255, (2.0 + sigma) * v))
#         edged = cv2.Canny(image, lower, upper)
#         print('yuv auto \nlower: %d  upper: %d median: %d' % (lower, upper, v))
#         return edged

#     def avg(self, data):
#         return sum(data) / len(data)


#     # 이미지 처리 메인 코드
#     def process_images(self):
#         for image_file in self.image_files:
#             frame = cv2.imread(image_file)
#             if frame is None:
#                 print(f"이미지를 읽을 수 없습니다: {image_file}")
#                 continue

#             # 오른쪽만 사용하기 위해 x축의 오른쪽만 잘라냄30
#             frame_resized = frame[:, self.width // 2:self.width]
#             height, width, _ = frame_resized.shape
            
#             # 노이즈를 줄이고 처리 속도를 빠르게 만들기 위해 이미지를 작게 수정
#             resized_image = cv2.resize(frame_resized, (width // 2, height // 2))

#             ### Hough Transform 결과를 띄우기 위해 resized_image 크기의 빈 행렬 생성
#             # 검출된 직선의 각도를 확인하기 위한 코드 (1)
#             base_hough = np.zeros_like(resized_image)

#             # YUV 변환하여 Y 정보만 추출
#             YUV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
#             Y, _, _ = cv2.split(YUV)

#             # YUV_transform 적용
#             Y_frame = self.YUV_transform(Y)

#             # YUV_transformd에 대해 Auto canny 적용
#             auto_canny = self.yuv_auto_canny(Y_frame)

#             #컨투어 도출하고 그리기
#             base_contours = np.zeros_like(resized_image)
#             contours, _ = cv2.findContours(auto_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             cv2.drawContours(base_contours, contours, -1, (255, 255, 255), 2)
#             cv2.imshow("contours",base_contours)

#             # Auto canny에 대해 허프 변환 적용
#             ##minLineLength: 검출할 직선의 최소 길이로, 이 값보다 짧은 선분은 무시됩니다.
#             ##maxLineGap: 동일한 직선으로 간주하기 위한 최대 간격으로, 이 값 이하의 간격을 가진 선분들은 하나의 직선으로 연결됩니다.
#             lines = cv2.HoughLinesP(auto_canny, rho=1, theta=np.pi / 180, threshold=30, minLineLength=10, maxLineGap=50)

#             # 중점을 계산하기 위한 리스트
#             midpoints = []

            # # 검출된 직선의 각도를 확인하기 위한 코드 (2)
            # if lines is not None:
            #     self.angle_list.clear()   # 차선의 각도를 출력한 리스트를 초기화 하기 위한 초기화 함수
            #     if lines is not None:  # 차선이 없지 않다면 실행
            #         # 직진 그리는 과정
            #         for line in lines:
            #             x1, y1, x2, y2 = line[0]
            #             # 선의 각도 출력 (angle)
            #             self.angle = np.degrees(np.arctan2((y2-y1),(x2-x1)))

            #             # 여기에 각도 값의 min max를 설정
            #             if 20 <= self.angle <= 30:
#                             # base_hough에 흰색 선 그리기
#                             cv2.line(base_hough, (x1, y1), (x2, y2), (255, 255, 255), 2)

#                             # 검출된 각도 크기를 눈으로 확인하기 위해 리스트에 저장하는 함수
#                             self.angle_list.append(self.angle)
#                             self.angle_add.append(self.angle)

#                             # 검출 된 선의 중점 계산
#                             cx = (x1 + x2) // 2
#                             cy = (y1 + y2) // 2
#                             midpoints.append((cx, cy))

#                             # 중점을 이미지에 표시 (cv2.circle(image,center(원의 중심좌표),radius(반지름),color,thickness=두께(-1은 채워진 원)))
#                             cv2.circle(base_hough, (cx, cy), 5, (0, 255, 0), -1)
                
#                 if midpoints :
#                     avg_cx = int(sum([p[0] for p in midpoints]) / len(midpoints))
#                     avg_cy = int(sum([p[1] for p in midpoints]) / len(midpoints))

#                     # 최종 중점을 이미지에 표시
#                     cv2.circle(base_hough, (avg_cx, avg_cy), 10, (0, 0, 255), -1)
#                     print(f"최종 중점: ({avg_cx}, {avg_cy})")
#                 else:
#                     print("중점을 계산할 수 있는 선이 없습니다.")

#             # 검출된 직선의 각도를 확인하기 위한 코드 (3)
#             print(self.angle_list)
            

#             cv2.imshow("Auto Canny", auto_canny)
#             cv2.imshow("base_hough", base_hough)
#             cv2.imshow("resized_image", resized_image)

#             key = cv2.waitKey(0) & 0xFF
#             if key == ord('q'):  # 'q'를 누르면 종료
#                 break
#             elif key == ord('n'):  # 'n'을 누르면 다음 이미지로 이동
#                 continue

#         cv2.destroyAllWindows()
#         average = self.avg(self.angle_add)
#         print(average)
#         # 1115/yellow_straight : 24.782316253159745
#         # 1115/white_straight : 26.872538590155468
#         # 전기관 1층 직진 차선 : 27.828398381986386
#         ## 대략 직선은 20~30 사이의 각도쯤 되는듯




# if __name__ == "__main__":
#     ## 유진님이 주신 사진
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_left'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_right'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_straight'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_left'
#     image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_right'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_straight'

#     ## 전기관 1층 사진
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/load_image_2' #직선사진 하나

#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/load_image_1/load_image_1'  ##리모 사진 모음
#     detector = LaneDetector(image_dir)
#     detector.process_images()

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ1
## 허프라인이랑 컨투어랑 겹치는 컨투어만 도출
# import cv2
# import numpy as np
# import os
# import math


# class LaneDetector:
#     def __init__(self, image_dir):
#         if not os.path.exists(image_dir):
#             raise ValueError("이미지 디렉토리가 잘못되었습니다.")
        
#         self.image_dir = image_dir
#         self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

#         if len(self.image_files) == 0:
#             raise ValueError("이미지 파일이 없습니다.")

#         test_image = cv2.imread(self.image_files[0])
#         if test_image is None:
#             raise ValueError("이미지를 열 수 없습니다.")
        
#         self.width = test_image.shape[1]
#         self.height = test_image.shape[0]
#         self.angle_list = []
#         self.angle_add = []

#     def YUV_transform(self, Y):
#         self.Y_max = np.max(Y)
#         self.Y_max_3_5 = self.Y_max * (3 / 5)

#         Y_trans = np.where(
#             (Y > 0) & (Y < self.Y_max_3_5),
#             Y / 3,
#             np.where(
#                 (Y >= self.Y_max_3_5) & (Y < self.Y_max),
#                 (Y * 2) - self.Y_max,
#                 Y
#             )
#         )
#         Y_trans_uint8 = Y_trans.astype(np.uint8)
#         gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (13, 13), 2)
#         return gaussian_image
    
#     def yuv_auto_canny(self, image, sigma=0.33):
#         v = np.median(image)
#         lower2 = int(min(255, (1.0 + sigma) * v))
#         lower = int(max(50, lower2))
#         upper = int(min(255, (2.0 + sigma) * v))
#         edged = cv2.Canny(image, lower, upper)
#         return edged

#     def filter_contours_with_hough(self, contours, lines, distance_threshold=10):
#         """
#         허프 직선과 겹치는 컨투어만 필터링
#         """
#         filtered_contours = []
#         for contour in contours:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]  # line[0]으로 수정하지 않고 line을 직접 사용
#                 for point in contour:
#                     px, py = point[0]
#                     # 점과 선 사이의 거리 계산
#                     distance = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
#                     if distance < distance_threshold:
#                         filtered_contours.append(contour)
#                         break
#         return filtered_contours

#     def process_images(self):
#         for image_file in self.image_files:
#             frame = cv2.imread(image_file)
#             if frame is None:
#                 print(f"이미지를 읽을 수 없습니다: {image_file}")
#                 continue

#             frame_resized = frame[:, self.width // 2:self.width]
#             height, width, _ = frame_resized.shape
#             resized_image = cv2.resize(frame_resized, (width // 2, height // 2))

#             base_hough = np.zeros_like(resized_image)

#             YUV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
#             Y, _, _ = cv2.split(YUV)
#             Y_frame = self.YUV_transform(Y)
#             auto_canny = self.yuv_auto_canny(Y_frame)

#             base_contours = np.zeros_like(resized_image)
#             contours, _ = cv2.findContours(auto_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#             lines = cv2.HoughLinesP(auto_canny, rho=1, theta=np.pi / 180, threshold=30, minLineLength=10, maxLineGap=50)

#             filtered_lines = []
#             if lines is not None:
#                 self.angle_list.clear()
#                 for line in lines:
#                     x1, y1, x2, y2 = line[0]
#                     self.angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))

#                     # 각도 필터링 (20~30도)
#                     if 20 <= self.angle <= 30:
#                         filtered_lines.append([[x1, y1, x2, y2]])
#                         cv2.line(base_hough, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                         self.angle_list.append(self.angle)

#                 print("필터링된 각도 리스트:", self.angle_list)

#                 if filtered_lines:
#                     # 허프 직선과 겹치는 컨투어 필터링
#                     filtered_contours = self.filter_contours_with_hough(contours, filtered_lines)

#                     # 필터링된 컨투어 시각화
#                     for contour in filtered_contours:
#                         cv2.drawContours(base_contours, [contour], -1, (255, 255, 255), 2)

#             cv2.imshow("Contours with Hough", base_contours)
#             cv2.imshow("Hough Lines", base_hough)
#             cv2.imshow("Auto Canny", auto_canny)
#             cv2.imshow("Resized Image", resized_image)

#             key = cv2.waitKey(0) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('n'):
#                 continue

#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_right'
#     detector = LaneDetector(image_dir)
#     detector.process_images()


#,ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
import cv2
import numpy as np
import os
import math


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
        self.angle_list = []
        self.angle_add = []

    def YUV_transform(self, Y):
        self.Y_max = np.max(Y)
        self.Y_max_3_5 = self.Y_max * (3 / 5)

        Y_trans = np.where(
            (Y > 0) & (Y < self.Y_max_3_5),
            Y / 3,
            np.where(
                (Y >= self.Y_max_3_5) & (Y < self.Y_max),
                (Y * 2) - self.Y_max,
                Y
            )
        )
        Y_trans_uint8 = Y_trans.astype(np.uint8)
        gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (13, 13), 2)
        return gaussian_image
    
    def yuv_auto_canny(self, image, sigma=0.33):
        v = np.median(image)
        lower2 = int(min(255, (1.0 + sigma) * v))
        lower = int(max(50, lower2))
        upper = int(min(255, (2.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        return edged

    def filter_contours_with_hough(self, contours, lines, thickness=10):
        """
        허프 직선의 크기(영역)와 겹치는 컨투어만 필터링
        """
        filtered_contours = []

        for contour in contours:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 직선의 영역 계산 (직선을 두껍게 설정)
                line_region = np.array([
                    [x1 - thickness, y1 - thickness],
                    [x1 + thickness, y1 + thickness],
                    [x2 + thickness, y2 + thickness],
                    [x2 - thickness, y2 - thickness]
                ], dtype=np.int32)

                # 컨투어가 직선 영역과 겹치는지 확인
                for point in contour:
                    px, py = int(point[0][0]), int(point[0][1])  # (x, y) 추출 및 정수 변환
                    if cv2.pointPolygonTest(line_region, (px, py), False) >= 0:
                        filtered_contours.append(contour)
                        break
        return filtered_contours

    def process_images(self):
        for image_file in self.image_files:
            frame = cv2.imread(image_file)
            if frame is None:
                print(f"이미지를 읽을 수 없습니다: {image_file}")
                continue

            frame_resized = frame[:, self.width // 2:self.width]
            height, width, _ = frame_resized.shape
            resized_image = cv2.resize(frame_resized, (width // 2, height // 2))

            base_hough = np.zeros_like(resized_image)

            YUV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
            Y, _, _ = cv2.split(YUV)
            Y_frame = self.YUV_transform(Y)
            auto_canny = self.yuv_auto_canny(Y_frame)

            base_contours = np.zeros_like(resized_image)
            contours, _ = cv2.findContours(auto_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            lines = cv2.HoughLinesP(auto_canny, rho=1, theta=np.pi / 180, threshold=30, minLineLength=10, maxLineGap=50)

            filtered_lines = []
            if lines is not None:
                self.angle_list.clear()
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    self.angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))

                    # 각도 필터링 (20~30도)
                    if 20 <= self.angle <= 30:
                        filtered_lines.append([[x1, y1, x2, y2]])
                        cv2.line(base_hough, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        self.angle_list.append(self.angle)

                print("필터링된 각도 리스트:", self.angle_list)

                if filtered_lines:
                    # 허프 직선 영역과 겹치는 컨투어 필터링
                    filtered_contours = self.filter_contours_with_hough(contours, filtered_lines)

                    # 필터링된 컨투어 시각화
                    for contour in filtered_contours:
                        cv2.drawContours(base_contours, [contour], -1, (255, 255, 255), 2)

            cv2.imshow("Contours with Hough", base_contours)
            cv2.imshow("Hough Lines", base_hough)
            cv2.imshow("Auto Canny", auto_canny)
            cv2.imshow("Resized Image", resized_image)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                continue

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_left'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_right'
    image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_straight'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_left'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_right'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_straight'
    detector = LaneDetector(image_dir)
    detector.process_images()
