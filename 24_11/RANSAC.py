import cv2
import numpy as np
import os
from sklearn.linear_model import RANSACRegressor
import time


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

    def YUV_transform(self, Y):
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
        gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (13, 13), 2)
        return gaussian_image

    def yuv_auto_canny(self, image, sigma=0.33):
        v = np.median(image)
        lower = int(max(50, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        return edged


class LaneDetectorWithRANSAC(LaneDetector):
    global start_t, end_t

    def detect_curve_with_ransac(self, edges, poly_degree=2):
        points = np.column_stack(np.where(edges > 0))
        if len(points) < poly_degree + 1:
            print("검출된 포인트가 부족하여 RANSAC 적용 불가.")
            return None, None

        x = points[:, 1].reshape(-1, 1)
        y = points[:, 0]

        ransac = RANSACRegressor()
        try:
            ransac.fit(x, y)
            inlier_mask = ransac.inlier_mask_
            poly_fit = np.polyfit(x[inlier_mask].flatten(), y[inlier_mask], poly_degree)
            poly_curve = np.poly1d(poly_fit)
            return poly_curve, inlier_mask
        except ValueError as e:
            print(f"RANSAC 실행 중 오류 발생: {e}")
            return None, None

        

    def process_images(self):
        for image_file in self.image_files:
            frame = cv2.imread(image_file)
            if frame is None:
                print(f"이미지를 읽을 수 없습니다: {image_file}")
                continue

            frame_resized = frame[:, self.width // 2:self.width]
            height, width, _ = frame_resized.shape

            resized_image = cv2.resize(frame_resized, (width // 2, height // 2))

            YUV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
            Y, _, _ = cv2.split(YUV)

            Y_frame = self.YUV_transform(Y)
            auto_canny = self.yuv_auto_canny(Y_frame)
            start_t = time.time()
            poly_curve, inlier_mask = self.detect_curve_with_ransac(auto_canny)

            base_curve = np.zeros_like(auto_canny)
            if poly_curve is not None:
                for x in range(width // 2):
                    y = int(poly_curve(x))
                    if 0 <= y < height // 2:
                        cv2.circle(base_curve, (x, y), 2, 255, -1)
            end_t = time.time()

            cv2.imshow("Auto Canny", auto_canny)
            cv2.imshow("Detected Curve (RANSAC)", base_curve)
            cv2.imshow("Original Image", resized_image)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                continue

        cv2.destroyAllWindows()
        print(end_t - start_t)


if __name__ == "__main__":
    image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_left'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_right'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_straight'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_left'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_right'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_straight'
    detector = LaneDetectorWithRANSAC(image_dir)
    detector.process_images()


##mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
## RANSAC에서 차선이 맞다고 판단한 부분, 아니라고 판단한 부분을 시각화 하는 코드

# import cv2
# import numpy as np
# import os
# from sklearn.linear_model import RANSACRegressor
# import time


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

#     def YUV_transform(self, Y):
#         Y_max = np.max(Y)
#         Y_max_3_5 = Y_max * (3 / 5)

#         Y_trans = np.where(
#             (Y > 0) & (Y < Y_max_3_5),
#             Y / 3,
#             np.where(
#                 (Y >= Y_max_3_5) & (Y < Y_max),
#                 (Y * 2) - Y_max,
#                 Y
#             )
#         )
#         Y_trans_uint8 = Y_trans.astype(np.uint8)
#         gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (13, 13), 2)
#         return gaussian_image

#     def yuv_auto_canny(self, image, sigma=0.33):
#         v = np.median(image)
#         lower = int(max(50, (1.0 - sigma) * v))
#         upper = int(min(255, (1.0 + sigma) * v))
#         edged = cv2.Canny(image, lower, upper)
#         return edged


# class LaneDetectorWithRANSAC(LaneDetector):
#     global start_t, end_t

#     def detect_curve_with_ransac(self, edges, poly_degree=2):
#         points = np.column_stack(np.where(edges > 0))
#         if len(points) < poly_degree + 1:
#             print("검출된 포인트가 부족하여 RANSAC 적용 불가.")
#             return None, None, None

#         x = points[:, 1].reshape(-1, 1)
#         y = points[:, 0]

#         ransac = RANSACRegressor()
#         try:
#             ransac.fit(x, y)
#             inlier_mask = ransac.inlier_mask_
#             outlier_mask = ~inlier_mask  # 인라이어가 아닌 데이터 (아웃라이어)

#             # 다항식 피팅
#             poly_fit = np.polyfit(x[inlier_mask].flatten(), y[inlier_mask], poly_degree)
#             poly_curve = np.poly1d(poly_fit)

#             return poly_curve, inlier_mask, points
#         except ValueError as e:
#             print(f"RANSAC 실행 중 오류 발생: {e}")
#             return None, None, None


        

#     def process_images(self):
#         for image_file in self.image_files:
#             frame = cv2.imread(image_file)
#             if frame is None:
#                 print(f"이미지를 읽을 수 없습니다: {image_file}")
#                 continue

#             frame_resized = frame[:, self.width // 2:self.width]
#             height, width, _ = frame_resized.shape

#             resized_image = cv2.resize(frame_resized, (width // 2, height // 2))

#             YUV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
#             Y, _, _ = cv2.split(YUV)

#             Y_frame = self.YUV_transform(Y)
#             auto_canny = self.yuv_auto_canny(Y_frame)

#             # RANSAC 실행
#             start_t = time.time()
#             poly_curve, inlier_mask, points = self.detect_curve_with_ransac(auto_canny)
#             end_t = time.time()

#             base_curve = np.zeros_like(auto_canny)
#             color_inlier = np.zeros_like(cv2.cvtColor(auto_canny, cv2.COLOR_GRAY2BGR))
#             color_outlier = color_inlier.copy()

#             if poly_curve is not None and points is not None:
#                 inliers = points[inlier_mask]  # 인라이어 포인트
#                 outliers = points[~inlier_mask]  # 아웃라이어 포인트

#                 # 인라이어와 아웃라이어를 시각적으로 구분
#                 for point in inliers:
#                     cv2.circle(color_inlier, (point[1], point[0]), 2, (0, 255, 0), -1)  # 초록색
#                 for point in outliers:
#                     cv2.circle(color_outlier, (point[1], point[0]), 2, (0, 0, 255), -1)  # 빨간색

#                 # 곡선도 이미지에 표시
#                 for x in range(width // 2):
#                     y = int(poly_curve(x))
#                     if 0 <= y < height // 2:
#                         cv2.circle(base_curve, (x, y), 2, 255, -1)
#             # 시각화
#             cv2.imshow("Auto Canny", auto_canny)
#             cv2.imshow("Detected Curve (RANSAC)", base_curve)
#             cv2.imshow("Inliers (Green)", color_inlier)
#             cv2.imshow("Outliers (Red)", color_outlier)
#             cv2.imshow("Original Image", resized_image)

#             key = cv2.waitKey(0) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('n'):
#                 continue

#     cv2.destroyAllWindows()
#     # print("처리 시간:", end_t - start_t)
# if __name__ == "__main__":
#     image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_left'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_right'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_straight'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_left'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_right'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_straight'
#     detector = LaneDetectorWithRANSAC(image_dir)
#     detector.process_images()
