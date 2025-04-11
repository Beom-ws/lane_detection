# import cv2
# import numpy as np
# import os
# from sklearn.linear_model import RANSACRegressor


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
#         lower2 = int(min(255, (1.0 + sigma) * v))
#         lower = int(max(50, lower2))
#         upper = int(min(255, (2.0 + sigma) * v))
#         edged = cv2.Canny(image, lower, upper)
#         return edged

#     def detect_curve_with_ransac(self, edges, poly_degree=2):
#         points = np.column_stack(np.where(edges > 0))
#         if len(points) < poly_degree + 1:
#             print("검출된 포인트가 부족하여 RANSAC 적용 불가.")
#             return None, None

#         x = points[:, 1].reshape(-1, 1)
#         y = points[:, 0]

#         ransac = RANSACRegressor()
#         try:
#             ransac.fit(x, y)
#             inlier_mask = ransac.inlier_mask_
#             poly_fit = np.polyfit(x[inlier_mask].flatten(), y[inlier_mask], poly_degree)
#             poly_curve = np.poly1d(poly_fit)
#             return poly_curve, inlier_mask
#         except ValueError as e:
#             print(f"RANSAC 실행 중 오류 발생: {e}")
#             return None, None

#     def process_images(self):
#         for image_file in self.image_files:
#             frame = cv2.imread(image_file)
#             if frame is None:
#                 print(f"이미지를 읽을 수 없습니다: {image_file}")
#                 continue

#             frame_resized = frame[:, self.width // 2:self.width]
#             height, width, _ = frame_resized.shape
#             resized_image = cv2.resize(frame_resized, (width // 2, height // 2))

#             height, width = resized_image.shape[:2]

#             YUV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
#             Y, _, _ = cv2.split(YUV)
#             Y_frame = self.YUV_transform(Y)
#             auto_canny = self.yuv_auto_canny(Y_frame)

#             # BEV 적용
#             p1 = [30, 100]
#             p2 = [120, 150]
#             p3 = [120, 100]
#             p4 = [250, 150]
#             corner_points_arr = np.float32([p1, p2, p3, p4])

#             image_p1 = [0, 0]
#             image_p2 = [0, height]
#             image_p3 = [width, 0]
#             image_p4 = [width, height]
#             image_params = np.float32([image_p1, image_p2, image_p3, image_p4])

#             mat = cv2.getPerspectiveTransform(corner_points_arr, image_params)
#             bev_img = cv2.warpPerspective(auto_canny, mat, (width, height))
#             _, bin_img = cv2.threshold(bev_img, 70, 255, cv2.THRESH_BINARY)
#             bev_img = bin_img

#             if bev_img is None or bev_img.size == 0:
#                 print("BEV 이미지 생성 실패")
#                 continue

#             white_pixels = np.column_stack(np.where(bev_img == 255))
#             if white_pixels.size == 0:
#                 print("버드아이뷰 이미지에서 흰 픽셀을 찾을 수 없습니다.")
#                 continue
            
#             start_point = white_pixels.mean(axis=0).astype(int)
#             cv2.circle(bev_img, tuple(start_point[::-1]), radius=10, color=(50, 50, 50), thickness=-1)

#             poly_curve, inlier_mask = self.detect_curve_with_ransac(bev_img)

#             base_curve = np.zeros_like(bev_img)
#             if poly_curve is not None:
#                 for x in range(width):
#                     y = int(poly_curve(x))
#                     if 0 <= y < height:
#                         cv2.circle(base_curve, (x, y), 2, 255, -1)

#             # 사다리꼴 영역을 원본 이미지에 시각화
#             pts = np.array([p1, p2, p4, p3], np.int32).reshape((-1, 1, 2))
#             cv2.polylines(auto_canny, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
            
#             cv2.imshow('BEV Image', bev_img)
#             cv2.imshow('base_curve', base_curve)
#             cv2.imshow("Auto Canny", auto_canny)
#             cv2.imshow("Resized Image", resized_image)

#             key = cv2.waitKey(0) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('n'):
#                 continue

#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_straight'
#     detector = LaneDetector(image_dir)
#     detector.process_images()

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# 양쪽 차선으로 코드 수정

# 이게 좌/우 판단도 하고 거의 최종임

import cv2
import numpy as np
import os
import time

# deque = 10개의 값을 평균내서 조향

# start_T = time.time()
# end_T = time.time()
# print(end_T - start_T)
# global start_T, end_T

class LaneDetector:
    def __init__(self, image_dir):

        # 이미지 디렉토리 확인 및 로드
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


    # 0.0006초
    def YUV_transform(self, Y): 
        # YUV 감마 보정 및 Gaussian Blur 적용
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
    
    
    # 0.0003초
    def yuv_auto_canny(self, image, sigma=0.33):
        # Auto Canny Edge Detection
        v = np.median(image)
        lower2 = int(min(255, (1.0 + sigma) * v))
        lower = int(max(50, lower2))
        upper = int(min(255, (2.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        return edged
    

    # 0.0179초
    def process_images(self):
        for image_file in self.image_files:
            frame = cv2.imread(image_file)
            if frame is None:
                print(f"이미지를 읽을 수 없습니다: {image_file}")
                continue

            # eframe_resized = frame[:, self.width // 2:self.width]
            frame_resized = frame[:, self.width//2 : self.width]
            height, width, _ = frame_resized.shape
            resized_image = cv2.resize(frame_resized, (width // 2, height // 2))

            height, width = resized_image.shape[:2]

            # YUV 변환 및 Canny Edge Detection
            YUV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
            Y, _, _ = cv2.split(YUV)
            Y_frame = self.YUV_transform(Y)
            auto_canny = self.yuv_auto_canny(Y_frame)

            # 버드아이뷰를 위한 사다리꼴 영역 설정
            # x , y
            # 1 3 
            # 2 4  
            # p1 = [40, 100]
            # p2 = [120, 150]
            # p3 = [135, 100]
            # p4 = [250, 150]

            p1 = [width*0.125, height*0.28]
            p2 = [width*0.375, height*0.42]
            p3 = [width*0.42, height*0.28]
            p4 = [width*0.78, height*0.42]
            corner_points_arr = np.float32([p1, p2, p3, p4])

            # BEV 투영 결과 이미지를 이미지 크기로 변환
            image_p1 = [0, 0]
            image_p2 = [0, height]
            image_p3 = [width, 0]
            image_p4 = [width, height]
            image_params = np.float32([image_p1, image_p2, image_p3, image_p4])


            # Perspective Transform
            mat = cv2.getPerspectiveTransform(corner_points_arr, image_params)
            bev_img = cv2.warpPerspective(auto_canny, mat, (width, height))
            
            # BEV 이미지 이진화 처리
            _, bin_img = cv2.threshold(bev_img, 70, 255, cv2.THRESH_BINARY)
            bev_img = bin_img

            if bev_img is None or bev_img.size == 0:
                print("BEV 이미지 생성 실패")
                continue

            # # 이미지의 높이 중심값
            h, w = bev_img.shape
            bev_img_h = bev_img[0:h//2,:]
            bev_img_b = bev_img[h//2:h,:]
 
            
            ##################################################################
            # 이미지 하단의 흰색 값 처리
            # 흰 픽셀 좌표 추출
            white_pixels_b = np.column_stack(np.where(bev_img_b == 255))
            if white_pixels_b.size == 0:
                print("버드아이뷰 이미지에서 흰 픽셀을 찾을 수 없습니다.")
                continue
            
            # 중심점 계산 및 시각화
            start_point_b = white_pixels_b.mean(axis=0).astype(int)
            cv2.circle(bev_img_b, tuple(start_point_b[::-1]), radius=10, color=(50, 50, 50), thickness=-1)

            cv2.imshow('bev_img_b',bev_img_b)

            ############################################################################3
            # 이미지 상단의 흰색 값 처리
            # 흰 픽셀 좌표 추출
            white_pixels_h = np.column_stack(np.where(bev_img_h == 255))
            if white_pixels_h.size == 0:
                print("버드아이뷰 이미지에서 흰 픽셀을 찾을 수 없습니다.")
                continue
            
            # 중심점 계산 및 시각화
            start_point_h = white_pixels_h.mean(axis=0).astype(int)
            cv2.circle(bev_img_h, tuple(start_point_h[::-1]), radius=10, color=(50, 50, 50), thickness=-1)

            cv2.imshow('bev_img_h',bev_img_h)
            ############################################################################3


            high_x = start_point_h[1]
            bottom_x = start_point_b[1]


            print(high_x - bottom_x)

            if high_x - bottom_x > 10:
                print('Right\n')
            elif high_x - bottom_x < -10:
                print('Left\n')
            else:
                print('Straight\n')






            # # 중심점을 기준으로 왼쪽은 좌회전 / 오른쪽은 직진 또는 우회전 -> 여기서 오른쪽 아래만 우회전
            # center_x_1_3 = bev_img.shape[1] / 3 
            # center_x_2_3 = center_x_1_3 * 2 

            # # .shape[1] : x의 값
            # # .shape[0] : y의 값

            # if start_point_b[1] < center_x_1_3 : # 밑에 왼쪽
            #     b_mode = 1
            # elif start_point_b[1] > center_x_2_3 : # 밑에 오른쪽
            #     b_mode = 3
            # else :                                  #밑에 중간
            #     b_mode = 2
            
            # if start_point_h[1] < center_x_1_3 : # 위에 왼쪽
            #     h_mode = 1
            # elif start_point_h[1] > center_x_2_3 : # 위에 오른쪽
            #     h_mode = 3
            # else :                                  #위에 중간
            #     h_mode = 2

            # if b_mode == h_mode:
            #     if (b_mode == 2) & (h_mode == 2) :
            #         print('직진')
            #     else:
            #         print('df')
            # elif (b_mode > h_mode) or (b_mode == 1 and h_mode==Noㅇne ): #(b_mode > h_mode) 그리고 반대경우도 : 경계선 노이즈가 나오면 애매하게 우회전/좌회전이 나옴 다시생각해보기
            #     print('좌회전')
            # elif (b_mode < h_mode) or (b_mode == 3 and b_mode==None):
            #     print('우회전')
            # else :
            #     print('오류')




            # b_mode = 0
            # h_mode = 0

            # 사다리꼴 영역을 원본 이미지에 시각화
            pts = np.array([p1, p2, p4, p3], np.int32).reshape((-1, 1, 2))
            cv2.polylines(auto_canny, [pts], isClosed=True, color=(255, 255, 255), thickness=2)

            # 결과 출력
            cv2.imshow("Auto Canny", auto_canny)
            cv2.imshow("Resized Image", resized_image)
            cv2.imshow("Y_frame", Y_frame)

            # BEV 이미지 출력
            # cv2.imshow('BEV Image', bev_img)


            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # 'q' 키로 종료
                break
            elif key == ord('n'):  # 'n' 키로 다음 이미지로 이동
                continue

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 이미지 디렉토리 설정
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_left'
    image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_right'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_straight'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_left'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_right'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_straight'  #조금 먼 감이 없지않아 있다.

    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/load_image_2' 
    
    # 전기관 1층 이미지
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1119_1cmd/1119_white_st/st'

    detector = LaneDetector(image_dir)
    detector.process_images()


