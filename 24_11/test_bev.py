

# import cv2
# import numpy as np
# import os
# import math
# from collections import deque


# class LaneDetector:
#     def __init__(self, image_dir):
#         # 기존 코드와 동일
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
#         self.angle_data = []
#         self.previous_x = []
#         self.angle = 0
#         self.Y_max = 0
#         self.Y_max_3_5 = 0
#         self.angle = 0
#         self.angle_list = []
#         self.angle_add = []

#     def YUV_transform(self, Y):
#         # 기존 코드와 동일
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
#         # 기존 코드와 동일
#         v = np.median(image)
#         lower2 = int(min(255, (1.0 + sigma) * v))
#         lower = int(max(50, lower2))
#         upper = int(min(255, (2.0 + sigma) * v))
#         edged = cv2.Canny(image, lower, upper)
#         return edged

#     def avg(self, data):
#         return sum(data) / len(data)

#     def fit_curve(self, points):
#         """
#         선의 중점 데이터를 기반으로 2차 곡선을 피팅
#         """
#         if len(points) < 3:
#             print("곡선 피팅에 충분한 데이터가 없습니다.")
#             return None
        
#         # 중점 포인트를 기반으로 x, y 좌표 분리
#         x_points = [p[0] for p in points]
#         y_points = [p[1] for p in points]
        
#         # 다항식 피팅 (2차)
#         fit = np.polyfit(y_points, x_points, 2)
#         return fit

#     def calculate_direction(self, fit, y_max):
#         """
#         피팅된 곡선을 기반으로 차량 진행 방향 판단
#         """
#         if fit is None:
#             return "직선 또는 차선 없음"

#         # 곡선의 기울기 계산
#         slope_bottom = 2 * fit[0] * y_max + fit[1]  # 아래쪽에서의 기울기
#         if slope_bottom > 0:
#             return "오른쪽으로 회전"
#         elif slope_bottom < 0:
#             return "왼쪽으로 회전"
#         else:
#             return "직진"

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

#             height, width = resized_image.shape[:2]

#             YUV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
#             Y, _, _ = cv2.split(YUV)
#             Y_frame = self.YUV_transform(Y)
#             auto_canny = self.yuv_auto_canny(Y_frame)

#             # p1 = [75, 100]
#             # p2 = [165, 150]
#             # p3 = [142, 100]
#             # p4 = [295, 150]
#             # 기존에서 x축 -45씩
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

#             white_pixels = np.column_stack(np.where(bev_img == 255))
#             start_point = white_pixels.mean(axis=0).astype(int)

#             cv2.circle(bev_img, tuple(start_point[::-1]), radius = 10, color = (50, 50, 50), thickness = -1)
            
#             # 버드아이뷰 출력
#             cv2.imshow('bev_img', bev_img)

            
#             # Draw the BEV rectangle
#             pts = np.array([p1, p2, p4, p3], np.int32)
#             pts = pts.reshape((-1, 1, 2)) 
#             cv2.polylines(auto_canny, [pts], isClosed=True, color=(255, 255, 255), thickness=2) # bev img

#             # 출력
#             cv2.imshow("Auto Canny", auto_canny)
#             cv2.imshow("resized_image", resized_image)

#             key = cv2.waitKey(0) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('n'):
#                 continue

#         cv2.destroyAllWindows()
#         average = self.avg(self.angle_add)
#         print(average)


# if __name__ == "__main__":
#     ### 유진님이 주신 사진
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_left'
#     image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_right'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_straight'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_left'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_right'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_straight'
#     detector = LaneDetector(image_dir)
#     detector.process_images()

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////222222222ㅈㅂㅂㅂㅈㅈㅈㅈㅈㅈㅈㅈㅈ
# 이게 좌/우 판단도 하고 거의 최종임

import cv2
import numpy as np
import os
import time

# start_T = time.time()
# end_T = time.time()
# print(end_T - start_T)
# global start_T, end_T

#좌/진/우 판단 변수
count_num=[0,0,0]

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
    

    # 0.0028 최고 기준
    def process_images(self):
        
        for image_file in self.image_files:
            frame = cv2.imread(image_file)
            if frame is None:
                print(f"이미지를 읽을 수 없습니다: {image_file}")
                continue
            h, w, _ = frame.shape
            orignal_resized = cv2.resize(frame,(w // 2, h // 2))

            frame_resized = frame[:, self.width // 2:self.width]
            height, width, _ = frame_resized.shape
            resized_image = cv2.resize(frame_resized, (width // 2, height // 2))
            height, width = resized_image.shape[:2]

            # YUV 변환 및 Canny Edge Detection
            YUV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
            Y, _, _ = cv2.split(YUV)
            Y_frame = self.YUV_transform(Y)
            auto_canny = self.yuv_auto_canny(Y_frame)

            # 버드아이뷰를 위한 사다리꼴 영역 설정 (전기관 1층)
            # x , y
            # 1 3 
            # 2 4  

            # 1번 많이 오른쪽 
            p1 = [55, 100]
            p2 = [145, 150]
            p3 = [150, 100]
            p4 = [275, 150]

            ## 2번 살짝 오른쪽
            # p1 = [45, 100]
            # p2 = [130, 150]
            # p3 = [140, 100]
            # p4 = [260, 150]

            # ## 3번 기존 코드
            # p1 = [40, 100]
            # p2 = [120, 150]
            # p3 = [135, 100]
            # p4 = [250, 150]
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

            # Convexity 기반 V자형 노이즈 제거
            contours, _ = cv2.findContours(bev_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = []

            for contour in contours:
                hull = cv2.convexHull(contour)
                contour_area = cv2.contourArea(contour)
                hull_area = cv2.contourArea(hull)

                if hull_area > 0:
                    convexity = contour_area / hull_area
                    if convexity > 0.3:  # Convexity 임계값 설정 #0.5 : 차선 사라짐 #0.4 : 잘 나오는 듯 보이다가 차선이 한번씩 사라짐 #0.3 : best인듯
                        filtered_contours.append(contour)


            # 필터링된 Blob 시각화
            filtered_bev_img = np.zeros_like(bev_img)
            cv2.drawContours(filtered_bev_img, filtered_contours, -1, 255, thickness=cv2.FILLED)

            # 중심점을 계산
            non_zero_points = np.argwhere(filtered_bev_img > 0)  # 비어 있지 않은 픽셀 좌표
            if len(non_zero_points) > 0: # 픽셀값이 한개라도 있을 때 실행
                start_point = np.mean(non_zero_points, axis=0).astype(int) # 중심값을 찾음
                cv2.circle(filtered_bev_img, tuple(start_point[::-1]), radius=10, color=(50,50,50), thickness=-1)

                # 1/3 지점, 2/3 지점 계산 코드
                center_x_1_3 = filtered_bev_img.shape[1] / 3 
                center_x_2_3 = center_x_1_3 * 2

                if start_point[1] < center_x_1_3:  # 중심점이 이미지 1/3 왼쪽에 위치
                    print("좌회전")
                    count_num[0]=count_num[0]+1

                elif center_x_2_3 < start_point[1] < filtered_bev_img.shape[1] :  # 중심점이 이미지 2/3 오른쪽에 위치
                    print("우회전")
                    count_num[2]=count_num[2]+1
                else :
                    print("직진")
                    count_num[1]=count_num[1]+1
                

            else: # 비어 있는 이미지 일 때
                print("비어 있는 이미지입니다. 중심점을 계산할 수 없습니다.")
            if bev_img is None or bev_img.size == 0: # 이미지가 아니거나 사이즈가 없는, 즉 이미지가 아닐 때
                print("BEV 이미지 생성 실패")
                continue

            # 사다리꼴 영역을 원본 이미지에 시각화
            pts = np.array([p1, p2, p4, p3], np.int32).reshape((-1, 1, 2))
            cv2.polylines(auto_canny, [pts], isClosed=True, color=(255, 255, 255), thickness=2)

            # 결과 출력
            cv2.imshow("Auto Canny", auto_canny)
            cv2.imshow("Resized Image", resized_image)

            # BEV 이미지 출력
            cv2.imshow('bev_img', bev_img)

            # convexity
            cv2.imshow('filtered_bev_img', filtered_bev_img)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # 'q' 키로 종료
                break
            elif key == ord('n'):  # 'n' 키로 다음 이미지로 이동
                continue
        
        cv2.destroyAllWindows()
        print(f'\n좌회전 = {count_num[0]}\n직진 = {count_num[1]}\n우회전 = {count_num[2]}')


if __name__ == "__main__":
    # 이미지 디렉토리 설정
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_left'
            # p1 = [55, 100]
            # p2 = [145, 150]
            # p3 = [150, 100]
            # p4 = [275, 150] # 1번
            # 좌회전 = 29
            # 직진 = 46
            # 우회전 = 0 : 75장 이미지에 대해 모두 검출

    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_right'
            # p1 = [40, 100]
            # p2 = [120, 150]
            # p3 = [135, 100]
            # p4 = [250, 150] # 3번 
            # 좌회전 = 1
            # 직진 = 38
            # 우회전 = 130 : 191장 중 22장은 출력 안되는 이미지 - 179장 이미지에 대해 1장의 오차율

    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_straight'
            # p1 = [45, 100]
            # p2 = [130, 150]
            # p3 = [140, 100]
            # p4 = [260, 150] # 2번 
            # 좌회전 = 0
            # 직진 = 18
            # 우회전 = 0 : 18장 모두 검출

    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_left'
            # p1 = [45, 100]
            # p2 = [130, 150]
            # p3 = [140, 100]
            # p4 = [260, 150] # 2번 
            # 좌회전 = 112
            # 직진 = 54
            # 우회전 = 1 : 167장 중 1장의 오차율

    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_right'
            # p1 = [45, 100]
            # p2 = [130, 150]
            # p3 = [140, 100]
            # p4 = [260, 150] # 2번 
            # 좌회전 = 1
            # 직진 = 51
            # 우회전 = 139 : 191장 중 1장의 오차율

    image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_straight' 
            # p1 = [55, 100]
            # p2 = [145, 150]
            # p3 = [150, 100]
            # p4 = [275, 150] # 1번 
            # 좌회전 = 0
            # 직진 = 106
            # 우회전 = 5 : 111 중 5장의 오차율

    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/load_image_2' #테스트 해보는 이미지 파일
    
    # 전기관 1층 이미지
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1119_1cmd/1119_white_st/st'

    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115'


    detector = LaneDetector(image_dir)
    detector.process_images()



  # 이건 그냥 버드아이뷰에 중심값 도출하는 코드
            # 흰 픽셀 좌표 추출
            # white_pixels = np.column_stack(np.where(bev_img == 255))
            # if white_pixels.size == 0:
                # print("버드아이뷰 이미지에서 흰 픽셀을 찾을 수 없습니다.")
                # continue

            # # 중심점 계산 및 시각화
            # start_point = white_pixels.mean(axis=0).astype(int)
            # cv2.circle(bev_img, tuple(start_point[::-1]), radius=10, color=(50, 50, 50), thickness=-1)
            # print(start_point[1]) # x의 값

            # 중심점을 기준으로 왼쪽은 좌회전 / 오른쪽은 직진 또는 우회전 -> 여기서 오른쪽 아래만 우회전



            # center_x = filtered_bev_img.shape[1] // 2  # 이미지의 너비 중심값
            # center_y = filtered_bev_img.shape[0] // 2  # 이미지의 높이 중심값

            # if start_point[0] > center_y:  # 중심점이 이미지 아래쪽에 위치
            #     if start_point[1] < center_x:  # 중심점이 왼쪽
            #         print("좌회전")
            #     elif start_point[1] > center_x:  # 중심점이 오른쪽
            #         print("우회전")
            #     else:  # 중심점이 중앙
            #         print("//직진")
            # else:  # 중심점이 이미지 위쪽에 위치
            #     if start_point[1] < center_x:  # 중심점이 왼쪽
            #         print("좌회전")
            #     else :
            #         print("직진")



# def blob_properties(contours):
#   cont_props= []
#   i = 0
#   for cnt in contours:
#     area= cv2.contourArea(cnt)
#     perimeter = cv2.arcLength(cnt,True)
#     convexity = cv2.isContourConvex(cnt)
#     x1,y1,w,h = cv2.boundingRect(cnt)
#     x2 = x1+w
#     y2 = y1+h
#     aspect_ratio = float(w)/h
#     rect_area = w*h
#     extent = float(area)/rect_area
#     hull = cv2.convexHull(cnt)
#     hull_area = cv2.contourArea(hull)
#     solidity = float(area)/hull_area
#     (xa,ya),(MA,ma),angle = cv2.fitEllipse(cnt)
#     rect = cv2.minAreaRect(cnt)
#     (xc,yc),radius = cv2.minEnclosingCircle(cnt)
#     ellipse = cv2.fitEllipse(cnt)
#     rows,cols = img.shape[:2]
#     [vx,vy,xf,yf] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
#     lefty = int((-xf*vy/vx) + yf)
#     righty = int(((cols-xf)*vy/vx)+yf)
#     # Add parameters to list
#     add = i+1, area, round(perimeter, 1), convexity, round(aspect_ratio, 3), round(extent, 3), w, h, round(hull_area, 1), round(angle, 1), x1, y1, x2, y2,round(radius, 6), xa, ya, xc, yc, xf[0], yf[0], rect, ellipse, vx[0], vy[0], lefty, righty
#     cont_props.append(add)
#     i += 1

#   return cont_props

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# import cv2
# import numpy as np
# import os

# class LaneDetector:
#     def __init__(self, image_dir):
#         if not os.path.exists(image_dir):
#             raise ValueError("이미지 디렉토리가 잘못되었습니다.")

#         self.image_dir = image_dir
#         self.image_files = sorted([
#             os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))
#         ])

#         if len(self.image_files) == 0:
#             raise ValueError("이미지 파일이 없습니다.")

#         test_image = cv2.imread(self.image_files[0])
#         if test_image is None:
#             raise ValueError("이미지를 열 수 없습니다.")

#         self.width = test_image.shape[1]
#         self.height = test_image.shape[0]

#     def process_images(self):
#         for image_file in self.image_files:
#             frame = cv2.imread(image_file)
#             if frame is None:
#                 print(f"이미지를 읽을 수 없습니다: {image_file}")
#                 continue

#             frame_resized = frame[:, self.width // 2:self.width]
#             height, width, _ = frame_resized.shape
#             resized_image = cv2.resize(frame_resized, (width // 2, height // 2))

#             # YUV 변환 및 Canny Edge Detection
#             YUV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
#             Y, _, _ = cv2.split(YUV)
#             gaussian_image = cv2.GaussianBlur(Y, (13, 13), 2)
#             edged = cv2.Canny(gaussian_image, 50, 150)

#             # Bird's Eye View 설정
#             p1, p2, p3, p4 = [40, 100], [120, 150], [135, 100], [250, 150]
#             corner_points_arr = np.float32([p1, p2, p3, p4])

#             image_p1, image_p2, image_p3, image_p4 = [0, 0], [0, height], [width, 0], [width, height]
#             image_params = np.float32([image_p1, image_p2, image_p3, image_p4])

#             mat = cv2.getPerspectiveTransform(corner_points_arr, image_params)
#             bev_img = cv2.warpPerspective(edged, mat, (width, height))

#             # BEV 이미지 이진화 처리
#             _, bin_img = cv2.threshold(bev_img, 70, 255, cv2.THRESH_BINARY)
#             bev_img = bin_img

#             # Convexity 기반 V자형 노이즈 제거
#             contours, _ = cv2.findContours(bev_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             filtered_contours = []

#             for contour in contours:
#                 hull = cv2.convexHull(contour)
#                 contour_area = cv2.contourArea(contour)
#                 hull_area = cv2.contourArea(hull)

#                 if hull_area > 0:
#                     convexity = contour_area / hull_area
#                     if convexity > 0.3:  # Convexity 임계값 설정
#                         filtered_contours.append(contour)

#             # 필터링된 Blob 시각화
#             filtered_bev_img = np.zeros_like(bev_img)
#             cv2.drawContours(filtered_bev_img, filtered_contours, -1, 255, thickness=cv2.FILLED)

#             # 중심점을 계산
#             non_zero_points = np.argwhere(filtered_bev_img > 0)  # 비어 있지 않은 픽셀 좌표
#             if len(non_zero_points) > 0:
#                 start_point = np.mean(non_zero_points, axis=0).astype(int)
#                 cv2.circle(filtered_bev_img, tuple(start_point[::-1]), radius=10, color=(50,50,50), thickness=-1)
#             else:
#                 print("비어 있는 이미지입니다. 중심점을 계산할 수 없습니다.")


#             # 중심점을 기준으로 왼쪽은 좌회전 / 오른쪽은 직진 또는 우회전 -> 여기서 오른쪽 아래만 우회전
#             center_x_1_3 = filtered_bev_img.shape[1] / 3 
#             center_x_2_3 = center_x_1_3 * 2

#             if start_point[1] < center_x_1_3:  # 중심점이 이미지 1/3 왼쪽에 위치
#                 print("좌회전")
#             elif start_point[1] > center_x_2_3:  # 중심점이 이미지 2/3 오른쪽에 위치
#                 print("우회전")
#             else :
#                 print("직진")

#             # 결과 시각화
#             cv2.imshow("Original Image", resized_image)
#             cv2.imshow("Bird's Eye View", bev_img)
#             cv2.imshow("Filtered BEV", filtered_bev_img)

#             key = cv2.waitKey(0) & 0xFF
#             if key == ord('q'):  # 'q' 키로 종료
#                 break

#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_left'  # 이미지 디렉토리 설정
#     image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_right'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_straight'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_left'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_right'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_straight'  #조금 먼 감이 없지않아 있다.

#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/load_image_2' 
    
#     # 전기관 1층 이미지
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1119_1cmd/1119_white_st/st'
#     detector = LaneDetector(image_dir)
#     detector.process_images()