# import cv2
# import numpy as np
# import os
# import math
# import time
# import matplotlib.pyplot as plt


# class LaneDetector:
#     def __init__(self, video_path):
#         if not os.path.exists(video_path):
#             raise ValueError("Video path does not exist")

#         self.cap = cv2.VideoCapture(video_path)
#         if not self.cap.isOpened():
#             raise ValueError("Cannot open video")

#         self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         self.angle_data = []
#         self.previous_x = []
#         self.angle = 0
#         self.Y_max = 0
#         self.Y_max_3_5 = 0


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

#         gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (13,13) ,2) # 7 -> 13 -> 21 -> 31(별로)

#         return gaussian_image


#     def re_auto_canny(self, image, sigma = 0.33): # 변경. (1.33/2.33) 값의 auto canny
#         global v       
#         v = np.median(image)
#         lower2 = int(min(255, (1.0 + sigma) * v))
#         lower = int(max(50,lower2))
#         upper = int(min(255, (2.0 + sigma) * v))
#         edged = cv2.Canny(image, lower, upper)

#         print('yuv auto \nlower: %d  upper: %d median: %d' % (lower, upper,v))
    
#         return edged

#     def original_auto_canny(self, image, sigma = 0.33): # 기존. (0.67/1.33) 값의 auto canny
#         global v1
#         v1 = np.median(image)
#         lower = int(max(0, (1.0 - sigma) * v1))
#         upper = int(min(255, (1.0 + sigma) * v1))
#         edged = cv2.Canny(image, lower, upper)

#         print('original \nlower: %d  upper: %d median: %d' % (lower, upper,v1))
    
#         return edged

#     def process_video(self):
#         while self.cap.isOpened():
#             ret, frame = self.cap.read()
#             if not ret:
#                 break

#             # frame_resized = frame[0:self.height//3*2, :]
#             frame_resized = frame[self.height//2:self.height,self.width//2:self.width]

#             height, width, _ = frame_resized.shape

#             resized_image = cv2.resize(frame_resized, (width // 2, height // 2))

#             gray_frame = cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)
            
#             original_gaussian = cv2.GaussianBlur(gray_frame, (5,5) ,0)
        

#             # YUV 변환 + 가우시안
#             YUV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
#             Y, _, _ = cv2.split(YUV)
#             Y_frame = self.YUV_transform(Y) #YUV 변환한 이미지

#             ## binary image 만들기
#             kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#             Y_binary2 = (Y_frame >= (np.max(Y_frame)* 0.7)).astype(np.uint8) * 255

#             dilated_image = cv2.dilate(Y_binary2, kernel, iterations=1)
#             Y_binary = cv2.erode(dilated_image, kernel, iterations=1)

#             # # 1. Original image에 auto canny 적용
#             original_auto_canny = self.original_auto_canny(original_gaussian)

#             # # 2. YUV image에 auto canny 적용
#             YUV_auto_canny = self.re_auto_canny(Y_frame)

#             ####################### 반 이
#             frame_resized2 = frame[:,self.width//2:self.width]

#             height2, width2, _ = frame_resized2.shape

#             resized_image2 = cv2.resize(frame_resized, (width2 // 2, height2 // 2))

#             gray_frame2 = cv2.cvtColor(resized_image2,cv2.COLOR_BGR2GRAY)
            
#             original_gaussian2 = cv2.GaussianBlur(gray_frame2, (5,5) ,0)
        

#             # YUV 변환 + 가우시안
#             YUV = cv2.cvtColor(frame_resized2, cv2.COLOR_BGR2YUV)
#             Y, _, _ = cv2.split(YUV)
#             Y_frame2 = self.YUV_transform(Y) #YUV 변환한 이미지

#             ## binary image 만들기
#             kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#             Y_binary3 = (Y_frame2 >= (np.max(Y_frame2)* 0.7)).astype(np.uint8) * 255

#             dilated_image2 = cv2.dilate(Y_binary3, kernel, iterations=1)
#             Y_binary2 = cv2.erode(dilated_image2, kernel, iterations=1)

#             # # 1. Original image에 auto canny 적용
#             original_auto_canny2 = self.original_auto_canny(original_gaussian2)

#             # # 2. YUV image에 auto canny 적용
#             YUV_auto_canny2 = self.re_auto_canny(Y_frame2)
#             cv2.imshow('half',YUV_auto_canny2 )
#             ##########################

#             # 더라기
#             bit_or = cv2.bitwise_or(Y_binary, YUV_auto_canny)
#             dilated_image2 = cv2.dilate(bit_or, kernel, iterations=1)
#             bit_or2 = cv2.erode(dilated_image2, kernel, iterations=1)
#             cv2.imshow('bit_or',bit_or2)


#             # imshow 합치기
#             top_row = np.hstack((original_gaussian, Y_frame))
#             bottom_row = np.hstack((original_auto_canny, YUV_auto_canny))
#             display = np.vstack((top_row, bottom_row))
#             # display = cv2.resize(display,[1600,850])
#             cv2.imshow('display', display)

#             cv2.imshow('binary image', Y_binary)

#             cv2.imshow('orignal',frame )

#             key = cv2.waitKey(0) & 0xFF
#             if key == ord('q'):  # 'q'키를 누르면 종료
#                 break
#             elif key == ord('n'):  # 'n'키를 누르면 다음 이미지로 이동
#                 continue

#         self.cap.release()
#         cv2.destroyAllWindows()
#         print(v)

# if __name__ == "__main__":
#     # video_path = '/home/ubuntu/beom_ws/src/lane_image/video/curve_video.avi' # 그림자 영상
#     video_path = '/home/ubuntu/beom_ws/src/lane_image/video/hwg_curve_video.avi' # curve_video 영상을 보면 auto보다 gamma가 더 잘 나타나는 경우가 있고, 그 외에는 auto가 더 좋음 -> auto + gamma
#     # video_path = '/home/ubuntu/beom_ws/src/lane_image/video/hw_curve_3.avi' # 시작할 때 빛 살짝 튀는 영상
#     # video_path = '/home/ubuntu/beom_ws/src/load_image_1/load_image_1/frame000445.png' # 리모 사진
#     # video_path =  '/home/ubuntu/beom_ws/src/lane_image/1727179667157.png' # 이상한 사진
#     lane_detector = LaneDetector(video_path)
#     lane_detector.process_video()

#################################################################################################################################################################################################################################################3
## 비디오 : 선형회귀 적용 (YUV_auto_canny)
# import cv2
# import numpy as np
# import os
# import math
# import time
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression


# class LaneDetector:
#     def __init__(self, video_path):
#         if not os.path.exists(video_path):
#             raise ValueError("Video path does not exist")

#         self.cap = cv2.VideoCapture(video_path)
#         if not self.cap.isOpened():
#             raise ValueError("Cannot open video")

#         self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         self.angle_data = []
#         self.previous_x = []
#         self.angle = 0
#         self.Y_max = 0
#         self.Y_max_3_5 = 0

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

#     def re_auto_canny(self, image, sigma=0.33):
#         global v
#         v = np.median(image)
#         lower2 = int(min(255, (1.0 + sigma) * v))
#         lower = int(max(50, lower2))
#         upper = int(min(255, (2.0 + sigma) * v))
#         edged = cv2.Canny(image, lower, upper)

#         print('yuv auto \nlower: %d  upper: %d median: %d' % (lower, upper, v))
    
#         return edged

#     def original_auto_canny(self, image, sigma=0.33):
#         global v1
#         v1 = np.median(image)
#         lower = int(max(0, (1.0 - sigma) * v1))
#         upper = int(min(255, (1.0 + sigma) * v1))
#         edged = cv2.Canny(image, lower, upper)

#         print('original \nlower: %d  upper: %d median: %d' % (lower, upper, v1))
    
#         return edged

#     def apply_right_lane_regression(self, edges):
#         edge_points = np.column_stack(np.where(edges > 0))
#         mid_x = edges.shape[1] // 2
#         right_points = edge_points[edge_points[:, 1] >= mid_x]

#         if len(right_points) > 0:
#             right_model = LinearRegression().fit(right_points[:, 1].reshape(-1, 1), right_points[:, 0])
#             right_slope = right_model.coef_[0]
#             right_intercept = right_model.intercept_
#             print(f"오른쪽 차선 기울기: {right_slope}")
#         else:
#             right_slope = None
#             right_intercept = None
#             print("오른쪽 차선 기울기를 계산할 포인트가 부족합니다.")

#         return right_slope, right_intercept

#     def draw_lane_line(self, image, slope, intercept, color=(0, 255, 0), thickness=5):
#         # 선을 그릴 시작점과 끝점을 계산하고 그리기
#         if slope is not None and intercept is not None:
#             y1 = image.shape[0]  # 이미지 하단
#             y2 = int(y1 * 0.6)   # 이미지의 60% 지점

#             x1 = int((y1 - intercept) / slope)
#             x2 = int((y2 - intercept) / slope)

#             cv2.line(image, (x1, y1), (x2, y2), color, thickness)

#     def process_video(self):
#         while self.cap.isOpened():
#             ret, frame = self.cap.read()
#             if not ret:
#                 break

#             # 원본 프레임에서 오른쪽 절반만 사용
#             frame_right = frame[:, self.width//2:self.width]
#             gray_frame = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
#             original_gaussian = cv2.GaussianBlur(gray_frame, (5,5), 0)

#             # YUV 변환 + 가우시안
#             YUV = cv2.cvtColor(frame_right, cv2.COLOR_BGR2YUV)
#             Y, _, _ = cv2.split(YUV)
#             Y_frame = self.YUV_transform(Y)

#             # 이진 이미지 생성
#             kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#             Y_binary2 = (Y_frame >= (np.max(Y_frame) * 0.7)).astype(np.uint8) * 255
#             dilated_image = cv2.dilate(Y_binary2, kernel, iterations=1)
#             Y_binary = cv2.erode(dilated_image, kernel, iterations=1)

#             # Original 및 YUV Canny 적용
#             original_auto_canny = self.original_auto_canny(original_gaussian)
#             YUV_auto_canny = self.re_auto_canny(Y_frame)

#             # 오른쪽 차선만 선형 회귀 적용
#             right_slope, right_intercept = self.apply_right_lane_regression(Y_binary)

#             # 오른쪽 차선을 원본 프레임 오른쪽 절반에 시각화 (파란색 두께 3으로 설정)
#             self.draw_lane_line(frame_right, right_slope, right_intercept, color=(255, 0, 0), thickness=3)

#             # 결과 시각화
#             bit_or = cv2.bitwise_or(Y_binary, YUV_auto_canny)
#             dilated_image2 = cv2.dilate(bit_or, kernel, iterations=1)
#             bit_or2 = cv2.erode(dilated_image2, kernel, iterations=1)
#             cv2.imshow('bit_or', bit_or2)

#             top_row = np.hstack((original_gaussian, Y_frame))
#             bottom_row = np.hstack((original_auto_canny, YUV_auto_canny))
#             display = np.vstack((top_row, bottom_row))
#             cv2.imshow('display', display)
#             cv2.imshow('binary image', Y_binary)

#             # 오른쪽 차선이 포함된 프레임 오른쪽 절반을 출력
#             cv2.imshow('Right Lane Line Visualization', frame_right)

#             key = cv2.waitKey(0) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('n'):
#                 continue

#         self.cap.release()
#         cv2.destroyAllWindows()
#         print(v)


# if __name__ == "__main__":
#     video_path = '/home/ubuntu/beom_ws/src/lane_image/video/hwg_curve_video.avi'
#     lane_detector = LaneDetector(video_path)
#     lane_detector.process_video()

# #############################################################33###############################################################################################################################################################################3
# ## 비디오 : Binary Image에 대해 Hough Transform 적용 
# import cv2
# import numpy as np
# import os
# import math
# import time
# import matplotlib.pyplot as plt


# class LaneDetector:
#     def __init__(self, video_path):
#         if not os.path.exists(video_path):
#             raise ValueError("Video path does not exist")

#         self.cap = cv2.VideoCapture(video_path)
#         if not self.cap.isOpened():
#             raise ValueError("Cannot open video")

#         self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         self.angle_data = []
#         self.previous_x = []
#         self.angle = 0
#         self.Y_max = 0
#         self.Y_max_3_5 = 0

#     # 감마 보정을 1차식으로 변경하고, 이 보정을 Y에 적용 
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

#         # 블러 (13,13) 적용
#         gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (13, 13), 2)
#         return gaussian_image

#     # 중앙값의 1.33/2.33 배 canny의 하향/상향 적용 
#     def re_auto_canny(self, image, sigma=0.33):
#         global v
#         v = np.median(image)
#         lower2 = int(min(255, (1.0 + sigma) * v))
#         lower = int(max(50, lower2))
#         upper = int(min(255, (2.0 + sigma) * v))
#         edged = cv2.Canny(image, lower, upper)
#         print('yuv auto \nlower: %d  upper: %d median: %d' % (lower, upper, v))
#         return edged

#     # 중앙값의 0.67/1.33 배 canny의 하향/상향 적용 
#     def original_auto_canny(self, image, sigma=0.33):
#         global v1
#         v1 = np.median(image)
#         lower = int(max(0, (1.0 - sigma) * v1))
#         upper = int(min(255, (1.0 + sigma) * v1))
#         edged = cv2.Canny(image, lower, upper)
#         print('original \nlower: %d  upper: %d median: %d' % (lower, upper, v1))
#         return edged

#     def process_video(self):
#         while self.cap.isOpened():
#             ret, frame = self.cap.read()
#             if not ret:
#                 break

#             # 오른쪽만 사용하기 위해서 x축의 오른쪽만 짜름
#             frame_resized = frame[:, self.width//2:self.width]
#             height, width, _ = frame_resized.shape
            
#             # 노이즈도 줄이고, 처리 속도를 빠르게 만들기 위해 이미지를 작게 수정
#             resized_image = cv2.resize(frame_resized, (width//2, height//2))

#             # YUV 변환 + 가우시안
#             YUV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
#             Y, _, _ = cv2.split(YUV)
#             Y_frame = self.YUV_transform(Y)

#             # binary image 만들기
#             kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#             Y_binary2 = (Y_frame >= (np.max(Y_frame) * 0.7)).astype(np.uint8) * 255
            
#             # 해 본 결과, 안한거랑 큰 차이 없음 그냥 Y_binary2로 진행해도 될 듯
#             dilated_image = cv2.dilate(Y_binary2, kernel, iterations=1)
#             Y_binary = cv2.erode(dilated_image, kernel, iterations=1)

#             # binary image에 대해 허프 변환 적용
#             lines = cv2.HoughLinesP(Y_binary,rho=1,theta=np.pi / 180,threshold=100,minLineLength=50,maxLineGap=150)

#             # 검출된 직선을 frame_resized에 그림
#             if lines is not None:
#                 for line in lines:
#                     x1, y1, x2, y2 = line[0]
#                     cv2.line(frame_resized, (x1, y1), (x2, y2), (255, 0, 0), 3)

#             # 차원 맞추기: Y_binary를 BGR로 변환 후 크기 조정 <- 필요 x
#             # Y_binary_colored = cv2.cvtColor(Y_binary, cv2.COLOR_GRAY2BGR)
#             # Y_binary_colored = cv2.resize(Y_binary_colored, (width // 2, height // 2))

#             # imshow 합치기
#             # top_row = np.hstack((original_gaussian, Y_frame))
#             # bottom_row = np.hstack((Y_binary_colored, frame_resized))
#             # display = np.vstack((top_row, bottom_row))
#             # cv2.imshow('Y_frame', Y_frame)
#             # cv2.imshow('Y_binary', Y_binary)
#             cv2.imshow('frame_resized', frame_resized)

#             key = cv2.waitKey(0) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('n'):
#                 continue

#         self.cap.release()
#         cv2.destroyAllWindows()
#         print(v)

# if __name__ == "__main__":
#     # video_path = '/home/ubuntu/beom_ws/src/lane_image/video/hwg_curve_video.avi'
#     video_path = '/home/ubuntu/beom_ws/src/lane_image/video/hw_curve_3.avi' # 시작할 때 빛 살짝 튀는 영상
#     lane_detector = LaneDetector(video_path)
#     lane_detector.process_video()

# #############################################################33##########################################################################################################################################
# ## 영상을 처리하는 코드 : Auto Canny에 대해 Hough Transform 적용 
# import cv2
# import numpy as np
# import os
# import math
# import time
# import matplotlib.pyplot as plt


# class LaneDetector:
#     def __init__(self, video_path):
#         if not os.path.exists(video_path):
#             raise ValueError("Video path does not exist")

#         self.cap = cv2.VideoCapture(video_path)
#         if not self.cap.isOpened():
#             raise ValueError("Cannot open video")

#         self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         self.angle_data = []
#         self.previous_x = []
#         self.angle = 0
#         self.Y_max = 0
#         self.Y_max_3_5 = 0

#     # 감마 보정을 1차식으로 변경하고, 이 보정을 Y에 적용 
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

#         # 블러 (13,13) 적용
#         gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (13, 13), 2)
#         return gaussian_image

#     # 중앙값의 1.33/2.33 배 canny의 하향/상향 적용 
#     def re_auto_canny(self, image, sigma=0.33):
#         global v
#         v = np.median(image)
#         lower2 = int(min(255, (1.0 + sigma) * v))
#         lower = int(max(50, lower2))
#         upper = int(min(255, (2.0 + sigma) * v))
#         edged = cv2.Canny(image, lower, upper)
#         print('yuv auto \nlower: %d  upper: %d median: %d' % (lower, upper, v))
#         return edged

#     # 중앙값의 0.67/1.33 배 canny의 하향/상향 적용 
#     def original_auto_canny(self, image, sigma=0.33):
#         global v1
#         v1 = np.median(image)
#         lower = int(max(0, (1.0 - sigma) * v1))
#         upper = int(min(255, (1.0 + sigma) * v1))
#         edged = cv2.Canny(image, lower, upper)
#         print('original \nlower: %d  upper: %d median: %d' % (lower, upper, v1))
#         return edged

#     def process_video(self):
#         while self.cap.isOpened():
#             ret, frame = self.cap.read()
#             if not ret:
#                 break

#             # 오른쪽만 사용하기 위해서 x축의 오른쪽만 짜름
#             frame_resized = frame[:, self.width//2:self.width]
#             height, width, _ = frame_resized.shape
            
#             # 노이즈도 줄이고, 처리 속도를 빠르게 만들기 위해 이미지를 작게 수정
#             resized_image = cv2.resize(frame_resized, (width//2, height//2))

#             # YUV 변환 + 가우시안
#             YUV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
#             Y, _, _ = cv2.split(YUV)
#             Y_frame = self.YUV_transform(Y)

#             # auto canny 만들기
#             auto_canny = self.re_auto_canny(Y_frame)

#             # auto canny에 대해 허프 변환 적용
#             lines = cv2.HoughLinesP(auto_canny,rho=1,theta=np.pi / 180,threshold=100,minLineLength=50,maxLineGap=150)

#             # 검출된 직선을 frame_resized에 그림
#             if lines is not None:
#                 for line in lines:
#                     x1, y1, x2, y2 = line[0]
#                     cv2.line(frame_resized, (x1, y1), (x2, y2), (255, 0, 0), 3)


#             cv2.imshow("auto_canny",auto_canny)

#             cv2.imshow('frame_resized', frame_resized)

#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('n'):
#                 continue

#         self.cap.release()
#         cv2.destroyAllWindows()
#         print(v)

# if __name__ == "__main__":
#     # video_path = '/home/ubuntu/beom_ws/src/lane_image/video/hwg_curve_video.avi'
#     video_path = '/home/ubuntu/beom_ws/src/lane_image/video/hw_curve_3.avi' # 시작할 때 빛 살짝 튀는 영상
#     lane_detector = LaneDetector(video_path)
#     lane_detector.process_video()

#####################################################################################################33################################################################################################################################################
## 이미지 파일을 처리하는 코드 : Auto Canny 이미지에 대해서 Hough Transform 진행 + hough에 중점을 원으로 그리는 코드

import cv2
import numpy as np
import os
import math
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
        self.angle_data = []
        self.previous_x = []
        self.angle = 0
        self.Y_max = 0
        self.Y_max_3_5 = 0
        self.angle = 0
        self.angle_list = []
        self.angle_add = []


    # Y에 대한 감마 보정을, 1차식으로 변경하여 적용 : YUV_transform
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
    

    # 중앙값의 1.33/2.33 배 canny의 하향/상향 적용 
    def yuv_auto_canny(self, image, sigma=0.33):
        global v
        v = np.median(image)
        lower2 = int(min(255, (1.0 + sigma) * v))
        lower = int(max(50, lower2))
        upper = int(min(255, (2.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        print('yuv auto \nlower: %d  upper: %d median: %d' % (lower, upper, v))
        return edged

    def avg(self, data):
        return sum(data) / len(data)


    # 이미지 처리 메인 코드
    def process_images(self):
        for image_file in self.image_files:
            frame = cv2.imread(image_file)
            if frame is None:
                print(f"이미지를 읽을 수 없습니다: {image_file}")
                continue

            # 오른쪽만 사용하기 위해 x축의 오른쪽만 잘라냄30
            frame_resized = frame[:, self.width // 2:self.width]
            height, width, _ = frame_resized.shape
            
            # 노이즈를 줄이고 처리 속도를 빠르게 만들기 위해 이미지를 작게 수정
            resized_image = cv2.resize(frame_resized, (width // 2, height // 2))

            ### Hough Transform 결과를 띄우기 위해 resized_image 크기의 빈 행렬 생성
            # 검출된 직선의 각도를 확인하기 위한 코드 (1)
            base_hough = np.zeros_like(resized_image)

            # YUV 변환하여 Y 정보만 추출
            YUV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
            Y, _, _ = cv2.split(YUV)

            # YUV_transform 적용
            Y_frame = self.YUV_transform(Y)

            # YUV_transformd에 대해 Auto canny 적용
            auto_canny = self.yuv_auto_canny(Y_frame)

            #컨투어 도출하고 그리기
            # base_contours = np.zeros_like(resized_image)
            # contours, _ = cv2.findContours(auto_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(base_contours, contours, -1, (255, 255, 255), 2)
            # cv2.imshow("contours",base_contours)

            # Auto canny에 대해 허프 변환 적용
            ##minLineLength: 검출할 직선의 최소 길이로, 이 값보다 짧은 선분은 무시됩니다.
            ##maxLineGap: 동일한 직선으로 간주하기 위한 최대 간격으로, 이 값 이하의 간격을 가진 선분들은 하나의 직선으로 연결됩니다.
            lines = cv2.HoughLinesP(auto_canny, rho=1, theta=np.pi / 180, threshold=30, minLineLength=10, maxLineGap=50)

            # 중점을 계산하기 위한 리스트
            # midpoints = []

            # 검출된 직선의 각도를 확인하기 위한 코드 (2)
            if lines is not None:
                self.angle_list.clear()   # 차선의 각도를 출력한 리스트를 초기화 하기 위한 초기화 함수
                if lines is not None:  # 차선이 없지 않다면 실행
                    # 직진 그리는 과정
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        # 선의 각도 출력 (angle)
                        self.angle = np.degrees(np.arctan2((y2-y1),(x2-x1)))

                        # # 여기에 각도 값의 min max를 설정
                        if 20 <= self.angle <= 30:
                            # base_hough에 흰색 선 그리기
                            cv2.line(base_hough, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        # elif abs(self.angle) < 20:
                        #     cv2.line(base_hough, (x1,y1), (x2,y2), (0,255,0), 2)
                        # elif 30 < (self.angle) < 90:
                        #     cv2.line(base_hough, (x1,y1), (x2,y2), (0,0,255), 2)
                        
                        # 검출된 각도 크기를 눈으로 확인하기 위해 리스트에 저장하는 함수
                        self.angle_list.append(self.angle)
                        self.angle_add.append(self.angle)

                        # 검출 된 선의 중점 계산
                        # cx = (x1 + x2) // 2
                        # cy = (y1 + y2) // 2
                        # midpoints.append((cx, cy))   

                        # # 중점을 이미지에 표시 (cv2.circle(image,center(원의 중심좌표),radius(반지름),color,thickness=두께(-1은 채워진 원)))
                        # cv2.circle(base_hough, (cx, cy), 5, (0, 255, 0), -1)
                
                # if midpoints :
                #     avg_cx = int(sum([p[0] for p in midpoints]) / len(midpoints))
                #     avg_cy = int(sum([p[1] for p in midpoints]) / len(midpoints))

                #     # 최종 중점을 이미지에 표시
                #     cv2.circle(base_hough, (avg_cx, avg_cy), 10, (0, 0, 255), -1)
                #     print(f"최종 중점: ({avg_cx}, {avg_cy})")
                else:
                    print("중점을 계산할 수 있는 선이 없습니다.")

            # 검출된 직선의 각도를 확인하기 위한 코드 (3)
            print(self.angle_list)
            

            cv2.imshow("Auto Canny", auto_canny)
            cv2.imshow("base_hough", base_hough)
            cv2.imshow("resized_image", resized_image)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # 'q'를 누르면 종료
                break
            elif key == ord('n'):  # 'n'을 누르면 다음 이미지로 이동
                continue

        cv2.destroyAllWindows()
        average = self.avg(self.angle_add)
        print(average)
        # 1115/yellow_straight : 24.782316253159745
        # 1115/white_straight : 26.872538590155468
        # 전기관 1층 직진 차선 : 27.828398381986386
        # 대략 직선은 20~30 사이의 각도쯤 되는듯




if __name__ == "__main__":
    ## 유진님이 주신 사진
    image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_left'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_right'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_straight'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_left'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_right'
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_straight'

    ## 전기관 1층 사진
    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/load_image_2' #직선사진 하나

    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/load_image_1/load_image_1'  ##리모 사진 모음
    detector = LaneDetector(image_dir)
    detector.process_images()


#####################################################################################################33########################################################################################################################################################
# ## 이미지 파일을 처리하는 코드 : Auto Canny 이미지에 대해서 Hough Transform 진행 + hough에 중점을 원으로 그리는 코드 + 곡선 판단 코드

# ######이게 hough에서 도출 된 선의 중점의 점 기준으로, 3개 이상의 점을 바탕으로 곡선을 피팅하는거라서 hough가 곡선이 도출되지 않으면 결국 2차함수가 피팅이 안되기 때문에 근본이 틀린 코드


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

#             YUV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
#             Y, _, _ = cv2.split(YUV)
#             Y_frame = self.YUV_transform(Y)
#             auto_canny = self.yuv_auto_canny(Y_frame)

#             lines = cv2.HoughLinesP(auto_canny, rho=1, theta=np.pi / 180, threshold=30, minLineLength=10, maxLineGap=30)
#             midpoints = []

#             if lines is not None:
#                 self.angle_list.clear()
#                 for line in lines:
#                     x1, y1, x2, y2 = line[0]
#                     self.angle = np.degrees(np.arctan2((y2-y1), (x2-x1)))

#                     if 20 <= self.angle <= 30:
#                         cv2.line(base_hough, (x1, y1), (x2, y2), (255, 255, 255), 2)
#                         self.angle_list.append(self.angle)
#                         self.angle_add.append(self.angle)

#                         # 선의 중점 계산
#                         cx = (x1 + x2) // 2
#                         cy = (y1 + y2) // 2
#                         midpoints.append((cx, cy))
#                         cv2.circle(base_hough, (cx, cy), 5, (0, 255, 0), -1)

#                 if midpoints:
#                     fit = self.fit_curve(midpoints)  # 곡선 피팅
#                     direction = self.calculate_direction(fit, resized_image.shape[0] - 1)
#                     print(f"차량 진행 방향: {direction}")

#             cv2.imshow("Auto Canny", auto_canny)
#             cv2.imshow("base_hough", base_hough)
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
#     image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_left'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_right'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_straight'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_left'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_right'
#     # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_straight'
#     detector = LaneDetector(image_dir)
#     detector.process_images()

