# import time
# start_t = time.time()

# end_t = time.time()
# print(end_t - start_t)

#############################################################################
#동영상 불러오기
# import cv2
# import numpy as np
# import os
# import math
# import time
# import matplotlib.pyplot as plt

# start_T = time.time()

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

#     def process_video(self):
#         while self.cap.isOpened():
#             ret, frame = self.cap.read()

#             if not ret:
#                 break
            
#             start_T = time.time()

#             frame_resized = cv2.resize(frame, (self.width//2, self.height//2))

#             # YUV 변환
#             YUV = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)
#             Y, U, V = cv2.split(YUV)

#             # Y채널 max 값 설정 (3/5)
#             Y_max = np.max(Y)
#             # Y_max = 255
#             Y_max_3_5 = Y_max * (3 / 5)

#             # np.where를 사용해 전체에 적용
#             Y_trans = np.where(
#                 (Y > 0) & (Y < Y_max_3_5),  # 조건 1: 0 < Y < Y_max_3_5
#                 Y / 3,                      # 조건 1이 참일 때 적용할 값
#                 np.where(                   # 중첩된 np.where 사용
#                     (Y >= Y_max_3_5) & (Y < Y_max),  # 조건 2: Y_max_3_5 <= Y < Y_max 
#                     (Y * 2) - Y_max,       # 조건 2가 참일 때 적용할 값
#                     Y                               # 조건이 모두 거짓일 때 원래 값 유지
#                 )
#             )

#             Y_trans_uint8 = Y_trans.astype(np.uint8)

#             # Gaussian Blur 사용
#             gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (7, 7),2)
#             gaussian_image_re = cv2.resize(gaussian_image, (800, 480))


#             # Canny edge detection
#             canny_image = cv2.Canny(gaussian_image, 150, 200)
#             canny_image_re = cv2.resize(canny_image, (800, 480))

#             end_T = time.time()
#             print(end_T - start_T)

#             # OpenCV를 사용하여 이미지를 표시
#             cv2.imshow('Original Frame', frame_resized)
#             cv2.imshow('Canny Edge Detection', canny_image_re)
#             cv2.imshow('Y Blur', gaussian_image_re)


#             # 클릭 또는 'n' 키를 누르면 다음 이미지로 넘어가게 하기
#             key = cv2.waitKey(0) & 0xFF
#             if key == ord('q'):  # 'q'키를 누르면 종료
#                 break
#             elif key == ord('n'):  # 'n'키를 누르면 다음 이미지로 이동
#                 continue

#         self.cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     video_path = '/home/ubuntu/beom_ws/src/lane_image/video/curve_video.avi'
#     # video_path = '/home/ubuntu/beom_ws/src/lane_image/video/hwg_curve_video.avi'
#     # video_path = '/home/ubuntu/beom_ws/src/lane_image/video/hw_curve_3.avi'
#     lane_detector = LaneDetector(video_path)
#     lane_detector.process_video()



####################################################################################################################
# #동영상 불러오기 임시 완성본- roi 나눠서 적용하기
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

#     def roi_processing(self, Y):
#         self.Y_max = np.max(Y)
#         self.Y_max_3_5 = self.Y_max * (3 / 5)

#         # np.where를 사용해 전체에 적용
#         Y_trans = np.where(
#             (Y > 0) & (Y < self.Y_max_3_5),  # 조건 1: 0 < Y < self.Y_maxx_3_5
#             Y / 3,                      # 조건 1이 참일 때 적용할 값
#             np.where(                   # 중첩된 np.where 사용
#                 (Y >= self.Y_max_3_5) & (Y < self.Y_max),  # 조건 2: self.Y_max_3_5 <= Y < self.Y_max    
#                 (Y * 2) - self.Y_max,       # 조건 2가 참일 때 적용할 값
#                 Y                               # 조건이 모두 거짓일 때 원래 값 유지
#             )
#         )

#         Y_trans_uint8 = Y_trans.astype(np.uint8)

#         # Gaussian Blur 사용
#         gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (7,7) ,2)

#         canny_image = cv2.Canny(gaussian_image, 150, 200)

#         return canny_image

#     def auto_canny(self, image, sigma = 0.33):
#         # compute the median of the single channel pixel intensities
#         v = np.median(image) # 0.0002319812774658203
        
#         start = time.time()

#         # apply automatic Canny edge detection using the computed median
#         lower = int(max(0, (1.0 - sigma) * v))
#         upper = int(min(255, (1.0 + sigma) * v))
#         edged = cv2.Canny(image, lower, upper)
#         # low upper 따고 canny 적용 -> 0.00011372566223144531

#         end = time.time()
#         print(end - start)


#         print('lower: %d  upper: %d' % (lower, upper))
    
#         # return the edged image
#         return edged   

#         # # 2차 시도 median 값 사용하는게 아니라 max 값 사용

#         # # compute the median of the single channel pixel intensities
#         # v_max = np.max(image)   
#         # v_mid = np.median(image)  
#         # v = (v_max + v_mid) / 2
 
#         # # apply automatic Canny edge detection using the computed median
#         # lower = int(max(0, (1.0 - sigma) * v))
#         # upper = int(min(255, (1.0 + sigma) * v))
#         # edged = cv2.Canny(image, lower, upper)

#         # # print('lower: %d  upper: %d' % (lower, upper))
    
#         # # return the edged image
#         # return edged


#     def process_video(self):
#         while self.cap.isOpened():
#             start = time.time()
#             ret, frame = self.cap.read()

#             if not ret:
#                 break

#             frame_resized = frame[0:self.height//3*2, :]
#             # frame_resized = cv2.resize(frame_re, (self.width//3*2, self.height//2))

#             height, width, _ = frame_resized.shape

#             # YUV 변환
#             YUV = cv2.cvtColor(frame_resized,cv2.COLOR_BGR2YUV)
#             Y,_,_ = cv2.split(YUV)
#             Y = cv2.GaussianBlur(Y, (11, 11), 0)

#             roi1 = Y[0:height//2, 0:width//2]
#             roi2 = Y[height//2:height, 0:width//2]
#             roi3 = Y[0:height//2, width//2:width]
#             roi4 = Y[height//2:height, width//2:width]
#             # cv2.imshow('roi1',roi1)
#             # cv2.imshow('roi2',roi2)
#             # cv2.imshow('roi3',roi3)
#             # cv2.imshow('roi4',roi4)

#             # canny_roi1 = self.roi_processing(roi1)
#             # canny_roi2 = self.roi_processing(roi2)
#             # canny_roi3 = self.roi_processing(roi3)
#             # canny_roi4 = self.roi_processing(roi4)

#             # auto_roi1 = self.auto_canny(roi1)
#             # auto_roi2 = self.auto_canny(roi2)
#             # auto_roi3 = self.auto_canny(roi3)
#             # auto_roi4 = self.auto_canny(roi4)

#             # result_roi1 = canny_roi1 + auto_roi1
#             # result_roi2 = canny_roi2 + auto_roi2
#             # result_roi3 = canny_roi3 + auto_roi3
#             # result_roi4 = canny_roi4 + auto_roi4
            
#             # result = np.zeros([height, width], dtype=np.uint8)
#             # result[0:height//2, 0:width//2] = result_roi1
#             # result[height//2:height, 0:width//2] = result_roi2
#             # result[0:height//2, width//2:width] = result_roi3
#             # result[height//2:height, width//2:width] = result_roi4


#             rois = [roi1, roi2, roi3, roi4]

#             # 각 ROI에 대해 처리한 결과 저장
#             canny_rois = [self.roi_processing(roi) for roi in rois]
#             auto_rois = [self.auto_canny(roi) for roi in rois]

#             result2 = np.zeros([height, width], dtype=np.uint8)
#             result2[0:height//2, 0:width//2] = canny_rois[0]
#             result2[height//2:height, 0:width//2] = canny_rois[1]
#             result2[0:height//2, width//2:width] = canny_rois[2]
#             result2[height//2:height, width//2:width] = canny_rois[3]


#             # Canny와 auto_canny 결과를 합산
#             result_rois = [canny + auto for canny, auto in zip(canny_rois, auto_rois)]

#             # 결과를 4개의 영역으로 나누어 할당
#             result = np.zeros([height, width], dtype=np.uint8)
#             result[0:height//2, 0:width//2] = result_rois[0]
#             result[height//2:height, 0:width//2] = result_rois[1]
#             result[0:height//2, width//2:width] = result_rois[2]
#             result[height//2:height, width//2:width] = result_rois[3]


#             # result = np.zeros([height, width], dtype=np.uint8)
#             # result[0:height//2, 0:width//2] = auto_rois[0]
#             # result[height//2:height, 0:width//2] = auto_rois[1]
#             # result[0:height//2, width//2:width] = auto_rois[2]
#             # result[height//2:height, width//2:width] = auto_rois[3]

#             # lines = cv2.HoughLinesP(result, rho=1, theta=np.pi/180, threshold=100, minLineLength=70, maxLineGap=50)
#             # result_line = np.zeros([height, width], dtype=np.uint8)

#             # # 직선을 원본 이미지에 그리기
#             # if lines is not None:
#             #     for line in lines:
#             #         x1, y1, x2, y2 = line[0]

#             #         if (x2 - x1) != 0:
#             #             angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                        
#             #         # # 직선의 기울기를 계산
#             #         # angle = math.degrees(math.atan2(y2 - y1, x2 - x1))  # 라디안을 도 단위로 변환
                    
#             #         # 30도 ~ 150도 범위 내의 직선만 그리기
#             #         if 10 <= abs(angle):  # 절댓값을 사용하는 이유는 기울기 방향을 고려
#             #             cv2.line(result_line, (x1, y1), (x2, y2), (50, 50, 50), 2)
            

#             # OpenCV를 사용하여 이미지를 표시
#             cv2.imshow('Original Frame', frame_resized)
#             # cv2.imshow('Canny Edge Detection', canny_image)
#             # cv2.imshow('Y Blur', gaussian_image)
#             # cv2.imshow('canny_roi1', canny_roi1)
#             # cv2.imshow('canny_roi2', canny_roi2)
#             # cv2.imshow('canny_roi3', auto_roi3)
#             # cv2.imshow('canny_roi4', auto_roi4)
#             # cv2.imshow('result1', result1)
#             # cv2.imshow('result2', result2)
#             cv2.imshow('result', result)
#             cv2.imshow('YUV_Four_ROI', result2)
#             # cv2.imshow('result_line', result_line)


#             # 클릭 또는 'n' 키를 누르면 다음 이미지로 넘어가게 하기
#             key = cv2.waitKey(0) & 0xFF
#             if key == ord('q'):  # 'q'키를 누르면 종료
#                 break
#             elif key == ord('n'):  # 'n'키를 누르면 다음 이미지로 이동
#                 continue

#         self.cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     video_path = '/home/ubuntu/beom_ws/src/lane_image/video/curve_video.avi'
#     # video_path = '/home/ubuntu/beom_ws/src/lane_image/video/hwg_curve_video.avi' # curve_video 영상을 보면 auto보다 gamma가 더 잘 나타나는 경우가 있고, 그 외에는 auto가 더 좋음 -> auto + gamma
#     # video_path = '/home/ubuntu/beom_ws/src/lane_image/video/hw_curve_3.avi'
#     lane_detector = LaneDetector(video_path)
#     lane_detector.process_video()

################################################################################################################################### 거의 최종
# edge 적용한 이미지 하나씩 다 띄워보기
# 1. original edge 적용 / original auto canny 적용
# 2. yuv edge 적용 / yuv auto canny 적용
# yuv auto canny 는 노이즈가 너무 많아서 못 씀
# 그냥 edge는 (100, 150) 잡기


import cv2
import numpy as np
import os
import math
import time
import matplotlib.pyplot as plt


class LaneDetector:
    def __init__(self, video_path):
        if not os.path.exists(video_path):
            raise ValueError("Video path does not exist")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Cannot open video")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.angle_data = []
        self.previous_x = []
        self.angle = 0
        self.Y_max = 0
        self.Y_max_3_5 = 0

    # def canny(self, image):
    #     img_canny = cv2.Canny(image, 100, 150)
    #     return img_canny

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

        gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (7,7) ,2)

        #canny_image = cv2.Canny(gaussian_image, 150, 200)

        return gaussian_image


    def auto_canny(self, image, sigma = 0.33):
        global v
        v = np.median(image)
        lower = int(min(255, (1.0 + sigma) * v))
        upper = int(min(255, (2.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        print('lower: %d  upper: %d' % (lower, upper))
    
        return edged

    def process_video(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_resized = frame[0:self.height//3*2, :]

            height, width, _ = frame_resized.shape

            resized_image = cv2.resize(frame_resized, (width // 2, height // 2))

            # gray_frame = cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY)
            
            # original_gaussian = cv2.GaussianBlur(gray_frame, (7,7) ,2)
            # cv2.imshow('original_gaussian',original_gaussian)

            # YUV 변환 + 가우시안
            YUV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2YUV)
            Y, _, _ = cv2.split(YUV)
            Y_frame = self.YUV_transform(Y) #YUV 변환한 이미지

            # # 1. original edge (100, 150) 적용
            # original_canny = self.canny(original_gaussian)

            # # 2. original auto canny 적용
            # original_auto_canny = self.auto_canny(original_gaussian)

            # # 3. YUV edge (100, 150) 적용
            # YUV_canny = self.canny(Y_frame)

            # 4. YUV auto canny 적용
            YUV_auto_canny = self.auto_canny(Y_frame)

            # OpenCV를 사용하여 4개의 이미지를 표시
            # cv2.imshow('Original Edge (100, 150)', original_canny)
            # cv2.imshow('Original Auto Canny', original_auto_canny)
            # cv2.imshow('YUV Edge (100, 150)', YUV_canny)
            cv2.imshow('YUV Auto Canny', YUV_auto_canny)
            cv2.imshow('YUV', Y_frame)

            # 클릭 또는 'n' 키를 누르면 다음 이미지로 넘어가게 하기
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # 'q'키를 누르면 종료
                break
            elif key == ord('n'):  # 'n'키를 누르면 다음 이미지로 이동
                continue

        self.cap.release()
        cv2.destroyAllWindows()
        print(v)

if __name__ == "__main__":
    video_path = '/home/ubuntu/beom_ws/src/lane_image/video/curve_video.avi'
    # video_path = '/home/ubuntu/beom_ws/src/lane_image/video/hwg_curve_video.avi' # curve_video 영상을 보면 auto보다 gamma가 더 잘 나타나는 경우가 있고, 그 외에는 auto가 더 좋음 -> auto + gamma
    # video_path = '/home/ubuntu/beom_ws/src/lane_image/video/hw_curve_3.avi'
    lane_detector = LaneDetector(video_path)
    lane_detector.process_video()




#######################################################################################################################3
# # #이미지 불러오기
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import time

# # # path = '/home/ubuntu/Downloads/frame000000.png' # 말안됨
# path = '/home/ubuntu/beom_ws/src/lane_image/frame000002.png' # 밝은날
# image = cv2.imread(path)

# height, width,_ = image.shape

# resized_image = cv2.resize(image, (width // 2, height // 2))

# # YUV 변환
# YUV = cv2.cvtColor(resized_image,cv2.COLOR_BGR2YUV)
# Y,U,V = cv2.split(YUV)


# # Y채널 max 값 설정 (3/5)
# global Y_max, Y_max_3_5
# Y_max = np.max(Y)
# Y_max_3_5 = Y_max * (3 / 5)

# # np.where를 사용해 전체에 적용
# Y_trans = np.where(
#     (Y > 0) & (Y < Y_max_3_5),  # 조건 1: 0 < Y < Y_max_3_5
#     Y / 3,                      # 조건 1이 참일 때 적용할 값
#     np.where(                   # 중첩된 np.where 사용
#         (Y >= Y_max_3_5) & (Y < Y_max),  # 조건 2: Y_max_3_5 <= Y < Y_max    
#         (Y * 2) - Y_max,       # 조건 2가 참일 때 적용할 값
#         Y                               # 조건이 모두 거짓일 때 원래 값 유지
#     )
# )

# # #Y채널 max 값 설정 (4/5)
# # global Y_max, Y_max_4_5
# # Y_max = np.max(Y)
# # Y_max_4_5 = Y_max * (4 / 5)

# # # np.where를 사용해 전체에 적용
# # Y_trans = np.where(
# #     (Y > 0) & (Y < Y_max_4_5),  # 조건 1: 0 < Y < Y_max_4_5
# #     Y / 4,                      # 조건 1이 참일 때 적용할 값
# #     np.where(                   # 중첩된 np.where 사용
# #         (Y >= Y_max_4_5) & (Y < Y_max),  # 조건 2: Y_max_4_5 <= Y < Y_max    
# #         (4 * Y) - (3 * Y_max),       # 조건 2가 참일 때 적용할 값
# #         Y                               # 조건이 모두 거짓일 때 원래 값 유지
# #     )
# # )


# Y_trans_uint8 = Y_trans.astype(np.uint8)

# # Gaussian Blur 사용
# gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (7,7) ,0)

# canny_image = cv2.Canny(gaussian_image, 150, 200)


# plt.figure('image')

# plt.subplot(2,2,1)
# plt.title("Original image")
# plt.imshow(image) 

# plt.subplot(2,2,2)
# plt.title("Y original")
# plt.imshow(Y,cmap='gray')

# plt.subplot(2,2,3)
# plt.title("Y change (+gaussian)")
# plt.imshow(gaussian_image,cmap='gray')

# plt.subplot(2,2,4)
# plt.title("Y Canny")
# plt.imshow(canny_image, cmap='gray')

# plt.show()

###############################################################################################################
# ###동적인 canny edge
# import cv2
# import numpy as np
# import os
# import time


# class LaneDetector:
#     def __init__(self, video_path):
#         if not os.path.exists(video_path):
#             raise ValueError("Video path does not exist")

#         self.cap = cv2.VideoCapture(video_path)
#         if not self.cap.isOpened():
#             raise ValueError("Cannot open video")

#         self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # 동적 Canny 함수
#     def auto_canny(self, image, sigma=1.5):
#         # 가우시안 블러 적용 (노이즈 제거)
#         image = cv2.GaussianBlur(image, (3, 3), 2)
#         # 이미지 밝기의 중간값 계산
#         v = np.median(image)import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import time

# # # path = '/home/ubuntu/Downloads/frame000000.png' # 말안됨
# path = '/home/ubuntu/beom_ws/src/lane_image/frame000002.png' # 밝은날
# img = cv2.imread(path)
# height, width,_ = img.shape
# image = img[0:height//3*2, :]

# height, width,_ = image.shape

# cv2.imshow('image', image)
# # resized_image = cv2.resize(image, (width // 2, height // 2))

# # resized_img = image[height//2:]
# # YUV 변환

# YUV = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
# Y,_,_ = cv2.split(YUV)

# # roi1 =  Y[0:height//2, 0:width//3]
# # cv2.imshow('roi1',roi1)
# # roi2 =  Y[height//2:height, 0:width//3]
# # cv2.imshow('roi2',roi2)
# roi3 = Y[0:height//2, width//3*2:width]
# cv2.imshow('roi3',roi3)
# # roi4 = Y[height//2:height, width//3*2:width]
# # cv2.imshow('roi4',roi4)



# Y = roi3
# start = time.time()
# # Y채널 max 값 설정 (3/5)
# global Y_max, Y_max_3_5
# Y_max = np.max(Y)
# Y_max_3_5 = Y_max * (3 / 5)

# # np.where를 사용해 전체에 적용
# Y_trans = np.where(
#     (Y > 0) & (Y < Y_max_3_5),  # 조건 1: 0 < Y < Y_max_3_5
#     Y / 3,                      # 조건 1이 참일 때 적용할 값
#     np.where(                   # 중첩된 np.where 사용
#         (Y >= Y_max_3_5) & (Y < Y_max),  # 조건 2: Y_max_3_5 <= Y < Y_max    
#         (Y * 2) - Y_max,       # 조건 2가 참일 때 적용할 값
#         Y                               # 조건이 모두 거짓일 때 원래 값 유지
#     )
# )


# Y_trans_uint8 = Y_trans.astype(np.uint8)

# # Gaussian Blur 사용
# gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (5,5) ,0)
# cv2.imshow('gaussian_image', gaussian_image)

# canny_image = cv2.Canny(gaussian_image, 150, 200)

# cv2.imshow('canny',canny_image)

# cv2.waitKey(0)

# end = time.time()
# print(end - start)

#         # 자동 임계값 계산
#         lower = int(max(0, (1.0 - sigma) * v))
#         upper = int(min(255, (1.0 + sigma) * v))
#         # Canny 엣지 검출 수행
#         edged = cv2.Canny(image, lower, upper)
#         return edged

#     def process_video(self):
#         while self.cap.isOpened():
#             ret, frame = self.cap.read()

#             if not ret:
#                 break

#             start_T = time.time()

#             # 프레임 크기 조정
#             frame_resized = cv2.resize(frame, (self.width // 2, self.height // 2))

#             # YUV 변환
#             YUV = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)
#             Y, _, _ = cv2.split(YUV)

#             # Y채널 max 값 설정 (3/5)
#             Y_max = np.max(Y)
#             Y_max_3_5 = Y_max * (3 / 5)

#             # np.where를 사용해 밝기 조정
#             Y_trans = np.where(
#                 (Y > 0) & (Y < Y_max_3_5), 
#                 Y / 3,
#                 np.where(
#                     (Y >= Y_max_3_5) & (Y < Y_max),  
#                     (Y * 2) - Y_max,
#                     Y
#                 )
#             )

#             Y_trans_uint8 = Y_trans.astype(np.uint8)

#             # Gaussian Blur 사용
#             gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (7, 7), 2)

#             # 동적 Canny Edge Detection 적용
#             dynamic_canny = self.auto_canny(gaussian_image)import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import time

# # # path = '/home/ubuntu/Downloads/frame000000.png' # 말안됨
# path = '/home/ubuntu/beom_ws/src/lane_image/frame000002.png' # 밝은날
# img = cv2.imread(path)
# height, width,_ = img.shape
# image = img[0:height//3*2, :]

# height, width,_ = image.shape

# cv2.imshow('image', image)
# # resized_image = cv2.resize(image, (width // 2, height // 2))

# # resized_img = image[height//2:]
# # YUV 변환

# YUV = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
# Y,_,_ = cv2.split(YUV)

# # roi1 =  Y[0:height//2, 0:width//3]
# # cv2.imshow('roi1',roi1)
# # roi2 =  Y[height//2:height, 0:width//3]
# # cv2.imshow('roi2',roi2)
# roi3 = Y[0:height//2, width//3*2:width]
# cv2.imshow('roi3',roi3)
# # roi4 = Y[height//2:height, width//3*2:width]
# # cv2.imshow('roi4',roi4)



# Y = roi3
# start = time.time()
# # Y채널 max 값 설정 (3/5)
# global Y_max, Y_max_3_5
# Y_max = np.max(Y)
# Y_max_3_5 = Y_max * (3 / 5)

# # np.where를 사용해 전체에 적용
# Y_trans = np.where(
#     (Y > 0) & (Y < Y_max_3_5),  # 조건 1: 0 < Y < Y_max_3_5
#     Y / 3,                      # 조건 1이 참일 때 적용할 값
#     np.where(                   # 중첩된 np.where 사용
#         (Y >= Y_max_3_5) & (Y < Y_max),  # 조건 2: Y_max_3_5 <= Y < Y_max    
#         (Y * 2) - Y_max,       # 조건 2가 참일 때 적용할 값
#         Y                               # 조건이 모두 거짓일 때 원래 값 유지
#     )
# )


# Y_trans_uint8 = Y_trans.astype(np.uint8)

# # Gaussian Blur 사용
# gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (5,5) ,0)
# cv2.imshow('gaussian_image', gaussian_image)

# canny_image = cv2.Canny(gaussian_image, 150, 200)

# cv2.imshow('canny',canny_image)

# cv2.waitKey(0)

# end = time.time()
# print(end - start)


#             end_T = time.time()
#             print(f"Processing time: {end_T - start_T:.2f} seconds")

#             # 결과 이미지 표시
#             cv2.imshow('Original Frame', frame_resized)
#             cv2.imshow('Dynamic Canny Edge Detection', dynamic_canny)
#             cv2.imshow('Y Blur', gaussian_image)



#             # 키 입력 대기
#             key = cv2.waitKey(0) & 0xFF
#             if key == ord('q'):  # 'q'키를 누르면 종료
#                 break
#             elif key == ord('n'):  # 'n'키를 누르면 다음 이미지로 이동
#                 continue

#         self.cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     video_path = '/home/ubuntu/beom_ws/src/lane_image/video/curve_video.avi'
#     lane_detector = LaneDetector(video_path)
#     lane_detector.process_video()



#################################################3###################################################################

# # #이미지 불러오기
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import time

# # # path = '/home/ubuntu/Downloads/frame000000.png' # 말안됨
# path = '/home/ubuntu/beom_ws/src/lane_image/frame000002.png' # 밝은날
# img = cv2.imread(path)
# height, width,_ = img.shape
# image = img[0:height//3*2, :]

# height, width,_ = image.shape

# cv2.imshow('image', image)
# # resized_image = cv2.resize(image, (width // 2, height // 2))

# # resized_img = image[height//2:]
# # YUV 변환

# YUV = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
# Y,_,_ = cv2.split(YUV)

# # roi1 =  Y[0:height//2, 0:width//3]
# # cv2.imshow('roi1',roi1)
# # roi2 =  Y[height//2:height, 0:width//3]
# # cv2.imshow('roi2',roi2)
# roi3 = Y[0:height//2, width//3*2:width]
# cv2.imshow('roi3',roi3)
# # roi4 = Y[height//2:height, width//3*2:width]
# # cv2.imshow('roi4',roi4)



# Y = roi3
# start = time.time()
# # Y채널 max 값 설정 (3/5)
# global Y_max, Y_max_3_5
# Y_max = np.max(Y)
# Y_max_3_5 = Y_max * (3 / 5)

# # np.where를 사용해 전체에 적용
# Y_trans = np.where(
#     (Y > 0) & (Y < Y_max_3_5),  # 조건 1: 0 < Y < Y_max_3_5
#     Y / 3,                      # 조건 1이 참일 때 적용할 값
#     np.where(                   # 중첩된 np.where 사용
#         (Y >= Y_max_3_5) & (Y < Y_max),  # 조건 2: Y_max_3_5 <= Y < Y_max    
#         (Y * 2) - Y_max,       # 조건 2가 참일 때 적용할 값
#         Y                               # 조건이 모두 거짓일 때 원래 값 유지
#     )
# )


# Y_trans_uint8 = Y_trans.astype(np.uint8)

# # Gaussian Blur 사용
# gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (5,5) ,0)
# cv2.imshow('gaussian_image', gaussian_image)

# canny_image = cv2.Canny(gaussian_image, 150, 200)

# cv2.imshow('canny',canny_image)

# cv2.waitKey(0)

# end = time.time()
# print(end - start)

# plt.figure('image')

# plt.subplot(2,2,1)
# plt.title("Original image")
# plt.imshow(image) 

# plt.subplot(2,2,2)
# plt.title("Y original")
# plt.imshow(Y,cmap='gray')

# plt.subplot(2,2,3)
# plt.title("Y change (+gaussian)")
# plt.imshow(gaussian_image,cmap='gray')

# plt.subplot(2,2,4)
# plt.title("Y Canny")
# plt.imshow(canny_image, cmap='gray')

# plt.show()