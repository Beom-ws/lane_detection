import cv2
import numpy as np
import os
import time

# start_T = time.time()
# end_T = time.time()
# print(end_T - start_T)
global start_T 
global end_T

#좌/진/우 판단 변수
count_num=[0,0,0]
all_count=[0]


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


    # 0.0006초  -> intensity transform을 먼저하고 gaussian을 하는 이유 : gaussian은 차선 정보를 훼손 시킬 확률이 높은데 먼저 진행하게 되면 강건하게 도출되지 않을 수 있다. 그래서 차선의 정보를 확실하게 뽑고 노이즈를 지우는 가우시안을 사용하는게 맞다.
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
        lower2 = int(min(255, (2.0 + sigma) * v))
        lower = int(max(50, lower2))
        upper = int(min(255, (3.0 + sigma) * v))
        print(v)
        print(lower)
        print(upper)
        edged = cv2.Canny(image, lower, upper)
        return edged
    

    # 0.0028 최고 기준
    def process_images(self):
        
        for image_file in self.image_files:
            start_T = time.time()
            frame = cv2.imread(image_file)
            if frame is None:
                print(f"이미지를 읽을 수 없습니다: {image_file}")
                continue
            h, w, _ = frame.shape
            orignal_frame = frame
            # cv2.imshow('orignal_frame',orignal_frame)
            # orignal_resized = cv2.resize(frame,(w // 2, h // 2))
            
            half_frame = frame[:, self.width // 2:self.width]
            # cv2.imshow('half_frame',half_frame)
            height, width, _ = half_frame.shape
            resized_frame = cv2.resize(half_frame, (width // 2, height // 2))
            cv2.imshow('resized_frame',resized_frame)
            height, width = resized_frame.shape[:2]

            # YUV 변환 및 Canny Edge Detection
            YUV = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2YUV)
            # cv2.imshow('YUV_frame',YUV)
            Y, _, _ = cv2.split(YUV)
            # cv2.imshow('Y_frame',Y)
            Y_frame = self.YUV_transform(Y)
            cv2.imshow('YUV_transform_frame',Y_frame)
            auto_canny = self.yuv_auto_canny(Y_frame)
            cv2.imshow('auto_canny_frame',auto_canny)

            # 버드아이뷰를 위한 사다리꼴 영역 설정 
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
            # cv2.imshow('bev_frame',bev_img)
            
            # BEV 이미지 이진화 처리
            _, bin_img = cv2.threshold(bev_img, 70, 255, cv2.THRESH_BINARY)
            bev_img = bin_img
            # cv2.imshow('threshold_bev_frame',bev_img)


            # 0.00020647048950195312
            # Convexity 기반 V자형 노이즈 제거
            contours, _ = cv2.findContours(bev_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.RETR_EXTERNAL : 외곽 윤곽선만을 찾도록 지정하는 플래그입니다.
                    # cv2.RETR_EXTERNAL
                    # 목적: 외곽 윤곽선만 추출
                    # 이 플래그는 이미지의 가장 바깥쪽 윤곽선만 찾습니다.
                    # 내부 윤곽선(예: 원 안의 원처럼 내부에 또 다른 윤곽이 있는 경우)은 무시됩니다.
            # cv2.CHAIN_APPROX_SIMPLE : 윤곽선의 점을 단순화하여 불필요한 점들을 제거합니다.
            filtered_contours = []

            for contour in contours:
                hull = cv2.convexHull(contour) 
                # cv2.convexHull은 주어진 윤곽선에 대해 볼록 다각형을 만듭니다. 볼록 다각형은 주어진 점들 중 모든 점을 포함하면서 가장 바깥쪽에 위치한 점들로 이루어진 다각형입니다.
                contour_area = cv2.contourArea(contour)

                hull_area = cv2.contourArea(hull)

                if hull_area > 0:
                    convexity = contour_area / hull_area
                    if convexity > 0.3:  # Convexity 임계값 설정 #0.5 : 차선 사라짐 #0.4 : 잘 나오는 듯 보이다가 차선이 한번씩 사라짐 #0.3 : best인듯
                        filtered_contours.append(contour)


            # 필터링된 Blob 시각화
            filtered_bev_img = np.zeros_like(bev_img)
            cv2.drawContours(filtered_bev_img, filtered_contours, -1, 255, thickness=cv2.FILLED)
            cv2.imshow('convexity_bev_frame',filtered_bev_img)

            # 중심점을 계산
            non_zero_points = np.argwhere(filtered_bev_img > 0)  # 비어 있지 않은 픽셀 좌표
            if len(non_zero_points) > 0: # 픽셀값이 한개라도 있을 때 실행
                start_point = np.mean(non_zero_points, axis=0).astype(int) # 중심값을 찾음
                cv2.circle(filtered_bev_img, tuple(start_point[::-1]), radius=10, color=(50,50,50), thickness=-1)
                # cv2.imshow('center_of_value_bev_frame',filtered_bev_img)

                # 1/3 지점, 2/3 지점 계산 코드
                center_x_1_3 = filtered_bev_img.shape[1] / 3 
                center_x_2_3 = center_x_1_3 * 2

                if start_point[1] < center_x_1_3:  # 중심점이 이미지 1/3 왼쪽에 위치
                    all_count[0]=all_count[0]+1
                    print(f"{all_count[0]}번째 이미지 = 좌회전")
                    count_num[0]=count_num[0]+1

                elif center_x_2_3 < start_point[1] < filtered_bev_img.shape[1] :  # 중심점이 이미지 2/3 오른쪽에 위치
                    all_count[0]=all_count[0]+1
                    print(f"{all_count[0]}번째 이미지 = 우회전")
                    count_num[2]=count_num[2]+1
                else :
                    all_count[0]=all_count[0]+1
                    print(f"{all_count[0]}번째 이미지 = 직진")
                    count_num[1]=count_num[1]+1
                

            else: # 비어 있는 이미지 일 때
                print("비어 있는 이미지입니다. 중심점을 계산할 수 없습니다.")
            if bev_img is None or bev_img.size == 0: # 이미지가 아니거나 사이즈가 없는, 즉 이미지가 아닐 때
                print("BEV 이미지 생성 실패")
                continue

            # 사다리꼴 영역을 원본 이미지에 시각화
            pts = np.array([p1, p2, p4, p3], np.int32).reshape((-1, 1, 2))
            cv2.polylines(auto_canny, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
            # cv2.imshow('auto_canny_bev_frame',auto_canny)

            # 결과 출력
            # cv2.imshow("Auto Canny", auto_canny)

            # BEV 이미지 출력
            # cv2.imshow('bev_img', bev_img)

            # convexity
            # cv2.imshow('filtered_bev_img', filtered_bev_img)
            end_T = time.time()
            k = end_T - start_T
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # 'q' 키로 종료
                break
            elif key == ord('n'):  # 'n' 키로 다음 이미지로 이동
                continue


        print(k)
        cv2.destroyAllWindows()
        print(f'\n좌회전 = {count_num[0]}\n직진 = {count_num[1]}\n우회전 = {count_num[2]}')



if __name__ == "__main__":
    # 이미지 디렉토리 설정
    image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_left'
            # p1 = [55, 100]
            # p2 = [145, 150]
            # p3 = [150, 100]
            # p4 = [275, 150] # 1번
            # 좌회전 = 29
            # 직진 = 46
            # 우회전 = 0 : 75장 이미지에 대해 모두 검출
            # 40 / 35 / 0

    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/white_right'
            # p1 = [40, 100]
            # p2 = [120, 150]
            # p3 = [135, 100]
            # p4 = [250, 150] # 3번 
            # 좌회전 = 1
            # 직진 = 38
            # 우회전 = 130 : 191장 중 22장은 출력 안되는 이미지 - 169장 이미지에 대해 1장의 오차율

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
            # 0 50 139 : 191장 중 2장 검출안됨 

    # image_dir = '/home/ubuntu/beom_ws/src/lane_image/1115/yellow_straight' 
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