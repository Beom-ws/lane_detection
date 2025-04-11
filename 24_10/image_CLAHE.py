import cv2
import numpy as np
import time


def apply_clahe(image):
    """YUV 색 공간에서 CLAHE 적용"""
    YUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(YUV)

    global start_T
    start_T = time.time()

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(Y)

    global end_T 
    end_T = time.time()
    

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
    gaussian_image = cv2.GaussianBlur(Y_trans_uint8, (7,7) ,2)

    return gaussian_image



def auto_canny(image, sigma = 0.33):
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    v = np.median(blurred)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(blurred, lower, upper)
    print('lower: %d  upper: %d' % (lower, upper))

    return edged





# def detect_lanes(image):
#     """차선(Lane)을 검출하는 함수"""
#     # 이미지를 CLAHE로 밝기와 대비 보정
#     corrected_image = apply_clahe(image)
#     cv2.imshow('clahe', corrected_image)

#     # HSV 색상 공간 변환
#     hsv_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2HSV)

#     # 흰색 차선 검출 # dtype=np.uint8 사용이유 확실히 명시함으로써 더욱 오류의 가능성을 줄인다.
#     lower_white = np.array([0, 0image
#     lower_yellow = np.array([15, 100, 100], dtype=np.uint8)
#     upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
#     yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

#     # 파란색 차선 검출 (HSV 범위)
#     lower_blue = np.array([90, 100, 100], dtype=np.uint8)
#     upper_blue = np.array([130, 255, 255], dtype=np.uint8)
#     blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

#     # 세 가지 차선 검출 결과를 합침
#     combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
#     combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

#     # 원본 이미지에 차선 검출 마스크 적용
#     detected_lanes = cv2.bitwise_and(image, image, mask=combined_mask)

#     return detected_lanes



# 이미지 파일을 읽기
image = cv2.imread('/home/ubuntu/beom_ws/src/lane_image/Screenshot from 2024-11-07 15-07-10.png')

# 이미지가 제대로 로드되지 않았을 경우 처리
if image is None:
    print(f"Error: 이미지를 불러올 수 없습니다. 경로를 확인하세요: {image_path}")
    exit()

# # 차선 검출 수행
# lanes = detect_lanes(image)

clahe_image = apply_clahe(image)

# 차선 검출 수행
lanes = auto_canny(clahe_image)


# 결과 화면 출력
cv2.imshow('orignal Lanes', image)
cv2.imshow('Detected Lanes', lanes)
cv2.imshow('CLAHE Lanes', clahe_image)

# 종료
cv2.waitKey(0)
cv2.destroyAllWindows()
print(end_T - start_T)