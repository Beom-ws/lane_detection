import cv2
import numpy as np

# 이미지 읽기
image = cv2.imread('/home/ubuntu/beom_ws/src/lane_image/frame000042.png')

# 이미지를 LAB 색상 공간으로 보정하는 함수
def apply_LAB(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    print('original', l)

    # L 채널에서 30 이하 값을 0으로, 225 이상 값을 255로 처리
    l_high = np.copy(l)
    l[l < 30] = 0
    l_high[l > 225] = 255

    # L 채널에서 30을 더하고 뺀 값을 사용
    L_low = np.where((l != 0), l - 30, l)
    L_high = np.where((l_high != 255), l_high + 30, l_high)

    print("L_low:", L_low)
    print("L_high:", L_high)

    # LAB 채널 병합
    lab_image_low = cv2.merge((L_low, a, b))
    lab_image_high = cv2.merge((L_high, a, b))

    # 다시 BGR로 변환
    corrected_image_low = cv2.cvtColor(lab_image_low, cv2.COLOR_LAB2BGR)
    corrected_image_high = cv2.cvtColor(lab_image_high, cv2.COLOR_LAB2BGR)

    return corrected_image_low, corrected_image_high

# 마우스 콜백 함수 (클릭한 위치의 HSV 값을 출력)
def show_HSV_values(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스가 클릭할 때
        hsv_image = param  # 전달된 파라미터는 HSV 이미지
        hsv_value = hsv_image[y, x]  # 마우스 좌표의 HSV 값
        print(f"(x,y) = ({x}, {y}) / HSV = {hsv_value}")
    


# 차선 검출 함수
def detect_lanes(image):
    corrected_image_low, corrected_image_high = apply_LAB(image)

    # HSV 색상 공간 변환
    hsv_image_l = cv2.cvtColor(corrected_image_low, cv2.COLOR_BGR2HSV)
    hsv_image_h = cv2.cvtColor(corrected_image_high, cv2.COLOR_BGR2HSV)

    # low와 high 차선 검출 마스크 결합
    lower_white_l = np.array([0, 0, 200], dtype=np.uint8)
    upper_white_l = np.array([180, 30, 255], dtype=np.uint8)
    white_mask_l = cv2.inRange(hsv_image_l, lower_white_l, upper_white_l)

    lower_yellow_l = np.array([10, 50, 100], dtype=np.uint8)
    upper_yellow_l = np.array([30, 150, 255], dtype=np.uint8)
    yellow_mask_l = cv2.inRange(hsv_image_l, lower_yellow_l, upper_yellow_l)

    lower_blue_l = np.array([90, 100, 100], dtype=np.uint8)
    upper_blue_l = np.array([130, 255, 255], dtype=np.uint8)
    blue_mask_l = cv2.inRange(hsv_image_l, lower_blue_l, upper_blue_l)

    combined_mask_low = cv2.bitwise_or(white_mask_l, yellow_mask_l)
    combined_mask_low = cv2.bitwise_or(combined_mask_low, blue_mask_l)

    # high 영역도 같은 방식으로 처리
    lower_white_h = np.array([0, 0, 200], dtype=np.uint8)
    upper_white_h = np.array([180, 30, 255], dtype=np.uint8)
    white_mask_h = cv2.inRange(hsv_image_h, lower_white_h, upper_white_h)

    lower_yellow_h = np.array([15, 100, 100], dtype=np.uint8)
    upper_yellow_h = np.array([35, 255, 255], dtype=np.uint8)
    yellow_mask_h = cv2.inRange(hsv_image_h, lower_yellow_h, upper_yellow_h)

    lower_blue_h = np.array([90, 100, 100], dtype=np.uint8)
    upper_blue_h = np.array([130, 255, 255], dtype=np.uint8)
    blue_mask_h = cv2.inRange(hsv_image_h, lower_blue_h, upper_blue_h)

    combined_mask_high = cv2.bitwise_or(white_mask_h, yellow_mask_h)
    combined_mask_high = cv2.bitwise_or(combined_mask_high, blue_mask_h)

    # low와 high 마스크 결합
    combined_mask = cv2.bitwise_or(combined_mask_low, combined_mask_high)

    # 원본 이미지에 차선 검출 마스크 적용
    detected_lanes = cv2.bitwise_and(image, image, mask=combined_mask)

    return detected_lanes

# 이미지 읽기
image = cv2.imread('/home/ubuntu/beom_ws/src/lane_image/frame000042.png')

# 이미지가 제대로 로드되지 않았을 경우 처리
if image is None:
    print("Error: 이미지를 불러올 수 없습니다.")
    exit()

# 차선 검출
lane = detect_lanes(image)

# LAB 보정 이미지
corrected_image_low, corrected_image_high = apply_LAB(image)

# HSV 이미지 변환
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 마우스 콜백 설정 (hsv_image의 HSV 값을 확인)
cv2.namedWindow('original_image')
cv2.setMouseCallback('original_image', show_HSV_values, hsv_image)

# 결과 창 띄우기
cv2.imshow('original_image', image)
cv2.imshow('lab_low', corrected_image_low)
cv2.imshow('lab_high', corrected_image_high)
cv2.imshow('lane_detect', lane)

cv2.waitKey(0)
cv2.destroyAllWindows()
