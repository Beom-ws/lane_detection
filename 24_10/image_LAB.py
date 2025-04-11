# -*- coding:utf-8 -*-
import cv2
import numpy as np


def apply_LAB(image):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)
    cv2.imshow('original', l)

    # 시작하기 전에 30 아래의 값들은 0으로 바꾸고 제외시킨다 

    l_high = np.copy(l)

    l[l<30] = 0

    l_high[l>225] = 255

    # np.where(조건식, 조건이 ture일 때 선택할 값, 조건이 false일 떄 선택할 값)
    L_low = np.where((l != 0), l - 30, l)
    L_high = np.where((l_high != 255), l_high + 30, l_high)

    cv2.imshow("L_low:", L_low)
    cv2.imshow("L_high:", L_high)

    # L_channel 합치기 -> lane detect하고 난 뒤에 or 이든 결과 값을 보고 수정하든 할듯
    # L_channel = cv2.bitwise_or(L_channel_low, L_channel_high)

    # 채널 병합
    lab_image_low = cv2.merge((L_low, a, b))
    lab_image_high = cv2.merge((L_high, a, b))

    # 다시 BGR로 변환
    corrected_image_low = cv2.cvtColor(lab_image_low, cv2.COLOR_LAB2BGR)
    corrected_image_high = cv2.cvtColor(lab_image_high, cv2.COLOR_LAB2BGR)

    #돌려주기
    return corrected_image_low, corrected_image_high


def show_HSV_values(event, x, y, flags, param) :
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_image = param
        hsv_value = hsv_image[y,x]
        print(f"(x,y) = ({x},{y}) / HSV = {hsv_value}")


def detect_lanes(image):
    """차선(Lane)을 검출하는 함수"""
    # 이미지를 CLAHE로 밝기와 대비 보정
    corrected_image_low ,corrected_image_high  = apply_LAB(image)

    # HSV 색상 공간 변환
    hsv_image_l = cv2.cvtColor(corrected_image_low, cv2.COLOR_BGR2HSV)
    hsv_image_h = cv2.cvtColor(corrected_image_high, cv2.COLOR_BGR2HSV)

    # 흰색 차선 검출 # dtype=np.uint8 사용이유 확실히 명시함으로써 더욱 오류의 가능성을 줄인다.
    lower_white_l = np.array([0, 0, 200], dtype=np.uint8)
    upper_white_l = np.array([180, 30, 255], dtype=np.uint8)
    white_mask_l = cv2.inRange(hsv_image_l, lower_white_l, upper_white_l)

    # 노랑색 차선 검출
    lower_yellow_l = np.array([0, 100, 100], dtype=np.uint8)
    upper_yellow_l = np.array([30, 255, 255], dtype=np.uint8)
    yellow_mask_l = cv2.inRange(hsv_image_l, lower_yellow_l, upper_yellow_l)

    # 파란색 차선 검출 (HSV 범위)
    lower_blue_l = np.array([90, 100, 100], dtype=np.uint8)
    upper_blue_l = np.array([130, 255, 255], dtype=np.uint8)
    blue_mask_l = cv2.inRange(hsv_image_l, lower_blue_l, upper_blue_l)

    # 세 가지 차선 검출 결과를 합침
    combined_mask_l = cv2.bitwise_or(white_mask_l, yellow_mask_l)
    combined_mask_low = cv2.bitwise_or(combined_mask_l, blue_mask_l)

    ##### again
    # high white
    lower_white_h = np.array([0, 0, 200], dtype=np.uint8)
    upper_white_h = np.array([180, 30, 255], dtype=np.uint8)
    white_mask_h = cv2.inRange(hsv_image_h, lower_white_h, upper_white_h)

    # high yellow
    lower_yellow_h = np.array([0, 100, 100], dtype=np.uint8)
    upper_yellow_h = np.array([30, 255, 255], dtype=np.uint8)
    yellow_mask_h = cv2.inRange(hsv_image_h, lower_yellow_h, upper_yellow_h)

    #high blue
    lower_blue_h = np.array([90, 100, 100], dtype=np.uint8)
    upper_blue_h = np.array([130, 255, 255], dtype=np.uint8)
    blue_mask_h = cv2.inRange(hsv_image_h, lower_blue_h, upper_blue_h)

    # 세 가지 차선 검출 결과를 합침
    combined_mask_h = cv2.bitwise_or(white_mask_h, yellow_mask_h)
    combined_mask_high = cv2.bitwise_or(combined_mask_h, blue_mask_h)

    return combined_mask_low, combined_mask_high


    ########수정필요
    ## low or high bitwise

    # 1번 그냥 합치기 -> 노이즈도 많이 끼고 별로인듯 보임
    #combined_mask = cv2.bitwise_or(combined_mask_low, combined_mask_high)

    # 2번, combined_mask_high와 combined_mask_low 각각 Hough Line 검출해서 일정 위치에 line가 더 긴 값을 도출하기
    #: 뭔가 노이즈에도 선이 생기고 애매모호함

    # 3번, 색변환 -> 가우시안 블러 처리 -> 케니엣지 -> 원본 케니엣지 -> 두 케니엣지 결과 비교하고 -> 더 나은 값 도출

    # 원본 이미지에 차선 검출 마스크 적용
    #detected_lanes = cv2.bitwise_and(image, image, mask=combined_mask)

    #return detected_lanes image_original

def apply_hough_line(mask):
    #Hough Line Transform을 적용하고, 검출된 선들을 반환
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=50)
    return lines

def draw_longest_lines(lines1, lines2, original_image):
    """두 개의 라인 리스트 중 더 긴 선만을 도출하여 그리기"""
    lane_image = np.zeros_like(original_image)

    def get_line_length(line):
        x1, y1, x2, y2 = line[0]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    if lines1 is not None:
        for line in lines1:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (255, 0, 0), 5)  # 파란색 선

    if lines2 is not None:
        for line in lines2:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 5)  # 초록색 선

    combined_image = cv2.addWeighted(original_image, 0.8, lane_image, 1, 0)
    return combined_image

def draw_lines(lines1, original_image):
    """두 개의 라인 리스트 중 더 긴 선만을 도출하여 그리기"""
    lane_image = np.zeros_like(original_image)

    def get_line_length(line):
        x1, y1, x2, y2 = line[0]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    if lines1 is not None:
# image_gaussian = cv2.GaussianBlur(image_original, (11, 11), 0)
# canny_edge_result = cv2.Canny(image_gaussian,100,200)
# cv2.imshow('image_gaussian', image_gaussian)
# cv2.imshow('canny_edge_result', canny_edge_result)

# canny_edge_3channel = cv2.merge([canny_edge_result, canny_edge_result, canny_edge_result])\

# image = cv2.bitwise_or(canny_edge_3channel, image_original)
        for line in lines1:
            x1, y1, x2, y2 = line[0]
            cv2.line(lane_image, (x1, y1), (x2, y2), (255, 0, 0), 5)  # 파란색 선

    combined_image = cv2.addWeighted(original_image, 0.8, lane_image, 1, 0)
    return combined_image

# 이미지 읽기
image_original = cv2.imread('/home/ubuntu/beom_ws/src/lane_image/frame000042.png')


# 이미지가 제대로 로드되지 않았을 경우 처리
if image_original is None:
    print(f"Error: 이미지를 불러올 수 없습니다. 경로를imageimage_original 확인하세요: {image_path}")
    exit()


image = np.copy(image_original)
# image_gaussian = cv2.GaussianBlur(image_original, (11, 11), 0)
# canny_edge_result = cv2.Canny(image_gaussian,100,200)
# cv2.imshow('image_gaussian', image_gaussian)
# cv2.imshow('canny_edge_result', canny_edge_result)

# canny_edge_3channel = cv2.merge([canny_edge_result, canny_edge_result, canny_edge_result])\

# image = cv2.bitwise_or(canny_edge_3channel, image_original)


#cv2.imshow('canny',canny_edge_result)

# 차선 검출
combined_mask_low, combined_mask_high = detect_lanes(image)

# 각각의 마스크에 대해 Hough Line 적용
lines_low = apply_hough_line(combined_mask_low)
lines_high = apply_hough_line(combined_mask_high)
# cv2.imshow('hough_high', lines_high)

# hough_lines_image = draw_longest_lines(lines_low, lines_high, image)
low_hough_lines_image = draw_lines(lines_low, image)
high_hough_lines_image = draw_lines(lines_high, image)
# cv2.imshow('hough_low', low_hough_lines_image)
# cv2.imshow('hough_high', high_hough_lines_image)

# 원본 띄우기
cv2.imshow('original_image', image)

#LAB_low 적용 이미지 띄우기
#cv2.imshow('lab_low', combined_mask_low)

#LAB_high 적용 이미지 띄우기
# cv2.imshow('lab_high', combined_mask_high)

# lane 합친 범위 띄우기
#cv2.imshow('lane_detect', lane)

# hough_line
#cv2.imshow('hough line', hough_lines_image)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 마우스 콜백 설정
cv2.namedWindow('original_image')
cv2.setMouseCallback('original_image', show_HSV_values, hsv_image)

# 결과 저장
#cv2.imwrite('corrected_image.jpg', lane)

cv2.waitKey(0)
cv2.destroyAllWindows()

