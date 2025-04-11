                                                                                                                                                                                                    # YUV 코드 구성
# -*- coding:utf-8 -*-
import cv2
import numpy as np

# 이미지 읽기
image = cv2.imread('/home/ubuntu/beom_ws/src/lane_image/frame000042.png')

# YUV 색 변환
yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

# Y 값 보정 (예: Y 값 1.2배 증가)
#yuv_image[:,:,0] = np.clip(yuv_image[:,:,0] * 1.0, 0, 255)

yuv_image[:,:,0] = yuv_image[:,:,0]-50

# 다시 BGR로 변환
corrected_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

# 결과 저장
cv2.imwrite('corrected_image.jpg', corrected_image)

#원본 띄우기
cv2.imshow('natural_image', image)

# 결과 띄우기
cv2.imshow('YUV_image', corrected_image)
print('y값', yuv_image[:,:,0])

cv2.waitKey(0)
cv2.destroyAllWindows()



