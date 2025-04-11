import cv2
import numpy as np
import os
import math
import time



class LaneDetector:
    def __init__(self, video_path):
        if not os.path.exists(video_path):
            raise ValueError("miss video path")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("can't open video")

        # cv2.namedWindow("hls img")
        # cv2.namedWindow("lab img")
        # cv2.setMouseCallback("hls img", self.mouse_callback1)
        # cv2.setMouseCallback("lab img", self.mouse_callback2)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.angle_data = []
        self.previous_x = []
        self.angle = 0


    # def mouse_callback1(self, event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         print("Clicked at (%s, %s)" % (y, x))
    #         hls_pixel = self.hls_img[y, x]
    #         print("H:%s, S:%s, V:%s" % (hls_pixel[0], hls_pixel[1], hls_pixel[2]))
    #
    # def mouse_callback2(self, event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         print("Clicked at (%s, %s)" % (y, x))
    #         lab_pixel = self.lab_img[y, x]
    #         print("H:%s, S:%s, V:%s" % (lab_pixel[0], lab_pixel[1], lab_pixel[2]))



    def process_video(self):

        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                break

            # frame = cv2.resize(frame, (int(self.width/2), int(self.height/2)))

            cv2.imshow('original', frame)

            start_t = time.time()
            # self.img_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            # l1_channel, a_channel, b_channel = cv2.split(self.img_lab)
            self.img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h1_channel, s1_channel, v1_channel = cv2.split(self.img_hsv)
            self.img_hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            h2_channel, l2_channel, s2_channel = cv2.split(self.img_hls)
            self.img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y_channel, u_channel, v2_channel = cv2.split(self.img_yuv)

            # lab_combined = np.hstack((l1_channel, a_channel, b_channel))
            hsv_combined = np.hstack((h1_channel, s1_channel, v1_channel))
            hls_combined = np.hstack((h2_channel, l2_channel, s2_channel))
            yuv_combined = np.hstack((y_channel, u_channel, v2_channel))

            # cv2.imshow("lab Channels", lab_combined)
            # cv2.imshow("hsv Channels", hsv_combined)
            # cv2.imshow("hls Channels", hls_combined)
            # cv2.imshow("yuv Channels", yuv_combined)

            s2_channel_int = s2_channel.astype(np.int16)
            s1_channel_int = s1_channel.astype(np.int16)
            v2_channel_int = v2_channel.astype(np.int16)
            s_s = s2_channel_int - s1_channel_int
            s_s[s_s < 0] = 0
            # s_v = s2_channel_int - v2_channel_int
            # s_v[s_v < 0] = 0

            s_s = s_s.astype(np.uint8)
            # s_v = s_v.astype(np.uint8)


            # cv2.imshow('s-v', s_v)

            cv2.imshow('s-s', s_s)

            # kernel = np.ones((3, 3), np.uint8)
            # # Erosion (침식)
            # eroded_image = cv2.erode(s_s, kernel, iterations=1)
            # cv2.imshow("eroded img", eroded_image)
            kernel = np.ones((7, 7), np.uint8)
            # Dilation (팽창)
            dilated_image = cv2.dilate(s_s, kernel, iterations=1)
            dilated_image = cv2.dilate(dilated_image, kernel, iterations=1)
            kernel = np.ones((7, 7), np.uint8)
            eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
            eroded_image = cv2.erode(eroded_image, kernel, iterations=1)
            morphology_img = cv2.dilate(eroded_image, kernel, iterations=1)
            cv2.imshow('morphology Image', morphology_img)

            blob_img = morphology_img
            _, blob_img = cv2.threshold(blob_img, 150, 255, cv2.THRESH_BINARY)
            kernel = np.ones((7, 7), np.uint8)
            blob_img = cv2.erode(blob_img, kernel, iterations=1)

            cv2.imshow('aa', blob_img)
            contours, _ = cv2.findContours(blob_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contour_img = np.zeros_like(blob_img)
            cv2.drawContours(contour_img, contours, -1, (255), thickness=cv2.FILLED)


            cv2.imshow('contour_img', contour_img)


            # edge = cv2.Canny(dilated_image, 100, 150)
            # cv2.imshow('edge', edge)











            # self.hls_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
            # cv2.imshow('hls img', self.hls_img)
            # height, width = self.hls_img.shape[:2]
            #
            # lower_white = np.array([15, 0, 220])
            # upper_white = np.array([70, 15, 255])
            #
            # mask_white = cv2.inRange(self.hls_img, lower_white, upper_white)
            #
            #
            # cv2.imshow('mask_white', mask_white)



            cv2.waitKey(0)

        self.cap.release()
        cv2.destroyAllWindows()





if __name__ == "__main__":
    video_path = './0828_video.avi'
    lane_detector = LaneDetector(video_path)
    lane_detector.process_video()
