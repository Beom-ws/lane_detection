#!/usr/bin/env python
# -*- coding:utf-8 -*-


import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import rospy

class find_roi:
    def __init__(self):
        rospy.init_node("find_roi")
        self.cvbridge = CvBridge()
        rospy.Subscriber("/camera/color/image_raw", Image, self.Image_process_callback)

        cv2.namedWindow("original")
        cv2.setMouseCallback("original", self.mouse_callback1)
        cv2.namedWindow("roi")
        cv2.setMouseCallback("roi", self.mouse_callback2)

        self.capture_count = 0

    def mouse_callback1(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Clicked at (%s, %s)" % (y, x))
            hls_pixel = self.hls[y, x]
            print("H:%s, L:%s, S:%s" % (hls_pixel[0], hls_pixel[1], hls_pixel[2]))

            capture_filename = "capture_{}.png".format(self.capture_count)
            cv2.imwrite(capture_filename, self.frame)
            print("Captured image saved as {}".format(capture_filename))
            self.capture_count += 1
    def mouse_callback2(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Clicked at (%s, %s)" % (y, x))

            capture_filename = "capture_{}.png".format(self.capture_count)
            cv2.imwrite(capture_filename, self.cropped_image)
            print("Captured image saved as {}".format(capture_filename))
            self.capture_count += 1


    def Image_process_callback(self, img):
        self.frame = self.cvbridge.imgmsg_to_cv2(img, "bgr8")
        # print(self.frame.shape)


        self.hls = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HLS)
        self.cropped_image = self.frame[420:480, 320:640]
        cv2.imshow("roi", self.cropped_image)
        cv2.imshow("original", self.frame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            cv2.destroyAllWindows()

if __name__ == "__main__":
    find_roi = find_roi()
    rospy.spin()
