#!/usr/bin/env python3

import cv2
import rospy
from sensor_msgs.msg import Joy, Image
from cv_bridge import CvBridge
import numpy as np
import argparse

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp

# Initialize GStreamer
Gst.init(None)

def make_capture_pipeline(device_number):
    """
    Create a GStreamer pipeline to capture video and send it to an appsink (appsink is a sink plugin that supports
    many different methods for making the application get a handle on the GStreamer data in a pipeline).
    """

    # Pipeline string
    pipeline_str = (
        f"decklinkvideosrc device-number={device_number} ! deinterlace !"
        "videoconvert ! "
        "videocrop left=310 right=310 top=28 bottom=28 ! " # Crop to 1300x1024
        "video/x-raw,format=BGR ! " # Convert to BGR format for OpenCV
        "appsink name=appsink"
    )

    # Pipeline GStreamer creation
    pipeline = Gst.parse_launch(pipeline_str)
    # appsink properties
    appsink = pipeline.get_by_name("appsink")
    appsink.set_property("emit-signals", True)
    appsink.set_property("max-buffers", 1) # Keep only the latest frame
    appsink.set_property("drop", True) # Drop old frames
    return pipeline, appsink


def gst_to_opencv(sample):
    """
    From GStreamer sample (buffer) to OpenCV image (numpy array).
    """
    buf = sample.get_buffer()  # Get the buffer from the sample
    caps = sample.get_caps()  # Get the capabilities/properties of the sample
    height = caps.get_structure(0).get_value("height")
    width = caps.get_structure(0).get_value("width")

    arr = np.ndarray(
        shape=(height, width, 3),
        buffer=buf.extract_dup(0, buf.get_size()),
        dtype=np.uint8)
    return arr




class ContactDisplay:
    def __init__(self, mtmr, mtml):
        self.mtmr = mtmr
        self.mtml = mtml
        self.contact_map = {'PSM1': False, 'PSM2': False, 'PSM3': False}
        self.r = 0
        self.l = 0

        rospy.init_node("contact_display")
        self.bridge = CvBridge()

        # Subscriber
        for psm in ['PSM1', 'PSM2', 'PSM3']:
            if psm in mtmr or psm in mtml:
                rospy.Subscriber(f"/IO/IO_1/{psm}_contact", Joy, self.make_callback(psm))

        rospy.Subscriber("/IO/IO_1/clutch", Joy, self.clutch_callback)

        # Publisher
        self.pub_left = rospy.Publisher("/contact_display/left_image", Image, queue_size=10)
        self.pub_right = rospy.Publisher("/contact_display/right_image", Image, queue_size=10)

        # Cameras
        self.pipeline_l, self.appsink_l = make_capture_pipeline(1)
        self.pipeline_r, self.appsink_r = make_capture_pipeline(0)
        self.pipeline_l.set_state(Gst.State.PLAYING)
        self.pipeline_r.set_state(Gst.State.PLAYING)

        # Display
        cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
        self.full_screen = False
        self.w, self.h = 1300, 512
        cv2.resizeWindow("Display", self.w, self.h)

    def make_callback(self, name):
        def cb(msg):
            self.contact_map[name] = (msg.buttons[0] == 1)
        return cb

    def clutch_callback(self, msg):
        if msg.buttons[0] == 2:
            if len(self.mtmr) > 0:
                self.r = (self.r + 1) % len(self.mtmr)
            if len(self.mtml) > 0:
                self.l = (self.l + 1) % len(self.mtml)

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            sample_l = self.appsink_l.emit("pull-sample")
            sample_r = self.appsink_r.emit("pull-sample")
            if sample_l is None or sample_r is None:
                continue

            frame_l = gst_to_opencv(sample_l)
            frame_r = gst_to_opencv(sample_r)
            if frame_l.shape != frame_r.shape:
                continue

            
            # Draw rectangles
            if len(self.mtml) > 0:
                contact_left  = self.contact_map[self.mtml[self.l]]
                if contact_left:
                    cv2.rectangle(frame_l, (0,0), (0,frame_l.shape[0]), (0,0,255), 50)
                    cv2.rectangle(frame_r, (0,0), (0,frame_r.shape[0]), (0,0,255), 50)

            if len(self.mtmr) > 0:
                contact_right = self.contact_map[self.mtmr[self.r]]
                if contact_right:
                    cv2.rectangle(frame_l, (frame_l.shape[1],0), (frame_l.shape[1],frame_l.shape[0]), (0,0,255), 50)
                    cv2.rectangle(frame_r, (frame_r.shape[1],0), (frame_r.shape[1],frame_r.shape[0]), (0,0,255), 50)

            # Publish
            self.pub_left.publish(self.bridge.cv2_to_imgmsg(frame_l, "bgr8"))
            self.pub_right.publish(self.bridge.cv2_to_imgmsg(frame_r, "bgr8"))

            concat = np.hstack((frame_l, frame_r))
            cv2.imshow("Display", cv2.resize(concat, (self.w, self.h)))

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('q'):
                self.toggle_fullscreen()

            rate.sleep()

        self.shutdown()

    def toggle_fullscreen(self):
        self.full_screen = not self.full_screen
        if self.full_screen:
            self.w, self.h = 2560, 1440
            cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            self.w, self.h = 1300, 512
            cv2.setWindowProperty("Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Display", self.w, self.h)

    def shutdown(self):
        self.pipeline_l.set_state(Gst.State.NULL)
        self.pipeline_r.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--mtmr', nargs='+', default=[])
    parser.add_argument('-l', '--mtml', nargs='+', default=[])
    args = parser.parse_args()

    display = ContactDisplay(args.mtmr, args.mtml)
    display.run()
