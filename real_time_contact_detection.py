#!/usr/bin/env python3

import rospy
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
from collections import deque
import queue
import numpy as np
import math
import time
import threading
# import tensorflow as tf
# import keras
import onnxruntime as ort
import matplotlib.pyplot as plt
import argparse

from geometry_msgs.msg import Vector3Stamped, PoseStamped, TwistStamped
from sensor_msgs.msg import JointState, Joy

# ============== PARSER =============
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--window_size', type=int, required=True, help='model input size')
parser.add_argument('-m', '--model', type=str, required=False, help='keras model path')
args = parser.parse_args()


# Data preparation variables
n_features = 18
window_size = args.window_size + 50
max_train =  np.array([np.float64(3.485053130513092), np.float64(9.439692061353348), np.float64(7.193834298072096),
                      np.float64(3.1415809715948155), np.float64(1.1308120366551533), np.float64(3.141533942597323),
                      np.float64(0.13043081760406494), np.float64(0.07509862631559373), np.float64(0.0971694439649582),
                      np.float64(3.967198848724365), np.float64(3.684584379196167), np.float64(4.2239089012146),
                      np.float64(0.8709573149681091), np.float64(0.5697018504142761), np.float64(0.07879145443439484),
                      np.float64(6.9430832862854), np.float64(13.5460205078125), np.float64(14.33579444885254)])
min_train =  np.array([np.float64(-3.6350660006094886), np.float64(-7.993727202915909), np.float64(-7.222530866210083),
                      np.float64(-3.141562093804221), np.float64(-0.97103923676692), np.float64(-3.1415258431977446),
                      np.float64(-0.07290492206811905), np.float64(-0.05183118209242821), np.float64(-0.09277243167161939),
                      np.float64(-2.82474136352539), np.float64(-4.987372875213623), np.float64(-11.021486282348633),
                      np.float64(-0.9773582816123961), np.float64(-0.7394002676010131), np.float64(-0.0926249623298645),
                      np.float64(-10.00264835357666), np.float64(-10.09317398071289), np.float64(-13.32011032104492)])


# Circular buffer to store data
buffer = deque(maxlen=window_size)
# fill buffer to avoid latency
for i in range(window_size):
    buffer.append(np.zeros(n_features))


# Thread variables
sample_queue = queue.Queue(maxsize=10)
data_queue = queue.Queue(maxsize=3)
lock = threading.Lock()

# Filter initialization
fs = 498.0
fc = 0.5
order = 2
b, a = butter(order, fc/(fs/2), btype='high', analog=False)

# Global variables for callbacks
tip_ori = np.zeros(3)
lin_vel = np.zeros(3)
ang_vel = np.zeros(3)
joint_vel = np.zeros(3)
joint_eff = np.zeros(3)

# List for saving all outputs
contact_stats_list = []
prediction_time_list = []


def accel_cb(msg: Vector3Stamped):
    raw = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
    with lock:
        feature_vec = np.concatenate([raw, tip_ori, lin_vel, ang_vel, joint_vel, joint_eff])

    try:
        sample_queue.put_nowait(feature_vec)
    except queue.Full:
        pass


def orientation_cb(msg: PoseStamped):
    global tip_ori
    quat = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    with lock:
        tip_ori[:] = R.from_quat(quat).as_euler('xyz', degrees=False)


def twist_cb(msg: TwistStamped):
    global lin_vel, ang_vel
    with lock:
        lin_vel[:] = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
        ang_vel[:] = [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]


def joint_cb(msg: JointState):
    global joint_vel, joint_eff
    with lock:
        joint_vel[:] = msg.velocity[:3]
        joint_eff[:] = msg.effort[:3]


def contact_cb(msg: Joy):
    if msg.buttons[0] == 1:
        contact_stats_list.append("\n-------------------------------------------------")
        contact_stats_list.append(["[GT] CONTACT: ", time.time()])

# ===== PRODUCER =====
def producer():
    global b, a
    counter = 0

    while not rospy.is_shutdown():
        feature_vec_producer = sample_queue.get()
        buffer.append(feature_vec_producer)
        counter += 1

        if counter == 10:  # every 20 samples (~40 ms)
            counter = 0
            X = np.array(buffer)  # (win, 18)
            acc_filt = filtfilt(b, a, X[:, :3], axis=0)
            X[:, :3] = acc_filt
            X_scaled = (X[25:125] - min_train) / (max_train - min_train) * 2 - 1
            X_scaled = X_scaled[np.newaxis, ...].astype(np.float32)
            try:
                data_queue.put_nowait(X_scaled)
            except queue.Full:
                pass

        sample_queue.task_done()


# ===== CONSUMER =====
def consumer(session):
    while not rospy.is_shutdown():
        start = time.time()
        for i in range(1000):
            X_consumer = data_queue.get()
            # model.predict(X_consumer, verbose=0)
            # y_pred = model.predict(X_consumer, verbose=0)

            # prediction_time_list.append(["Start: ", time.time()])
            y_pred = session.run([output_name], {input_name: X_consumer})[0]
            # prediction_time_list.append(["End: ", time.time()])

            # print("Prediction:", y_pred)
            if y_pred >= 0.5:
                contact_stats_list.append(["[PRED] CONTACT: ", time.time()])
            data_queue.task_done()

        end = time.time()
        print("average prediction time: ", (end - start)/1000)


def save_lists():

    with open("contact_stats.txt", 'w') as file:
        for item in contact_stats_list:
            file.write(f"{item}\n")

    # with open("prediction_times.txt", 'w') as file:
    #     for item in prediction_time_list:
    #         file.write(f"{item}\n")

    print("Lists saved.")


# ===== MAIN =====
if __name__ == "__main__":
    rospy.init_node("contact_prediction")

    # class DummyModel:
    #     def predict(self, X, verbose=0):
    #         print(X.shape)

    # model = DummyModel()

    # model = tf.keras.models.load_model(args.model)

    # with open("model.json", "r") as json_file:
    #     model_json = json_file.read()

    # model = tf.keras.models.model_from_json(model_json)
    # model.load_weights("model_weights.weights.h5")

    session = ort.InferenceSession("best_model_cpu.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    rospy.Subscriber("/accelerometer/data", Vector3Stamped, accel_cb, queue_size=1)
    rospy.Subscriber("/PSM2/local/measured_cp", PoseStamped, orientation_cb, queue_size=1)
    rospy.Subscriber("/PSM2/local/measured_cv", TwistStamped, twist_cb, queue_size=1)
    rospy.Subscriber("/PSM2/measured_js", JointState, joint_cb, queue_size=1)
    rospy.Subscriber("/IO/IO_1/PSM2_contact", Joy, contact_cb, queue_size=1)

    t_prod = threading.Thread(target=producer, daemon=True)
    t_cons = threading.Thread(target=consumer, args=(session,), daemon=True)
    t_prod.start()
    t_cons.start()

    rospy.on_shutdown(save_lists)

    rospy.spin()
