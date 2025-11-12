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
import matplotlib.pyplot as plt
import argparse

from geometry_msgs.msg import Vector3Stamped, PoseStamped, TwistStamped
from sensor_msgs.msg import JointState, Joy


lock = threading.Lock()

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--file_name', type=str, required=True, help='name of the file you want to save your data to')
arg = parser.parse_args()


# Global variables for callbacks
tip_ori = np.zeros(3)
lin_vel = np.zeros(3)
ang_vel = np.zeros(3)
joint_vel = np.zeros(3)
joint_eff = np.zeros(3)
contact = np.zeros(1)
data = []


def accel_cb(msg: Vector3Stamped):
    global data
    raw = np.array([msg.header.stamp.to_sec(), msg.vector.x, msg.vector.y, msg.vector.z])
    feature_vec = np.concatenate([raw, tip_ori, lin_vel, ang_vel, joint_vel, joint_eff, contact])
    data.append(feature_vec)


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
    global contact
    with lock:
        contact[0] = msg.buttons[0]

def save_data():

    time.sleep(2)

    data_n = np.array(data)
    print(data_n)

    t_sensor = data_n[:, 0] - data_n[0, 0]
    acc_data = data_n[:, 1:4]
    contacts = data_n[:, -1]
    eff = data_n[:, -4:-1]

    fig, axs = plt.subplots(3, 1, sharex=True)
    for j in range(3):
        axs[j].plot(t_sensor, acc_data[:, j], alpha=1)
        axs2 = plt.twinx(axs[j])
        axs2.plot(t_sensor, contacts, color='k')
        axs[j].set_ylabel(f'Acc [$m/s^2$]')
    axs[2].set_xlabel('Time [s]')
    fig.suptitle('ACCELEROMETER DATA')
    plt.tight_layout()

    fig, axs = plt.subplots(3, 1, sharex=True)
    for j in range(3):
        axs[j].plot(t_sensor, eff[:, j], color='purple')
        axs[j].set_ylabel(f'J{j}')
    axs[2].set_xlabel('Time [s]')
    fig.suptitle('JOINT EFFORT')
    plt.tight_layout()

    plt.show()

    header = ('timestamps_sensor,'
          'contacts,'
          'acc_x,acc_y,acc_z,'
          'tip_euler_x,tip_euler_y,tip_euler_z,'
          'tip_v_x,tip_v_y,tip_v_z,'
          'tip_w_x,tip_w_y,tip_w_z,'
          'j_vel_x,j_vel_y,j_vel_z,'
          'j_eff_x,j_eff_y,j_eff_z')
    
    file_name = arg.file_name
    
    save = input('Do you want to save your data? [y/n]\t')
    if save == 'y':
        np.savetxt(file_name + '.csv', data_n, delimiter=',', header=header, comments='')
        print(f"All data saved in {file_name}'.csv'.")
    else:
        print('Data not saved.')



if __name__ == "__main__":
    rospy.init_node("data_collection")
    rospy.Subscriber("/accelerometer/data", Vector3Stamped, accel_cb, queue_size=1)
    rospy.Subscriber("/PSM2/local/measured_cp", PoseStamped, orientation_cb, queue_size=1)
    rospy.Subscriber("/PSM2/local/measured_cv", TwistStamped, twist_cb, queue_size=1)
    rospy.Subscriber("/PSM2/measured_js", JointState, joint_cb, queue_size=1)
    rospy.Subscriber("/IO/IO_1/PSM2_contact", Joy, contact_cb, queue_size=1)

    rospy.on_shutdown(save_data)

    rospy.spin()

