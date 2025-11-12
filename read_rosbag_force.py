#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--file_name', type=str, required=True,
                    help='rosbag record with external forces')
arg = parser.parse_args()

file_name = arg.file_name

cf_x_sensor = []
cf_y_sensor = []
cf_z_sensor = []
ct_x_sensor = []
ct_y_sensor = []
ct_z_sensor = []
t_sensor = []

t_pred, val_pred, button_pred = [], [], []

bagpath = Path(file_name)
typestore = get_typestore(Stores.LATEST)

print(f"Leggo il bag: {file_name}")


with AnyReader([bagpath], default_typestore=typestore) as reader:
    # collection of all available connections (topic)
    conns = reader.connections
    # selection of the desired ones
    pred_conn = [c for c in conns if c.topic == '/contact_predictions'][0]
    cf_conn = [c for c in conns if c.topic == '/measured_cf'][0]

    for conn, t, rawdata in reader.messages(connections=[pred_conn]):
        msg = reader.deserialize(rawdata, conn.msgtype)
        # msg.axes[0] contiene y_pred, msg.buttons[0] contiene la soglia >= 0.5
        t_pred.append(t)
        val_pred.append(msg.axes[0])
        button_pred.append(msg.buttons[0])

    for conn, t, rawdata in reader.messages(connections=[cf_conn]):
        msg = reader.deserialize(rawdata, conn.msgtype)
        cf_x_sensor.append(msg.wrench.force.x)
        cf_y_sensor.append(msg.wrench.force.y)
        cf_z_sensor.append(msg.wrench.force.z)
        ct_x_sensor.append(msg.wrench.torque.x)
        ct_y_sensor.append(msg.wrench.torque.y)
        ct_z_sensor.append(msg.wrench.torque.z)
        t_sensor.append(t)


t_pred = np.array(t_pred)
val_pred = np.array(val_pred)
button_pred = np.array(button_pred)

cf_x_sensor = np.array(cf_x_sensor)
cf_y_sensor = np.array(cf_y_sensor)
cf_z_sensor = np.array(cf_z_sensor)
ct_x_sensor = np.array(ct_x_sensor)
ct_y_sensor = np.array(ct_y_sensor)
ct_z_sensor = np.array(ct_z_sensor)
t_sensor = np.array(t_sensor)

t0 = min(t_pred[0], t_sensor[0])
t_pred -= t0
t_pred = t_pred * 1e-9
t_sensor -= t0
t_sensor = t_sensor* 1e-9

mask_pred = (t_pred > 115) & (t_pred < 178)
mask_sensor = (t_sensor > 115) & (t_sensor < 178)

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True)
axs[0].plot(t_pred[mask_pred], val_pred[mask_pred], label='Predictions', color='black', alpha=0.3)
axs[0].plot(t_pred[mask_pred], button_pred[mask_pred], label='Predictions\n(t=0.5)', color='black')
axs[0].plot(t_sensor[mask_sensor], cf_x_sensor[mask_sensor], label='F/T Sensor', color='mediumblue')
axs[0].set_ylabel('Force X [N] -\nContact Probability', fontsize=13)
axs[0].grid(True)
axs[0].legend()

axs[1].plot(t_pred[mask_pred], val_pred[mask_pred], label='Predictions', color='black', alpha=0.3)
axs[1].plot(t_pred[mask_pred], button_pred[mask_pred], label='Predictions\n(t=0.5)', color='black')
axs[1].plot(t_sensor[mask_sensor], cf_y_sensor[mask_sensor], label='F/T Sensor', color='mediumblue')
axs[1].set_ylabel('Force Y [N] -\nContact Probability', fontsize=13)
axs[1].grid(True)

axs[2].plot(t_pred[mask_pred], val_pred[mask_pred], label='Predictions', color='black', alpha=0.3)
axs[2].plot(t_pred[mask_pred], button_pred[mask_pred], label='Predictions\n(t=0.5)', color='black')
axs[2].plot(t_sensor[mask_sensor], cf_z_sensor[mask_sensor], label='F/T Sensor', color='mediumblue')
axs[2].set_ylabel('Force Z [N] -\nContact Probability', fontsize=13)
axs[2].grid(True)

plt.xlabel("Tempo [s]", fontsize=13)
plt.tight_layout()
plt.show()

