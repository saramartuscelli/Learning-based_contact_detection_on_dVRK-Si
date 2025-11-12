#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# === CONFIGURAZIONE ===
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--file_name', type=str, required=True,
                    help='rosbag record with external forces')
arg = parser.parse_args()

file_name = arg.file_name
# === LETTURA DEL BAG ===
time_pred, val_pred, button_pred = [], [], []
time_event, val_event = [], []

bagpath = Path(file_name)
typestore = get_typestore(Stores.LATEST)

print(f"Leggo il bag: {file_name}")
with AnyReader([bagpath], default_typestore=typestore) as reader:
    # collection of all available connections (topic)
    conns = reader.connections
    # selection of the desired ones
    pred_conn = [c for c in conns if c.topic == '/contact_predictions'][0]
    gt_conn = [c for c in conns if c.topic == '/IO/IO_1/PSM2_contact'][0]

    for conn, t, rawdata in reader.messages(connections=[pred_conn]):
        msg = reader.deserialize(rawdata, conn.msgtype)
    
        # msg.axes[0] contiene y_pred, msg.buttons[0] contiene la soglia >= 0.5
        time_pred.append(t)
        val_pred.append(msg.axes[0])
        button_pred.append(msg.buttons[0])

    for conn, t, rawdata in reader.messages(connections=[gt_conn]):
        msg = reader.deserialize(rawdata, conn.msgtype)
        # Joy event (es. transizioni di contatto)
        time_event.append(t)
        val_event.append(msg.buttons[0])
        

# === CONVERSIONE A NUMPY ===
time_pred = np.array(time_pred) 
val_pred = np.array(val_pred)
# button_pred = np.array(button_pred)

button_pred = np.zeros(len(val_pred))
button_pred[val_pred >= 0.4] = 1

time_event = np.array(time_event)
val_event = np.array(val_event)

# Normalizzazione tempo relativo
t0 = min(time_pred[0], time_event[0])
time_pred -= t0
time_pred = time_pred * 1e-9
time_event -= t0
time_event = time_event * 1e-9

import numpy as np


# ricostruzione GT continua
gt = np.zeros_like(time_pred)
current_state = 0
event_index = 0

for i, t in enumerate(time_pred):
    while event_index < len(time_event) and t >= time_event[event_index]:
        current_state = val_event[event_index]
        event_index += 1
    gt[i] = current_state

gt[gt==2] = 1
# ora puoi calcolare MSE



# from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
# print("MSE:", mean_squared_error(gt, button_pred))
# print(classification_report(gt, button_pred, digits=4))
# print("CONFUSION MATRIX:\n", confusion_matrix(gt, button_pred))

start = 18050
end = 20550

# === PLOT ===
plt.figure(figsize=(11, 2.5))

# Predizioni continue (ogni 2 ms)
plt.plot(time_pred[start:end], val_pred[start:end], label='Prediction', color='black', alpha=0.3)
plt.plot(time_pred[start:end], button_pred[start:end], label='Prediction\n(thr = 0.5)', color='black', alpha=0.9)

# Eventi discreti
# plt.scatter(time_event, val_event, label='PSM2_contact event', color='red', marker='x')
plt.plot(time_pred[start:end], gt[start:end], 'g--', linewidth=4, label='Ground\nTruth')

plt.xlabel("Time [s]", fontsize=14)
plt.ylabel("Contact Probability", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()



# ============================= LATENCY CONTACT =============================
# gt_start_times = [time_event[i] for i in range(1, len(val_event))
#                   if val_event[i-1] == 0 and val_event[i] == 1]

# for i in range(len(gt_start_times)): print(gt_start_times[i])

# pred_start_times = [time_pred[i] for i in range(1, len(button_pred))
#                     if button_pred[i-1] == 0 and button_pred[i] == 1]

# print('-------------------------------')

# latencies = []
# for t_gt in gt_start_times:
#     later_preds = [t for t in pred_start_times if t >= t_gt]
#     if len(later_preds) > 0:
#         latency = later_preds[0] - t_gt
#         latencies.append(latency)

# for i in range(len(latencies)): print(latencies[i])


# ============================= LATENCY DECONTACT =============================
# gt_end_times = [time_event[i] for i in range(1, len(val_event))
#                   if val_event[i-1] == 1 and val_event[i] == 0]

# for i in range(len(gt_end_times)): print(gt_end_times[i])

# pred_end_times = [time_pred[i] for i in range(1, len(button_pred))
#                     if button_pred[i-1] == 1 and button_pred[i] == 0]

# print('-------------------------------')

# latencies = []
# for t_gt in gt_end_times:
#     later_preds = [t for t in pred_end_times if t >= t_gt]
#     if len(later_preds) > 0:
#         latency = later_preds[0] - t_gt
#         latencies.append(latency)

# for i in range(len(latencies)): print(latencies[i])


plt.show()