
from natnet_client import DataDescriptions, DataFrame
import math
import numpy as np


# ===== settings =====

SERVER_IP = "132.68.35.30"
CLIENT_IP = "132.68.35.95"
NAME = 'tali'
RIGID_BODY_ID = 550
RATE = 5

# ===============

def mul_q(q1, q2):
	w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
	w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

	w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
	x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
	y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
	z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

	new_q = (w, x, y, z)

	return new_q

# ===== handlers =====

def receive_new_frame(data_frame: DataFrame):
	Logs.num_frames += 1
	if (Logs.num_frames % RATE == 0):
		for t in data_frame.rigid_bodies:
			if t.id_num == RIGID_BODY_ID:
				pos = np.array([t.pos[0], -t.pos[2], t.pos[1]])
				q1 = t.rot
				q2 = (math.cos(-math.pi / 4), 0, 0, math.sin(-math.pi / 4))
				rot = np.array(mul_q(q1, q2))
				NatNetData.pos = pos
				NatNetData.rot = rot
				NatNetData.tracking_valid = t.tracking_valid
				
				if Logs.active == True:
					Logs.positions.append(pos)
					Logs.rotations.append(rot)
					Logs.lost_tracking += 1


def receive_new_desc(desc: DataDescriptions):
	print("Received data descriptions.")


# ===== data structures =====
class NatNetData():

	pos = np.array((0,0,0))
	rot = np.array((0,0,0,0))
	tracking_valid = True


class Logs():

	active = True
	num_frames = 0
	positions = []
	rotations = []
	lost_tracking = 0

	@staticmethod
	def reset_log():
		Logs.num_frames = 0
		Logs.positions = []
		Logs.rotations = []
		Logs.lost_tracking = 0

