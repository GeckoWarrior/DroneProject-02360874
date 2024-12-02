
from config import Config

import numpy as np
from utilities.timing import FPS_Clock

from djitellopy import Tello
from drone_controller.drone import Drone
from natnet_client import NatNetClient
from utilities.natnet_handlers import NatNetData


def control_sim(drone: Drone):
	dt = 1/Config.FPS
	c = 0
	while not drone.reached_end() and c < Config.ITER_LIMIT:
		drone.update(dt, drone.emulateted_positions[-1], None)
		print(drone.prev_u)
		c += 1

	if c == Config.ITER_LIMIT:
		Config.REACHED_MAX_ITER = True


def control_live_nomoc(drone, tello):

	dt = 1 / Config.FPS
	clock = FPS_Clock(Config.FPS)

	tello.takeoff()
	for i in range(Config.FPS * Config.WAIT_TIME):
		clock.busy_tick()

	c = 0
	while not drone.reached_end() and not Config.EMERGENCY and c < Config.ITER_LIMIT:
		drone.update(dt, drone.emulateted_positions[-1], None)
		print(drone.prev_u)
		clock.busy_tick()
		c += 1

	if c == Config.ITER_LIMIT:
		Config.REACHED_MAX_ITER = True

	for i in range(Config.FPS * Config.WAIT_TIME):
		clock.busy_tick()
	tello.land()


def control_live_moc(drone:Drone, tello:Tello, streaming_client:NatNetClient):

	dt = 1 / Config.FPS
	clock = FPS_Clock(Config.FPS)

	print('takeoff')
	tello.takeoff()
	for i in range(Config.FPS * Config.WAIT_TIME):
		clock.busy_tick()
	print('up')

	with streaming_client:

		streaming_client.request_modeldef()
		streaming_client.update_sync()
		corner1 = np.array([4,2,2])
		corner2 = np.array([-4,-2,0])

		c = 0
		while not drone.reached_end() and not Config.EMERGENCY and c < Config.ITER_LIMIT:
			streaming_client.update_sync()

			# ===== Check NatNet Validity =====
			if not NatNetData.tracking_valid \
			or not (np.all(NatNetData.pos >= corner2) and np.all(NatNetData.pos <= corner1)):
				tello.land()
				print('NatNet failure!')
				Config.EMERGENCY = True
				continue
			# =================================

			drone.update(dt, NatNetData.pos, NatNetData.rot)
			print(drone.prev_u)
			clock.busy_tick()
			c += 1

		if c >= Config.ITER_LIMIT:
			Config.REACHED_MAX_ITER = True

		for i in range(Config.FPS * Config.WAIT_TIME):
			clock.busy_tick()
		tello.land()
