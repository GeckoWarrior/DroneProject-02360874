
import numpy as np
from djitellopy import Tello
from scipy.interpolate import BSpline
from utilities.timing import timeit
from utilities.logger import Logger


class Drone:

	# flight modes
	SIM = 0
	LIVE = 1
	SIM_LIVE = 2

	# drone parameters
	v_max = 8  # m/s
	v_mean = 3  # m/s, stable on turns
	max_power = 100

	# flight parameters
	braking_distance = 0.5 # m
	error_distance = 0.15 # m, how far from the endpoint is considered as a reach.
	safe_stopping_speed = 0.1 # m/s, a speed where the drone momentum is irrelevant.

	def __init__(self, tello:Tello, spl:BSpline, mode = SIM, logger:Logger = None):
		self.tello = tello
		self.spl = spl
		self.prev_u = 0
		self.emulateted_positions = [spl(0)]
		self.mode = mode
		self.log = logger
		self.last_position = spl(0)

	@timeit
	def update(self, dt, position:np.ndarray, rotation):
		u = self.prev_u
		reach = Drone.v_mean * dt
		
		if 1 - u < 0.1: # slow down
			reach *= ((1 - u)/0.2 + 0.5)
			print('slowing down')

		epsilon = 1e-4  # Small increment

		# find the furthest point on the apline that is in reach
		du = epsilon
		p = self.spl(u)
		# Reach is calculated as a 3D box, in order to make the axis independant.
		while np.all(np.abs(p - position) < reach) and (u + du) < 1: 
			du += epsilon
			p = self.spl(u + du)

		# vel is the direction needed in the world coords in the correct magnitude
		vel = (self.spl(u + du - epsilon) - position) / dt
		self.prev_u = u + du - epsilon
		
		# update expected new position
		cmd = None
		if self.mode == Drone.SIM or self.mode == Drone.SIM_LIVE:
			self.emulate_new_position(dt, position, vel)

		# send a matching command to the drone
		if self.mode == Drone.LIVE or self.mode == Drone.SIM_LIVE:
			# translate to a command and send
			x_pow, y_pow, z_pow = (vel / Drone.v_max) * Drone.max_power  # change this
			self.tello.send_rc_control(-int(y_pow), int(x_pow), int(z_pow), 0)
			cmd = (-int(y_pow), int(x_pow), int(z_pow))

		# log the current frame if needed
		if self.log is not None:
			aim = self.spl(self.prev_u) #(vel * dt) + position
			self.log.add(position, aim, cmd)
		
		self.last_velocity = (position - self.last_position) / dt
		self.last_position = position

	def emulate_new_position(self, dt, pos, vel):
		self.emulateted_positions.append((vel * dt) + pos)
	
	def reached_end(self):
		return self.prev_u + 1e-3 >= 1 \
				and np.linalg.norm(self.last_position - self.spl(1)) <= Drone.error_distance \
				and np.all(np.abs(self.last_velocity) <= Drone.safe_stopping_speed)
	