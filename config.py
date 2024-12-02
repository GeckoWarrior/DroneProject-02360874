
import numpy as np

class Config():

	# Map Parameters
	START 		= np.array([-3, -1])	#np.array([-3,-1])	#np.array([-3.5, -0.8])
	END 		= np.array([3, 1])	#np.array([-3,1])	#np.array([3.5, 1])
	OBS_SEED	= 74328
	OBS_NUM		= 5

	# Systems
	USE_NATNET 	= True
	USE_TELLO 	= True

	# Tello parameters
	FPS 		= 10	# command rate, tello's buffer behaves well with 10
	WAIT_TIME 	= 1		# time for stabelizing
	
	# Flight parameters
	ITER_LIMIT 		= 600	# max flight time = ITER_LIMIT/FPS
	FLIGHT_HEIGHT 	= 0.7 	# meters. desired flight height

	# flags - state related (do not change)
	EMERGENCY 			= False
	REACHED_MAX_ITER 	= False