
# ==== General ====
from config import Config
from utilities.obs_generator import *
from utilities.logger import Logger
import signal
import time

# ======= Path =======
from path_planning.SPA import plan_smooth_path

# ==== NatNet ====
from natnet_client import NatNetClient
from utilities import natnet_handlers

# ==== Drone ====
from drone_controller.drone import Drone
from drone_controller import control_modes
from djitellopy import Tello


def main():

	# =========== Plan Path ===========

	#obstacles = generate_obstacles(Config.OBS_SEED, Config.OBS_NUM)
	obstacles = challenge_map2() #+ FRAME
	
	try:
		spl = plan_smooth_path(obstacles, Config.START, Config.END)
	except:
		print("Unable to find suitable path")
		plot_obstacles(obstacles, Config.START, Config.END, None)
		exit(1)

	for point in spl.c:
		point[2] = Config.FLIGHT_HEIGHT

	plot_obstacles(obstacles, Config.START, Config.END, spl)
	decision = input("Do you wish to proceed? (Y/N)")
	if decision.lower() == "y":
		print("good, let's go!")
	else:
		exit(1)

	# =========== Initiate systems ===========

	# ==== Init optitrack ====
	streaming_client = None
	if(Config.USE_NATNET):
		streaming_client = NatNetClient(server_ip_address=natnet_handlers.SERVER_IP, local_ip_address=natnet_handlers.CLIENT_IP, use_multicast=False)
		streaming_client.on_data_description_received_event.handlers.append(natnet_handlers.receive_new_desc)
		streaming_client.on_data_frame_received_event.handlers.append(natnet_handlers.receive_new_frame)

	# ==== Init tello ====
	tello = None
	if(Config.USE_TELLO):
		tello = Tello()
		tello.connect()
		print("Battery:", tello.get_battery())

		def signal_handler(signum, frame):
			print("^C")
			print(f"preforming emergency landing")
			tello.land()
			Config.EMERGENCY = True

		signal.signal(signal.SIGINT, signal_handler)

	# ===== Drone Controller ====

	control_mode = Drone.SIM
	if(Config.USE_TELLO):
		control_mode = Drone.SIM_LIVE

	logger = Logger(obstacles, spl)
	drone = Drone(tello, spl, control_mode, logger)

	# =========== Initialization Completed ===========

	# =========== Start Flight ===========
	try:
		if Config.USE_TELLO and Config.USE_NATNET:
			control_modes.control_live_moc(drone, tello, streaming_client)
		elif Config.USE_TELLO:
			control_modes.control_live_nomoc(drone, tello)
		else:
			control_modes.control_sim(drone)
	except Exception as e:
		print('failed to follow path, reason: {}'.format(str(e)))
		tello.land()

	if Config.EMERGENCY:
		print('Exited because EMERGENCY')
	elif Config.REACHED_MAX_ITER:
		print('Exited because REACHED_MAX_ITER')
	else:
		print('Completed!')
	
	# =========== Flight Completed ===========

	# =========== Save Logs & Show Results ===========
	logger.save_as_file(time.time())
	logger.show_data()


if __name__ == "__main__":
	main()
