
from utilities.logger import Logger

PATH = r'C:\Tech\Courses\sem_6\DroneProject\DroneProject\final\logs\flight_1732050841.log'

def main():
	log = Logger()
	log.read_from_file(PATH)
	log.show_data()


if __name__ == "__main__":
	main()