
import time


def timeit(func):
	'''
	A decorator to put on functions you wish to time their execution.
	'''
	def inner(*args, **kwargs):
		begin = time.time()
		res = func(*args, **kwargs)
		end = time.time()
		print(f"Total time of {func.__name__} is {end - begin}", flush=False)
		return res
	return inner


class FPS_Clock:
	def __init__(self, fps):
		self.fps = fps
		self.interval = 1.0 / fps
		self.last_time = time.time()
	
	def tick(self):
		current_time = time.time()
		elapsed_time = current_time - self.last_time
		wait_time = self.interval - elapsed_time
		if wait_time > 0:
			time.sleep(wait_time)
		dt = time.time() - self.last_time # total time elapsed since last call
		self.last_time = time.time()
		return dt

	def busy_tick(self):
		current_time = time.time()
		elapsed_time = current_time - self.last_time
		wait_time = self.interval - elapsed_time
		
		while wait_time > 0:
			current_time = time.time()
			elapsed_time = current_time - self.last_time
			wait_time = self.interval - elapsed_time
		
		dt = time.time() - self.last_time # total time elapsed since last call
		self.last_time = time.time()
		return dt