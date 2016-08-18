import subprocess
import threading

class GPS_Thread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.setDaemon(True)
	def run(self):
		GPS()

class Record_Thread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.setDaemon(True)
	def run(self):
		Record()

def Record():
	print "Staring recording"
	subprocess.call('./capture_30.sh')

def GPS():
	print "Starting GPS"
	subprocess.call(['python', 'gps/gps.py'])

gps_thread = GPS_Thread()
gps_thread.start()

record_thread = Record_Thread()
record_thread.start()