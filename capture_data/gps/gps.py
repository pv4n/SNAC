import os
import pynmea2

os.system('stty -F /dev/ttyUSB0 19200')
gps_stream = open('/dev/ttyUSB0', 'r')

streamreader = pynmea2.NMEAStreamReader()

log_num = 0
log_file_name = 'logs/gps_log_' + str(log_num) + '.txt'

while os.path.isfile(log_file_name):
	log_num += 1
	log_file_name = 'logs/gps_log_' + str(log_num) + '.txt'
else:
	gps_log_file = open(log_file_name, 'w')

while True:
	try:
		gps_data = gps_stream.readline()
		log_data = ''

		try:
			for msg in streamreader.next(gps_data):
				# print msg
				try:
					log_data += str(msg.timestamp) + '\n'
				except:
					pass

				try:
					print msg.num_sats
				except:
				 	pass

				try:
					log_data += str(msg.latitude) + ' '
				except:
					pass

				try:
					log_data += str(msg.lat_dir) + '\n'
				except:
					pass

				try:
					log_data += str(msg.longitude) + ' '
				except:
					pass

				try:
					log_data += str(msg.lon_dir) + '\n\n'
				except:
					pass

				if log_data:
					print log_data,
					gps_log_file.write(log_data)

		except:
			pass
	except KeyboardInterrupt:
		print "\nEnding logging, closing log file\n"
		gps_log_file.close()
		exit(0)