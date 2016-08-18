gps_log = open('logs/gps_log_2.txt', 'r')
parsed_file = open('logs/parsed.txt', 'w')

line = gps_log.readline()

while True:
	line = gps_log.readline()
	if not line:
		break
	if (line[0] == '\n'):
		line = gps_log.readline()
		line = gps_log.readline()
	if not line:
		break
	print line.split()[0] + ',',
	parsed_file.write(line.split()[0] + ', ',)
	line = gps_log.readline()
	print line.split()[0]
	parsed_file.write(line.split()[0] + '\n')
	# line = gps_log.readline()

print "done"