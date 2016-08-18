from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains
import time

import os
import cv2

# Make sure permissions are 777 or it wont work...
# Also, selenium needs absolute paths
if not os.path.exists('streetcam_frames'):
	os.mkdir('streetcam_frames')

chromedriver = '/home/ajay/GoogleDrive/workspace/SNAC/511_traffic_camera/chromedriver'
os.environ["webdriver.chrome.driver"] = chromedriver

driver = webdriver.Chrome(chromedriver)
driver.get("http://www.511nj.org/cameras.aspx?default=NJ%20Turnpike%20Tour")
#assert "Traffic Count Using BS" in driver.title

time.sleep(1)
dropdown = driver.find_element_by_id('cphContent_ddlRoadway')
dropdown.send_keys(Keys.SPACE)
time.sleep(1)
for i in range(4):
	dropdown.send_keys(Keys.DOWN)

dropdown.send_keys(Keys.RETURN)

time.sleep(1)
actions = ActionChains(driver)

for i in range(10):
	actions.send_keys(Keys.TAB)
actions.send_keys(Keys.RETURN)
actions.perform()

time.sleep(4)

#record the camera and click to continuously stream
iframe = driver.find_element_by_id('EvetnsMap')
x = 0
print("hi")
while (1):
	time.sleep(1/30)
	iframe.click()
	driver.get_screenshot_as_png()
	filename = '/home/ajay/GoogleDrive/workspace/SNAC/511_traffic_camera/streetcam_frames/{0:05d}.png'.format(x)

	driver.save_screenshot(filename)
	cmd = "mogrify -crop 356x230+306+369 /home/ajay/GoogleDrive/workspace/SNAC/511_traffic_camera/streetcam_frames/{0:05d}.png".format(x)
	os.system(cmd)
	# window = cv2.imread(filename, cv2.IMREAD_COLOR)
	# window = window[375:220, 304:355]
	# cv2.imwrite(filename, window)
	x+=1