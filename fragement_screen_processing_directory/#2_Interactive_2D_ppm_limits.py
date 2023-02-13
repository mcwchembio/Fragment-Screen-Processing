#!/usr/bin/python3

# Create a 1D interactive plot from NMRPipe data
# Get contour levels and x, y mins

# Import needed modules
import os
import nmrglue as ng
import matplotlib.pyplot as plt
import numpy as np
import fnmatch



#Change directory to data location
os.chdir("./ft2")
print(os.getcwd())

#Pattern to find the control
search_pattern = "*" + "control" + '*' 
for file in os.listdir('./'):
	if fnmatch.fnmatch(file, search_pattern):
		control = file


#Select the file to test plotting and level parameters
dic, data = ng.pipe.read(control)



# plot parameters, set start, levels, factors proton and nitrogen dimensions
contour_start = int(input('Enter contour level: default 200000 ') or 200000) # contour level start value
contour_num = 20 # number of contour levels
contour_factor = 1.4 # scaling factor between contour levels
UH = float(input('Enter upper proton limit in ppm: default 13 ') or 13)
LH = float(input('Enter lower proton limit in ppm: default 4.78 ') or 4.78)
UN= float(input('Enter upper nitrogen limit in ppm: default 136 ') or 136)
LN = float(input('Enter lower nitrogen limit in ppm: default 102 ') or 102)


# calculate contour levels
cl = contour_start * contour_factor ** np.arange(contour_num)


# make ppm scales
uc_1H = ng.pipe.make_uc(dic, data, dim=1)
ppm_1H = uc_1H.ppm_scale()
ppm_1H_0, ppm_1H_1 = uc_1H.ppm_limits()
uc_15n = ng.pipe.make_uc(dic, data, dim=0)
ppm_15n = uc_15n.ppm_scale()
ppm_15n_0, ppm_15n_1 = uc_15n.ppm_limits()

# create the figure
fig = plt.figure()
ax = fig.add_subplot(111)

# plot the contours
ax.contour(data, cl, colors = 'r',
extent=(ppm_1H_0, ppm_1H_1, ppm_15n_0, ppm_15n_1))


# decorate the axes
ax.set_ylabel("15N (ppm)")
ax.set_xlabel("1H (ppm)")
ax.set_title(control + " Close window to quit")
ax.set_xlim(UH, LH)
ax.set_ylim(UN, LN)
plt.show()

