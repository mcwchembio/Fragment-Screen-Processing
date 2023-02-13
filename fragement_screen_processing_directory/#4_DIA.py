#!/usr/bin/python3



#import modules
import os
import fnmatch
from shutil import copyfile
import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


#Find all the subrtaced ft2 files in the subtracted_ft2 directory
ft2_files = []
for file in os.listdir("./subtracted_ft2"):
    if fnmatch.fnmatch(file, '*.ft2'):
        ft2_files.append(file)

#Sort the list of ft2 files
ft2_files.sort()

#input parameters needed for plotting
print("What type of Experiment was this")
Experiment_type = input()
contour_start = int(input('Enter contour level for difference plots: default 200000 ') or 200000) # contour level start value
upper_Hppm = float(input('Enter upper proton limit in ppm: default 13 ') or 13)
lower_Hppm = float(input('Enter lower proton limit in ppm: default 4.78 ') or 4.78)
upper_Nppm= float(input('Enter upper nitrogen limit in ppm: default 136 ') or 136)
lower_Nppm = float(input('Enter lower nitrogen limit in ppm: default 102 ') or 102)
print(contour_start)



#Funtion to cut data down to defined axis limits
def baseline_and_limits(expdata, lower_Hppm, upper_Hppm, lower_Nppm, upper_Nppm):
    x0 = uc_1H(str(upper_Hppm) + ' ppm')
    x1 = uc_1H(str(lower_Hppm) + ' ppm')
    y0 = uc_15n(str(upper_Nppm) + ' ppm')
    y1 = uc_15n(str(lower_Nppm) + ' ppm')
    cutdata = expdata[y0:y1, x0:x1]
    return(cutdata)



#Create empty lists of positive and negatige magnitudes, and indexed by the compound or matrix name
posmag = []
negmag = []
namemag = []

#Append the positive and negative magnitudes
for file in ft2_files:
    dic, data = ng.pipe.read('./subtracted_ft2/' + file)
    # make ppm scales
    uc_1H = ng.pipe.make_uc(dic, data, dim=1)
    uc_15n = ng.pipe.make_uc(dic, data, dim=0)
    limit = baseline_and_limits(data, lower_Hppm = lower_Hppm, upper_Hppm = upper_Hppm, lower_Nppm = lower_Nppm, upper_Nppm = upper_Nppm)
    posmag.append(np.sum(limit[limit > contour_start]))
    negmag.append(np.sum(limit[limit < -contour_start]))
    print(limit[limit > contour_start])
    print(limit[limit < -contour_start])
    name = file.split('_')
    namemag.append(name[3])

fig = plt.figure()
ax = fig.add_subplot(111)
plt.bar(range(1, len(posmag) + 1, 1), posmag, align = 'center',
                 color='b', label = 'Positive')
plt.bar(range(1, len(posmag) + 1, 1), negmag, align = 'center',
                 color='r', label = 'Negative')
plt.axhline(np.std(posmag + negmag) + np.mean(posmag + negmag), c ='k', ls = '--', label = '1 std')
plt.axhline(-1 * (np.std(posmag + negmag)) + np.mean(posmag + negmag), c ='k', ls = '--')
plt.xlim(0,len(posmag) + 1)

plt.axhline(2 * np.std(posmag + negmag) + np.mean(posmag + negmag), c ='k', ls = '-', label = '2 std')
plt.axhline(-2 * np.std(posmag + negmag) + np.mean(posmag + negmag), c ='k', ls = '-')
plt.xlabel('Compound')
plt.ylabel('Magnitudes')
protein_id = file.split('_')[1]
library_id = file.split('_')[2]
plt.title(Experiment_type + ' ' + library_id + ' ' + protein_id + ' DIA')
plt.xticks(np.arange(start = 1, stop = len(namemag) + 1), namemag, rotation = 70, fontsize = 7)
plt.legend(loc = 4)
plt.grid()
plt.tight_layout()
plt.savefig('./DIA/'  + str(contour_start) + '_contoured_magnitudes' + '.pdf', dpi = 600)
plt.savefig('./DIA/'  + str(contour_start) + '_contoured_magnitudes' + '.png', dpi = 600)
plt.close()
plt.clf()

collected = pd.DataFrame([namemag, posmag, negmag], index = ['Compound', 'Positive', 'Negative'])
print(collected)
collected.to_csv('./DIA/' + str(contour_start) + '_contoured_magnitudes' + '.csv', sep = ',')
