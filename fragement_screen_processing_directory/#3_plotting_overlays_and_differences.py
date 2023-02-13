#!/usr/bin/python3

#import modules
import os
import fnmatch
from shutil import copyfile
import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



#Find all the processed ft2 files in the ft directory
ft2_files = []
for file in os.listdir("./ft2"):
    if fnmatch.fnmatch(file, '*.ft2'):
        ft2_files.append(file)

#Identify which protein is the control
search_pattern = '*' + "control" + '*'
for file in os.listdir('./ft2'):
    if fnmatch.fnmatch(file, search_pattern):
        control = file

#Remove the control protein from the list and then sort the list
ft2_files.remove(control)
ft2_files.sort()

#Read in the control spectra
condic, condata = ng.pipe.read("./ft2/" + control)


#Prompt for plotting parameters and labeling
print("What type of Experiment was this")
Experiment_type = input()
contour_start = int(input('Enter contour level for overlays: default 200000 ') or 200000) # contour level start value
contour_start2 = int(input('Enter contour level for difference plots: default 200000 ') or 200000) # contour level start value
contour_num = int(input('Enter number of contour levels: default 20 ') or 20)
contour_factor = float(input("Enter the scaling factor to apply, 1.4?") or 1.4)
upper_Hppm = float(input('Enter upper proton limit in ppm: default 13 ') or 13)
lower_Hppm = float(input('Enter lower proton limit in ppm: default 4.78 ') or 4.78)
upper_Nppm= float(input('Enter upper nitrogen limit in ppm: default 136 ') or 136)
lower_Nppm = float(input('Enter lower nitrogen limit in ppm: default 102 ') or 102)

#Loop to walk through the ft2 files and plot them out

for file in ft2_files:
    print(file)
    #split the file name for relevant naming
    nameing = file.split('_')
    compound_id = nameing[3]
    compound_id_split = compound_id.split('.')
    if compound_id_split[-1] == 'ft2':
        compound_id_split.remove('ft2')
    compound_id = '.'.join(compound_id_split)
    protein_id = nameing[1]
    library_id = nameing[2]
    date_id = nameing[0]

    #-------------------------Standard overlay, experiment over control-----------------------------#
    #Select the file to test plotting and level parameters
    dic, data = ng.pipe.read('./ft2/' + file)

    # calculate contour levels
    cl = contour_start * contour_factor ** np.arange(contour_num)
    cl2 =np.array(list(reversed(-1*cl)))
    cl3 = np.append(cl2, cl)
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
    ax.contour(-condata, cl3, cmap='copper', linewidths = 0.4, linestyles = 'solid', vmin = -contour_start, vmax = contour_start, extent=(ppm_1H_0, ppm_1H_1, ppm_15n_0, ppm_15n_1))
    ax.contour(data, cl3, cmap='bwr', linewidths = 0.4, linestyles = 'solid', vmin = -contour_start, vmax = contour_start, extent=(ppm_1H_0, ppm_1H_1, ppm_15n_0, ppm_15n_1))
    exp = mpatches.Patch(facecolor = 'red', edgecolor='blue', label= compound_id)
    con = mpatches.Patch(facecolor='black', edgecolor = '#b87333', label='Control')
    plt.legend(handles=[exp, con], loc = 2)

    # decorate the axes
    ax.set_ylabel("15N (ppm)")
    ax.set_xlabel("1H (ppm)")
    ax.set_title(Experiment_type + ' ' + protein_id + ' ' + date_id)
    ax.set_xlim(upper_Hppm, lower_Hppm)
    ax.set_ylim(upper_Nppm, lower_Nppm)
    print('Plotting and printing ' + file)
    plt.savefig('./overlays/pdf/' + Experiment_type + '-' + protein_id + '-' + compound_id + '-' + date_id + '.pdf', dpi = 600)
    plt.savefig('./overlays/png/' + Experiment_type + '-' + protein_id + '-' + compound_id + '-' + date_id + '.png', dpi = 600)
    plt.close()
    plt.clf()

    #-------------------------Plot the difference spectrum-----------------------------#
    # create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # calculate contour levels
    cl = contour_start2 * contour_factor ** np.arange(contour_num)
    cl2 =np.array(list(reversed(-1*cl)))
    cl3 = np.append(cl2, cl)

    # Subtract the control spectrum from the experimental spectrum
    subdata = data - condata
    # Create a negative subtracted data set to plot the negative peaks seperately and in a different color
    negsubdata = -1 * subdata

    # plot the contours
    ax.contour(subdata, cl, colors = 'b', extent=(ppm_1H_0, ppm_1H_1, ppm_15n_0, ppm_15n_1))
    ax.contour(negsubdata, cl, colors = 'r',extent=(ppm_1H_0, ppm_1H_1, ppm_15n_0, ppm_15n_1))
    pos = mpatches.Patch(color='blue', label='Positive')
    neg = mpatches.Patch(color='red', label='Negative')
    plt.legend(handles=[pos, neg], loc = 2)


    # decorate the axes
    ax.set_ylabel("15N (ppm)")
    ax.set_xlabel("1H (ppm)")
    ax.set_title(Experiment_type + ' ' + protein_id + ' ' + date_id + ' ' + compound_id + ' ' + 'difference spectrum')
    ax.set_xlim(upper_Hppm, lower_Hppm)
    ax.set_ylim(upper_Nppm, lower_Nppm)
    plt.savefig('./difference_plots/pdf/' + Experiment_type + '-' + protein_id + '-' + compound_id + '-' + date_id + '-diff.pdf', dpi = 600)
    plt.savefig('./difference_plots/png/' + Experiment_type + '-' + protein_id + '-' + compound_id + '-' + date_id + '-diff.png', dpi = 600)
    plt.close()
    plt.clf()
