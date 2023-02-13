#!/usr/bin/python3

#import modules
import os
import fnmatch
import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import pandas


#loop through and find all the ft2 files
ft2_files = []
for file in os.listdir("./ft2"):
	if fnmatch.fnmatch(file, '*.ft2'):
		ft2_files.append(file)

#sort the ft2_files list
ft2_files.sort()

#loop through the list and remove unwanted spectra
for files in ft2_files:
	print('Index number: ' + str(ft2_files.index(files)) + '	' + files)
print("Do you want to remove any spectra, Y or N")
answer = input()
while answer == "Y":
	for files in ft2_files:
		print('Index number: ' + str(ft2_files.index(files)) + '	' + files)
	print("Which spectra do you want to remove, enter the index number")
	removed_spectra = int(input())    
	ft2_files.pop(removed_spectra)
	for files in ft2_files:
		print('Index number: ' + str(ft2_files.index(files)) + '	' + files)
	print("remove more, Y or N")
	answer = input()

#prompt for parameters needed for baselining the spectra before PCA
print("What type of Experiment was this")
Experiment_type = input()
baseline = int(input('Enter contour level for overlay plots: default 100000 ') or 100000) # contour level start value
UH = float(input('Enter upper proton limit in ppm: default 13 ') or 13)
LH = float(input('Enter lower proton limit in ppm: default 4.78 ') or 4.78)
UN= float(input('Enter upper nitrogen limit in ppm: default 136 ') or 136)
LN = float(input('Enter lower nitrogen limit in ppm: default 102 ') or 102)




counter = 0

for file in ft2_files:
	#Select the file to test plotting and level parameters
	name = file.split('_')[-1]
	protein = file.split('_')[1]
	name = name.split('.')
	name.remove('ft2')
	name = '.'.join(name)
	dic, data = ng.pipe.read('./ft2/' + file)
	uc0 = ng.pipe.make_uc(dic,data,dim=0)
	uc1 = ng.pipe.make_uc(dic,data,dim=1)
	ndata = data[uc0(str(UN) + ' ppm'):uc0(str(LN) + ' ppm'), uc1(str(UH) + ' ppm'):uc1(str(LH) + ' ppm')]
	vec = np.reshape(ndata, newshape = -1)
	if counter == 0:
		compvec = vec
		counter = counter + 1 
		name_list = [name]
	else:
		compvec = np.vstack([compvec, vec]) 
		name_list.append(name)

#Determine number of components to calculate
#I want to calculate the first 15, but less if the number of experiments is less.
numrow = compvec.shape[0]
if numrow <= 15:
	numcomp = numrow - 1
else:
	numcomp = 15
listrange = np.arange(start = 1, stop = numcomp + 1)	

#PCA function
myPCA = PCA(n_components=numcomp)

#Run the PCA function and get data frames        
Comp = myPCA.fit_transform(compvec)
cumulative_sum = myPCA.explained_variance_ratio_.cumsum()
nocomponents = myPCA.n_components_ + 1

#save the Cumulative and Explained variances, and principal components into labeled pandas data frames and save them at a csv
collected = pandas.DataFrame([cumulative_sum, myPCA.explained_variance_ratio_], index = ['cumulative_variance', 'explained_variance'], columns = listrange)
print(collected)
collected.to_csv('./PCA/' + 'components_cumul_expl_variances_nobaseline' + '.csv', sep = ',')
Comp2 = pandas.DataFrame(Comp, index = name_list, columns = listrange)
Comp2.to_csv('./PCA/' + "PCA_nobaseline" + '.csv', sep = ',')


plt.plot(range(1, nocomponents, 1),cumulative_sum * 100, 'o-', markersize = 7, color = 'blue', alpha = 0.5)
plt.xlabel('Principal Component')
plt.ylabel('% Variance')
plt.title('PCA ' + protein + ' ' + Experiment_type +  ' Cumulative Variance, No Baseline')
plt.savefig('./PCA/No_baseline/pdf/' + 'Cumul_Var' + '.pdf', dpi = 600)
plt.savefig('./PCA/No_baseline/png/' + 'Cumul_Var' + '.png', dpi = 600)
plt.close()
plt.clf()
print('Plotting % Variance, No Baseline')

plt.bar(range(1, nocomponents, 1),myPCA.explained_variance_ratio_ * 100)
plt.xlabel('Principal Component')
plt.ylabel('% Variance')
plt.title('PCA ' + protein + ' ' + Experiment_type + ' Explained Variance, No Baseline')
plt.savefig('./PCA/No_baseline/pdf/' + 'Exp_Var' + '.pdf', dpi = 600)
plt.savefig('./PCA/No_baseline/png/' + 'Exp_Var'  + '.png', dpi = 600)
plt.close()
plt.clf()
print('Plotting Explainded Variance, No Baseline')

pc = range(0, 5, 1)
for i in pc:
    for j in pc:
        pci = i + 1
        pcj = j + 1
        plt.plot(Comp[:,i], Comp[:,j], 'o', markersize = 7, color = 'blue', alpha = 0.5)
        plt.xlabel('PC' + str(pci))
        plt.ylabel('PC' + str(pcj))
        plt.title('PCA ' + protein + ' ' + Experiment_type)
        for p, txt in enumerate(name_list):
            plt.annotate(txt, (Comp[:,i][p],Comp[:,j][p]))
        plt.savefig('./PCA/No_baseline/pdf/' + 'PC' + str(pci) + '_' + 'PC' + str(pcj) + '.pdf', dpi = 600)
        plt.savefig('./PCA/No_baseline/png/' + 'PC' + str(pci) + '_' + 'PC' + str(pcj) + '.png', dpi = 600)
        plt.close()
        plt.clf()
        print('Plotting ' + 'PC' + str(pci) + '_' + 'PC' + str(pcj) + 'No Baseline')

counter = 0
for file in ft2_files:
	#Select the file to test plotting and level parameters
	name = file.split('_')[-1]
	protein = file.split('_')[1]
	name = name.split('.')
	name.remove('ft2')
	name = '.'.join(name)
	dic, data = ng.pipe.read('./ft2/' + file)
	uc0 = ng.pipe.make_uc(dic,data,dim=0)
	uc1 = ng.pipe.make_uc(dic,data,dim=1)
	data[data < baseline] = 0
	ndata = data[uc0(str(UN) + ' ppm'):uc0(str(LN) + ' ppm'), uc1(str(UH) + ' ppm'):uc1(str(LH) + ' ppm')]
	vec = np.reshape(ndata, newshape = -1)
	if counter == 0:
		compvec = vec
		counter = counter + 1 
		name_list = [name]
	else:
		compvec = np.vstack([compvec, vec]) 
		name_list.append(name)

numrow = compvec.shape[0]
if numrow <= 15:
	numcomp = numrow - 1
else:
	numcomp = 15

myPCA = PCA(n_components=numcomp)

Comp = myPCA.fit_transform(compvec)
cumulative_sum = myPCA.explained_variance_ratio_.cumsum()
nocomponents = myPCA.n_components_ + 1

collected = pandas.DataFrame([cumulative_sum, myPCA.explained_variance_ratio_], index = ['cumulative_variance', 'explained_variance'], columns = listrange)
print(collected)
collected.to_csv('./PCA/' + 'components_cumul_expl_variances_baselined' + '.csv', sep = ',')
Comp2 = pandas.DataFrame(Comp, index = name_list, columns = listrange)
Comp2.to_csv('./PCA/' + "PCA_baselined" + '.csv', sep = ',')

plt.plot(range(1, nocomponents, 1),cumulative_sum * 100, 'o-', markersize = 7, color = 'blue', alpha = 0.5)
plt.xlabel('Principal Component')
plt.ylabel('% Variance')
plt.title('PCA ' + protein + ' ' + Experiment_type + ' Cumulative Variance, Baselined')
plt.savefig('./PCA/baselined/pdf/' + 'Cumul_Var' + '.pdf', dpi = 600)
plt.savefig('./PCA/baselined/png/' + 'Cumul_Var' + '.png', dpi = 600)
plt.close()
plt.clf()
print('Plotting % Variance, with Baseline')

plt.bar(range(1, nocomponents, 1),myPCA.explained_variance_ratio_ * 100)
plt.xlabel('Principal Component')
plt.ylabel('% Variance')
plt.title('PCA ' + protein + ' ' + Experiment_type + ' Explained Variance, Baselined')
plt.savefig('./PCA/baselined/pdf/' + 'Exp_Var' + '.pdf', dpi = 600)
plt.savefig('./PCA/baselined/png/' + 'Exp_Var'  + '.png', dpi = 600)
plt.close()
plt.clf()
print('Plotting Explainded Variance, with Baseline')

pc = range(0, 5, 1)
for i in pc:
    for j in pc:
        pci = i + 1
        pcj = j + 1
        plt.plot(Comp[:,i], Comp[:,j], 'o', markersize = 7, color = 'blue', alpha = 0.5)
        plt.xlabel('PC' + str(pci))
        plt.ylabel('PC' + str(pcj))
        plt.title('PCA ' + protein + ' ' + Experiment_type + ', baselined')
        for p, txt in enumerate(name_list):
            plt.annotate(txt, (Comp[:,i][p],Comp[:,j][p]))
        plt.savefig('./PCA/baselined/pdf/' + 'PC' + str(pci) + '_' + 'PC' + str(pcj) + '.pdf', dpi = 600)
        plt.savefig('./PCA/baselined/png/' + 'PC' + str(pci) + '_' + 'PC' + str(pcj) + '.png', dpi = 600)
        plt.close()
        plt.clf()
        print('Plotting ' + 'PC' + str(pci) + '_' + 'PC' + str(pcj) + 'with Baseline')


