#!/usr/bin/python3

print("************************************************************")
print("*						          *")
print("*	Use to Process Bruker 2D NMR data		  *")
print("*							  *")
print("* Be sure to set the name of the fid.com file, pipe script *")
print("*	and subdirectory of the experiment.		  *")
print("*							  *")
print("************************************************************")



#Enter which directory to process, name of fid file and processing file
exp_to_process = '11'
fid = 'fid.com'
pip = 'nmr_ft.com'





#Load needed modules
import os
import shutil 
import fnmatch
import subprocess
import nmrglue as ng

#Set the working directory and make the fid.com and pipescript executable
top_wd = os.getcwd()
os.chmod(fid, 0o777)
os.chmod(pip, 0o777)

#Pattern to find the experimental directories, matches directory that start with full date, Year/Month/Date
search_pattern = "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]" + '_' + '*' + '_' + '*'+ '_' + '*' 
Exp_dir = []
for file in os.listdir(top_wd + "/../"):
	if fnmatch.fnmatch(file, search_pattern):
		Exp_dir.append(file)

#Identify the control spectrum
search_pattern ='*' + "control" + '*' 
for file in os.listdir(top_wd + "/../"):
	if fnmatch.fnmatch(file, search_pattern):
		control = file
Exp_dir.remove(control)


#Sort the experiment list
Exp_dir.sort()

# Process and move Control Spectrum to ft2 directory
os.chdir(top_wd + '/../' + control + '/' + exp_to_process)
shutil.copyfile(top_wd + '/' + fid, './' + fid)
shutil.copyfile(top_wd + '/' + pip, './' + pip)
os.chmod(fid, 0o777)
os.chmod(pip, 0o777)
os.system('./' + fid)
os.system('./' + pip)
naming = control.split('_')
name = (naming[0], naming[1], naming[2], naming[3])
shutil.copyfile('test.ft2', top_wd + '/ft2/' + '_'.join(name) + '.ft2')
condic, condata = ng.pipe.read('test.ft2')
os.system('rm ./test.ft2')
os.chdir(top_wd)


# Walk through the experimental directories, process and move to ft2 directory
# Then do the subtractions and move to the subtracted_ft2 directory
for Exp in Exp_dir:
	print(Exp)
	os.chdir(top_wd + '/../' + Exp + '/' + exp_to_process)
	shutil.copyfile(top_wd + '/' + fid, './' + fid)
	shutil.copyfile(top_wd + '/' + pip, './' + pip)
	os.chmod(fid, 0o777)
	os.chmod(pip, 0o777)
	os.system('./' + fid)
	os.system('./' + pip)
	naming = Exp.split('_')
	name = (naming[0], naming[1], naming[2], naming[3])
	shutil.copyfile('test.ft2', top_wd + '/ft2/' + '_'.join(name) + '.ft2')
	dic, data = ng.pipe.read('test.ft2')
	subdata = data - condata
	name = (naming[0], naming[1], naming[2], naming[3], '-control')
	ng.pipe.write(top_wd + '/subtracted_ft2/' + '_'.join(name) + '.ft2', dic, subdata, overwrite=True)
	os.system('rm ./test.ft2')
	os.chdir(top_wd)
    



