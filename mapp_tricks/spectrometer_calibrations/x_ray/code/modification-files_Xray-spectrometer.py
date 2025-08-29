#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Samuel Juillerat
Date: 18.08.2025

Description: This script performs a modification of the peak files of InterSpec.
             It adds required information to the files (e.g. real time, peaking
             time, etc.) such that they can be used for the calibration of the
             X-ray spectrometer.

User input: - Choose if it should modifiy all files automatically or folder-wise per input [True/False]

Output: Saves the modified files in the correct directory for further processing.

Notice: Only files get modifed which are placed in the 'unread' folder!
        If a file got modified the raw file get moved from the 'unread' to the 'read' folder.
"""

# %% INPUT
# choose if the script should modifiy all files automatically or manually (folder-wise) {True/False} (DEFAULT = True)
automatic_execution = True

# %% FUNCTIONS
import os
import shutil
import sys

def exclude_hidden_files(files0):
    # function for excluding hidden files of the list
    # returns list without hidden files if there were any
    files1 = []
    for file in files0:
        if file[0] != '.':
            files1.append(file)
    return files1

def read_file(file0):
    global path_unr
    f = open(path_unr + file0, 'r')
    lines0 = f.readlines()
    f.close()
    return lines0

def modify_data(lines0, acc_time0, real_time0, sCounts0, fCounts0, peaking_time0, ftWi0, gain0):
    header = [0, 1, 2, 10, 11, 12, 13, 14]
    lines1 = []
    for j in range(len(lines0)):
        line1 = ''
        line0 = lines0[j]
        for i in range(0,3):
            line1 += line0.split(',')[header[i]]
            line1 += ','
        if j == 0:
            line1 += ', , '
        elif j == 1:
            line1 += 'AccumulatedTime, RealTime, '
        else:
            line1 += '%.3f, %.3f,' %(acc_time0, real_time0)
        for i in range(3,8):        
            line1 += line0.split(',')[header[i]]
            line1 += ','
        if j == 0:
            line1 += ','
        elif j == 1:
            line1 += 'slowCounts, fastCounts, peakingTime, flatTopWidth, Gain'
        else:
            line1 += '%d, %d, %f, %f, %g' %(sCounts0, fCounts0, peaking_time0, ftWi0, gain0)
        lines1.append(line1)
    return lines1

def write_modified_data_to_file(file0, lines0):
    global path_sto
    os.makedirs(path_sto, exist_ok=True)
    filename = file0.split('.')[0]
    f = open(path_sto + filename + '_modified.CSV', 'w')
    for line in lines0:
        f.write(line + '\n')
    f.close()

def get_sCount(data1):
    # data1 = data0.split('\n')
    for line in data1:
        if line[:10].lower() == 'slow count':
            value = int(line.split(':')[1])
            return value
    print('Slow Counts not found!')
    sys.exit('Slow Count value not callable')

def get_fCount(data1):
    # data1 = data0.split('\n')
    for line in data1:
        if line[:10].lower() == 'fast count':
            value = int(line.split(':')[1])
            return value
    print('Fast Counts not found!')
    sys.exit('Fast Count value not callable')

def get_aTime(data1):
    # data1 = data0.split('\n')
    for line in data1:
        if line[:17].lower() == 'accumulation time':
            value = float(line.split(':')[1])
            return value
    print('Accumulation Time not found!')
    sys.exit('Accumulation Time value not callable')

def get_rTime(data1):
    # data1 = data0.split('\n')
    for line in data1:
        if line[:9].lower() == 'real time':
            value = float(line.split(':')[1])
            return value
    print('Real Time not found!')
    sys.exit('Real Time value not callable')

def get_pTime(data1):
    # data1 = data0.split('\n')
    for line in data1:
        if line[:4].lower() == 'tpea':
            value = float(line.split(';')[0].split('=')[1])
            return value
    print('Peaking Time not found!')
    sys.exit('Peaking Time value not callable')

def get_ftWi(data1):
    # data1 = data0.split('\n')
    for line in data1:
        if line[:4].lower() == 'tfla':
            value = float(line.split(';')[0].split('=')[1])
            return value
    print('Flat Top Width not found!')
    sys.exit('Flat Top Width value not callable')

def get_gain(data1):
    # data1 = data0.split('\n')
    for line in data1:
        if line[:4].lower() == 'gain':
            value = float(line.split(';')[0].split('=')[1])
            return value
    print('Gain not found!')
    sys.exit('Gain value not callable')

def return_CONFIGURATION(list_data):
    i0, i1 = 0, -1
    for index, line in enumerate(list_data):
        if line == '<<DP5 CONFIGURATION>>\n':
            i0 = index
        if line == '<<DP5 CONFIGURATION END>>\n':
            i1 = index
    return list_data[i0+1:i1]

def return_STATUS(list_data):
    i0, i1 = 0, -1
    for index, line in enumerate(list_data):
        if line == '<<DPP STATUS>>\n':
            i0 = index
        if line == '<<DPP STATUS END>>\n':
            i1 = index
    return list_data[i0+1:i1]

def check_existence_file(path_files_InterSpec, file0):
    for file in os.listdir(path_files_InterSpec):
        if file == file0:
            return True
    print("-> File '%s' not found in folder 'raw mca files'!" %(file0))
    sys.exit()

def get_required_data_from_file(path_files_InterSpec, file0):
    filename0 = file0.split('.')[0][6:] + '.mca'
    check_existence_file(path_files_InterSpec, filename0)
    f = open(path_files_InterSpec + filename0, 'r', encoding='latin1')
    data0 = f.readlines()
    f.close()
    data_CONF, data_STAT = return_CONFIGURATION(data0), return_STATUS(data0)
    acc_time0 = get_aTime(data_STAT)
    real_time0 = get_rTime(data_STAT)
    sCounts0 = get_sCount(data_STAT)
    fCounts0 = get_fCount(data_STAT)
    peaking_time0 = get_pTime(data_CONF)
    ftWi0 = get_ftWi(data_CONF)
    gain0 = get_gain(data_CONF)
    print("\n   Extraced values of '%s': " %(file0))
    print('   Accumulated Time: %g s' %(acc_time0))
    print('   Real Time: %g s' %(real_time0))
    print('   Slow Counts: %d' %(sCounts0))
    print('   Fast Counts: %d' %(fCounts0))
    print('   Peaking time: %g µs' %(peaking_time0))
    print('   Flat Top Width: %g µs' %(ftWi0))
    print('   Gain: %g' %(gain0))
    return acc_time0, real_time0, sCounts0, fCounts0, peaking_time0, ftWi0, gain0

def get_required_data(file0):
    print('-> File: %s' %(file0))
    data_file = input("\n   Enter the 'File Remarks': ")
    data_spectra = input("\n   Enter the 'Spectra Remarks': ")
    acc_time0 = get_aTime(data_file)
    real_time0 = get_rTime(data_file)
    sCounts0 = get_sCount(data_file)
    fCounts0 = get_fCount(data_file)
    peaking_time0 = get_pTime(data_spectra)
    ftWi0 = get_ftWi(data_spectra)
    gain0 = get_gain(data_spectra)
    print('\n   Extraced values: ')
    print('   Accumulated Time: %g s' %(acc_time0))
    print('   Real Time: %g s' %(real_time0))
    print('   Slow Counts: %d' %(sCounts0))
    print('   Flow Counts: %d' %(fCounts0))
    print('   Peaking time: %g µs' %(peaking_time0))
    print('   Flat Top Width: %g µs' %(ftWi0))
    print('   Gain: %g' %(gain0))
    return acc_time0, real_time0, sCounts0, fCounts0, peaking_time0, ftWi0, gain0

def get_accTime_realTime_counts(file0):
    state = False
    print('-> File: %s' %(file0))
    while(state == False):
        acc_time0 = float(input('   Enter the accumulated time in seconds: '))
        real_time0 = float(input('   Enter the real time in seconds: '))
        counts0 = int(input('   Enter the total counts: '))
        peaking_time0 = float(input('   Enter the peaking time [µs]: '))
        print('\n   Are these values correct? [Enter = yes]')
        print('   Accumulated Time: %g s' %(acc_time0))
        print('   Real Time: %g s' %(real_time0))
        print('   Counts: %d' %(counts0))
        print('   Peaking time: %g µs' %(peaking_time0))
        state_str = input('   ')
        if state_str == '':
            state = True
    return acc_time0, real_time0, counts0, peaking_time0

def create_folders(path_list):
    for path in path_list:
        os.makedirs(path, exist_ok = True)

def get_subfolder():
    sub_folder_str = input('Enter the folder name of the files: ')
    print('')
    if sub_folder_str == '':
        return ''
    else:
        return sub_folder_str + '/'

def move_files_to_folder(path_folder, path_loc, path_dest):
    os.makedirs(path_dest, exist_ok=True)
    shutil.move(path_loc, path_dest)
    if os.path.isdir(path_folder)  and not exclude_hidden_files(os.listdir(path_folder)):
        # hidden_files = os.listdir(path_folder)
        for file in os.listdir(path_folder):
            os.remove(path_folder + file)
        os.rmdir(path_folder)

def delete_folder_if_empty(path_folder):
    if os.path.isdir(path_folder)  and not exclude_hidden_files(os.listdir(path_folder)):
        # hidden_files = os.listdir(path_folder)
        for file in os.listdir(path_folder):
            os.remove(path_folder + '/' + file)
        os.rmdir(path_folder)


# %% EXECUTION
if automatic_execution:
    while True:         # complicated loop which looks into all folders, searches files, modifies them and afterwards deletes the folder
        sub_folder, file = '', ''
        path_stage0 = '../raw files/raw peak files/unread'
        folder0, folder1, folder2, folder3 = '', '', '', ''
        folder_stage0 = exclude_hidden_files(os.listdir(path_stage0))
        if not not folder_stage0:
            folder0 = folder_stage0[0]
            if os.path.isfile(path_stage0 + '/' + folder0):
                sub_folder = ''
                file = folder0
            else:
                folder_stage1 = exclude_hidden_files(os.listdir(path_stage0 + '/' + folder0))
                if not not folder_stage1:
                    folder1 = folder_stage1[0]
                    if os.path.isfile(path_stage0 + '/' + folder0 + '/' + folder1):
                        sub_folder = folder0 + '/'
                        file = folder1
                    else:
                        folder_stage2 = exclude_hidden_files(os.listdir(path_stage0 + '/' + folder0 + '/' + folder1))
                        if not not folder_stage2:
                            folder2 = folder_stage2[0]
                            if os.path.isfile(path_stage0 + '/' + folder0 + '/' + folder1 + '/' + folder2):
                                sub_folder = folder0 + '/' + folder1 + '/'
                                file = folder2
                        else:
                            delete_folder_if_empty(path_stage0 + '/' + folder0 + '/' + folder1)
                else:
                    delete_folder_if_empty(path_stage0 + '/' + folder0)
        else:
            break
        
        if not not file:
            path_file_InterSpec = '../raw files/raw mca files/'            # path of InterSpec files
            path_unr = '../raw files/raw peak files/unread/' + sub_folder  # path of unread file
            path_r = '../raw files/raw peak files/read/' +  sub_folder     # path to move unread file
            path_sto = '../modified peak files/' + sub_folder                   # path to store new file
            
            create_folders([path_unr, path_r, path_sto])        # create the required folders
            
            files = os.listdir(path_unr)                # list all files in the unread folder
            files = exclude_hidden_files(files)         # exclude hidden files from the list
            
            acc_time, real_time, sCounts, fCounts, pTime, ftWi, gain = get_required_data_from_file(path_file_InterSpec, file)   # extract data from the InterSpec file
            lines = read_file(file)     # read the unread .CSV file
            lines = modify_data(lines, acc_time, real_time, sCounts, fCounts, pTime, ftWi, gain)        # append data with important parameters
            write_modified_data_to_file(file, lines)        # write data to new file
            move_files_to_folder(path_unr, path_unr + file, path_r)     # move unread file to the 'read' folder and delete old folder if empty
            print("\n   -> unread file moved to the 'read' folder\n")   # print success

else:
    sub_folder = get_subfolder()        # get sub-folder via user input
    
    path_file_InterSpec = '../raw files/raw mca files/'            # path of InterSpec files
    path_unr = '../raw files/raw peak files/unread/' + sub_folder  # path of unread file
    path_r = '../raw files/raw peak files/read/' +  sub_folder     # path to move unread file
    path_sto = '../modified peak files/' + sub_folder                   # path to store new file
    
    create_folders([path_unr, path_r, path_sto])        # create the required folders
    
    files = os.listdir(path_unr)                # list all files in the unread folder
    files = exclude_hidden_files(files)         # exclude hidden files from the list
    
    for file in files:
        acc_time, real_time, sCounts, fCounts, pTime, ftWi, gain = get_required_data_from_file(path_file_InterSpec, file)   # extract data from the InterSpec file
        lines = read_file(file)     # read the unread .CSV file
        lines = modify_data(lines, acc_time, real_time, sCounts, fCounts, pTime, ftWi, gain)        # append data with important parameters
        write_modified_data_to_file(file, lines)        # write data to new file
        move_files_to_folder(path_unr, path_unr + file, path_r)     # move unread file to the 'read' folder and delete old folder if empty
        print("\n   -> unread file moved to the 'read' folder\n")   # print success

