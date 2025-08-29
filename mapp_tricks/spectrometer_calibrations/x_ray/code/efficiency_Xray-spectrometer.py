#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Samuel Juillerat
Date: 01.07.2025

Description: This script calculates the efficiency of the X-ray spectrometer for the energy of interest.

Input:  - energy of interest in [keV]
        - level of the X-ray spectrometer (from 0 to 14)
        - radius of the source in [mm]
        - unique setup name (calls the data of this setting)
        - if the final plot should be saved or not [True/False]

Notice: The following efficiencies correspond to sources which are measured ON a 3 mm THICK COIN, 
        NOT on the GROUND PLATE of the mount!
        This is due to a ~3 mm thick calibration source.
        
        The plotted efficiency curve corresponds to a point source. If the radius is
        non-negligible the computed efficiency point will deviate from the plotted curve!

Additional Information: The script will load the computed efficiencies from another script.
                        If there is more than one file in the corresponding folder, it will load the lastest one!
                        The script also loads automatically the correct efficiency curve attenuated
                        by air for the different levels.
                        
                        This script takes some time to run due to complex calculations in the background!
"""

# %% USER INPUT
# energy of interest [keV]
E = 39.75

# level of the spectrometer {0,1,2,..,14}
level = 0

# radius of the source [mm]
radius = 0

# name of the setup (e.g. 'Gain3-398_PeakingTime1us')
setup = 'Gain3-398_PeakingTime1us'

# Want to store the final plot {True/False}?
store_plot = False

# %% FUNCTIONS
import os
import sys
import math as m
import numpy as np
from uncertainties import ufloat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from utils.functions_geometrical_factor import geometrical_factor_coin, geometrical_factor_point
from utils.functions_normalized_efficiency_detector import efficiency_normalized_accurate, efficiency_normalized_accurate_with_air


def exclude_hidden_files(files0):
    files1 = []
    for file in files0:
        if file[0] != '.':
            files1.append(file)
    return files1

def check_if_setup_exists(path0, setup0):
    setups0 = exclude_hidden_files(os.listdir(path0))
    for s in setups0:
        if s == setup0:
            return 0
    print("-> Setup '%s' not found in the directory '%s'!\n" %(setup0, path0))
    print('   Available setups:', setups0)
    print()
    sys.exit('Setup name not found!')

def check_if_level_exists(path0, level0):
    levels0 = exclude_hidden_files(os.listdir(path0))
    for s in levels0:
        if s == level0:
            return 0
    print("-> Folder '%s' not found in the directory '%s'!\n" %(level0, path0))
    print('   Available levels:', sorted(levels0))
    print()
    sys.exit('Level name not found!')

def list_text_files(files0):
    files1 = []
    for file in files0:
        if file[-4:] == '.txt':
            files1.append(file)
    return sorted(files1)

def get_text_filename(path0, level0):       # returns the last text-file of the directory
    files0 = os.listdir(path0)
    files0 = exclude_hidden_files(files0)
    files0 = list_text_files(files0)
    return files0[-1]

def get_data_peaks_nuclides(peaks0):
    nuclides0 = []
    data_peaks0 = np.zeros([len(peaks0), 4], dtype=object)
    for index, line in enumerate(peaks0):
        line_ele = line.split('\t')
        for j in range(3):
            data_peaks0[index,j] = float(line_ele[j])
        data_peaks0[index,3] = line_ele[3]
        nuclides0.append(line_ele[3].split('\n')[0])
    return data_peaks0, nuclides0

def get_nuclide_number(nuclides0):
    nuclides1 = list(set(nuclides0))
    nuclides_list = []
    for nuc0 in nuclides0:
        for index, nuc1 in enumerate(nuclides1):
            if nuc0 == nuc1:
                nuclides_list.append(index)
    nuc_array = np.zeros(len(nuclides_list))
    for i in range(len(nuclides_list)):
        nuc_array[i] = nuclides_list[i]
    return nuc_array, nuclides1

def get_cov_mat(covariance0):
    return np.array([[ float(covariance0[0].split('\t')[0]), float(covariance0[0].split('\t')[1])],
                         [ float(covariance0[1].split('\t')[0]), float(covariance0[1].split('\t')[1])]])

def load_data_file(path0, filename0):
    f = open(path0 + filename0, 'r')
    data = f.readlines()
    f.close()
    
    filename1 = data[0][23:].split('\n')[0]
    
    cov_mat0 = get_cov_mat(data[7:9])
    
    parameters0 = data[4].split('\t')
    a0, b0 = ufloat(float(parameters0[0]), np.sqrt(cov_mat0[0,0])), ufloat(float(parameters0[1]), np.sqrt(cov_mat0[1,1]))
    
    data_peaks0, nuclides0 = get_data_peaks_nuclides(data[12:])
    # data_peaks0[:,3], nuc_list0 = get_nuclide_number(nuclides0)
    
    return a0, b0, cov_mat0, data_peaks0, filename1, []

def load_data_efficiency_file(filename1, level0):
    path = '../files/SDD Efficiency files/'
    if filename1 == 'SDD_normalized-Efficiency_default.txt':
        data = np.loadtxt(path + filename1, skiprows=10)
        return np.transpose(np.vstack((data[:,0], data[:,3])))
    else:
        data = np.loadtxt(path + filename1, skiprows=1, delimiter=',')
        return np.transpose(np.vstack((data[:,0], data[:,level0+1])))

def interpolate(X, Y, X_new):
    interp_func = interp1d(X, Y, kind='linear')  # or 'cubic', 'quadratic'
    return interp_func(X_new)

def efficiency(E0, a0, b0):
    global eff_curve
    return a0.nominal_value + interpolate(eff_curve[:,0], eff_curve[:,1], E0)*b0.nominal_value

def efficiency_raw(E0):
    global eff_curve
    return interpolate(eff_curve[:,0], eff_curve[:,1], E0)

def efficiency_error(E0, a0, b0, cov_mat0):
    T = np.array([1, efficiency_raw(E0)])
    return np.sqrt(T @ cov_mat0 @ T)

def get_efficiency_and_error(E0, a0, b0, cov_mat0, state_plot):
    global eff_curve
    if state_plot == False:
        if E0 < eff_curve[0,0] or E0 > eff_curve[-1,0]:
            print('-> Energy of %.2f keV is out of range!' %(E0))
            sys.exit()
    eff0 = efficiency(E0, a0, b0)
    eff_err0 = efficiency_error(E0, a0, b0, cov_mat0)
    # print(eff, eff_err)
    return eff0, eff_err0

def compute_efficiency_and_error_plot(a0, b0, cov_mat0):
    global data_peaks, E
    E_upper = np.max([np.max(data_peaks[:,0]), E]) + 20
    E_arr0 = np.linspace(1, E_upper, 200)
    eff_arr0 = np.zeros(200,)
    eff_err_arr0 = np.zeros(200,)
    
    for i in range(200):
        eff_arr0[i], eff_err_arr0[i] = get_efficiency_and_error(E_arr0[i], a0, b0, cov_mat0, True)
    return E_arr0, eff_arr0, eff_err_arr0

def get_nuclide_color(nuc0):
    colors_list = ['tab:blue', 'tab:green', 'tab:purple', 'tab:olive', 'tab:cyan']
    for index, color in enumerate(colors_list):
        if nuc0 == index:
            return color
    return 'tab:gray'

def get_nuclide_label(nuc0, nuc_list0):
    global plot_label_list
    for num_label in plot_label_list:
        if nuc0 == num_label:
            return None
    plot_label_list.append(nuc0)
    nuc_name = nuc_list0[int(nuc0)]
    if len(nuc_name.split('-')) == 1:
        n = len(nuc_name)
        for i in range(n):
            if nuc_name[:n-i].isdecimal() == True:
                num = int(nuc_name[:n-i])
                iso = nuc_name[n-i:]
                break
        return 'Data points of $^{%d}$%s' %(num, iso)
    
    elif len(nuc_name.split('-')) == 2:
        nuc_name0 = nuc_name.split('-')[0]
        n = len(nuc_name0)
        for i in range(n):
            if nuc_name0[:n-i].isdecimal() == True:
                num = int(nuc_name0[:n-i])
                iso = nuc_name0[n-i:]
                break
        nuc_name1 = nuc_name.split('-')[1]
        if nuc_name1.lower() == 'gamma':
            dec = r'$\gamma$'
        elif nuc_name1.lower() == 'xray':
            dec = 'X-ray'
        else:
            dec = nuc_name1
        return 'Data points of $^{%d}$%s, %s' %(num, iso, dec)
    return nuc_list0[nuc0]

def return_result(eff0, eff_err0):
    num = 0
    for i in range(0,10):
        if abs(eff0*10**i) >= 1:
            num = i
            break
    return eff0*10**num, eff_err0*10**num, -num

def return_result_parameters(a0, b0):
    a_nom, a_std = a0.nominal_value, a0.std_dev
    b_nom, b_std = b0.nominal_value, b0.std_dev
    num_a, num_b = 0, 0
    for i in range(0,10):
        if abs(a_nom*10**i) >= 1:
            num_a = i
            break
    for i in range(0,10):
        if abs(b_nom*10**i) >= 1:
            num_b = i
            break
    return a_nom*10**num_a, a_std*10**num_a, -num_a, b_nom*10**num_b, b_std*10**num_b, -num_b

def return_power_of_value(integer):
    if integer < 10 and integer > 0:
        return 'e+0%d' %(integer)
    elif integer > -10 and integer < 0:
        return 'e-0%d' %(abs(integer))
    elif integer >= 10:
        return 'e+%d' %(integer)
    elif integer == 0:
        return ''
    else:
        return 'e%d' %(integer)

def return_2numbers(integer):
    if integer < 10 and integer >= 0:
        return '0%d' %(integer)
    else:
        return '%d' %(integer)

def return_3numbers(integer):
    if integer < 10 and integer >= 0:
        return '00%d' %(integer)
    elif integer >= 10 and integer < 100:
        return '0%d' %(integer)
    else:
        return '%d' %(integer)


def print_efficiency(E0, eff0, eff_err0, level0):
    eff1, eff_err1, num = return_result(eff0, eff_err0)
    result = '(%.3f ± %.3f)%s' %(eff1, eff_err1, return_power_of_value(num))
    if E0 < 25:
        print('CAUTION! Energy of %.2f keV is below lowest measured data point!' %(E0))
    print('-> The efficiency at E = %.2f keV on %s is %s.\n' %(E0, level0, result))

def convert_pTime_in_number(string):
    string0 = string.split(' ')[0]
    if len(string0.split('-')) == 2:
        value1 = int(string0.split('-')[0])
        value2 = int(string0.split('-')[1])
        value = value1 + value2*10**(-m.ceil(np.log10(value2)))
        return value
    elif len(string0.split('-')) == 1:
        return int(string0)
    else:
        return string0

def convert_pTime_list_in_numbers(pTime_list0):
    pTime_list1 = []
    for pTime0 in pTime_list0:
        pTime_list1.append(convert_pTime_in_number(pTime0))
    return sorted(pTime_list1)

def convert_peakingTime(value_pTime):
    if value_pTime == 1:
        return '1 us'
    elif value_pTime == 25.6:
        return '25-6 us'
    else:
        path = '../result files calibration/'
        pTime_list = os.listdir(path)
        pTime_list = exclude_hidden_files(pTime_list)
        pTime_list = convert_pTime_list_in_numbers(pTime_list)
        print('-> Peaking time of %g µs does not exist!' %(value_pTime))
        print('   Available peaking times:', pTime_list, 'µs\n')
        sys.exit()

def convert_level(level0):
    if level0 >= 0 and level0 <= 14 and int(level0) == level0:
        if level0 < 10:
            return 'level 0%d' %(int(level0))
        else:
            return 'level %d' %(int(level0))
    else:
        print('-> Level %g does not exist!' %(level0))
        print('   Existing levels: {0, 1, 2, ..., 14}\n')
        sys.exit()

def return_float_with_minus(value):
    string0 = '%g' %(value)
    string1 = string0.split('.')
    if len(string1) == 2:
        return  '%s-%s' %(string1[0], string1[1])
    else:
        return string1[0]

def define_filename_plot(E0, level0, setup0, radius0, path_sto):
    files_sto = os.listdir(path_sto)
    files_sto = exclude_hidden_files(files_sto)
    filename0 = 'efficiency_Level%s' %(return_2numbers(level0)) + '_%s' %(setup0) + '_Radius%smm' %(return_float_with_minus(radius0)) + '_Energy%s' %(return_float_with_minus(float(E0))) + '_'
    for i in range(1,999):
        state = False
        if i < 10:
            for file in files_sto:
                if file[:-4] == filename0 + 'v00%d' %(i):
                    state = True
                    break
            if state == False:
                return filename0 + 'v00%d.png' %(i)
        elif i < 100:
            for file in files_sto:
                if file[:-4] == filename0 + 'v0%d' %(i):
                    state = True
                    break
            if state == False:
                return filename0 + 'v0%d.png' %(i)
        else:
            for file in files_sto:
                if file[:-4] == filename0 + 'v%d' %(i):
                    state = True
                    break
            if state == False:
                return filename0 + 'v%d.png' %(i)

def get_label_Nuc(index0, Nuc_fit0):
    s = Nuc_fit0[index0].split(' ')[0]
    numbers = ''.join([ch for ch in s if ch.isdigit()])
    letters = ''.join([ch for ch in s if ch.isalpha()])
    if len(Nuc_fit0[index0].split('x-ray')) == 1:
        return r'Data points of $^{%s}$%s, $\gamma$' %(numbers, letters)
    else:
        return r'Data points of $^{%s}$%s, X-ray' %(numbers, letters)

def define_plot_label(index0, Nuc_fit0):
    if index0 == 0:
        return True
    else:
        if Nuc_fit0[index].strip() == Nuc_fit0[index-1].strip():
            return False
        else:
            return True

def get_color_Nuc(index0, Nuc_fit0):
    colors = ['tab:blue', 'tab:green', 'tab:purple', 'tab:red', 'tab:orange']
    Nuc_list0 = []
    for Nuc0 in Nuc_fit0:
        Nuc_list0.append(Nuc0.split(' ')[0].strip())
    Nuc_list1 = set(Nuc_list0)
    for index, Nuc1 in enumerate(Nuc_list1):
        if Nuc_fit0[index0].split(' ')[0].strip() == Nuc1:
            return colors[index]

def get_fmt_Nuc(index0, Nuc_fit0):
    if len(Nuc_fit0[index0].split('x-ray')) == 1:
        return '.'
    else:
        return 'x'

def pack_ufloat_array(value_row, error_row):
    n = len(value_row)
    array_ufloat = np.zeros(n, dtype=object)
    for i in range(n):
        array_ufloat[i] = ufloat(value_row[i], error_row[i])
    return array_ufloat

def return_distance_source_detector(level0, d_err0):
    return ufloat(11.4 - 3 + 10*level0, d_err0)      # [mm]

def add_correction_geometrical_factor(radius0, level0):
    if radius0 < 1E-3:
        radius0 = 1E-3
    s_dD = 0.5
    dD = return_distance_source_detector(level0, s_dD)
    fG_coin = geometrical_factor_coin(radius0, 0, dD.nominal_value, dD.std_dev)     # <<-- Error of distance necessary?!?
    fG_point = geometrical_factor_point(dD.nominal_value, 0)
    fG_coin_ufloat = ufloat(fG_coin[0], fG_coin[1])
    fG_point_ufloat = ufloat(fG_point[0], fG_point[1])
    return fG_coin_ufloat/fG_point_ufloat

def add_correction_shape_effciency_curve(E0, level0, radius0, filename0):
    if radius0 < 1E-3:
        radius0 = 1E-3
    s_d0 = 0.5
    d0 = return_distance_source_detector(level0, s_d0)
    if filename0 == 'SDD_normalized-Efficiency_point-source_all-levels.csv':
        eta_coin = efficiency_normalized_accurate(E0, d0.nominal_value, radius0)
        eta_point = efficiency_normalized_accurate(E0, d0.nominal_value, 1E-3)
        return eta_coin/eta_point
    elif filename0 == 'SDD_normalized-Efficiency_point-source_with-air_all-levels.csv':
        eta_coin = efficiency_normalized_accurate_with_air(E0, d0.nominal_value, radius0)
        eta_point = efficiency_normalized_accurate_with_air(E0, d0.nominal_value, 1E-3)
        return eta_coin/eta_point
    else:
        return 1

def get_level_string(val):
    if val == -1:
        return '-1'
    elif val < 10:
        return '0%d' %(val)
    else:
        return '%d' %(val)

def return_label_attenuation(filename0, level0):
    lvl_str = get_level_string(level0)
    if filename0 == 'SDD_normalized-Efficiency_default.txt':
        return '$f(E)$: default efficiency curve'
    elif filename0 == 'SDD_normalized-Efficiency_point-source_all-levels.csv':
        return '$f(E)$: modified efficiency curve\n    for level %s w/o attenuation' %(lvl_str)
    elif filename0 == 'SDD_normalized-Efficiency_point-source_with-air_all-levels.csv':
        return '$f(E)$: modified efficiency curve\n    for level %s w/ attenuation' %(lvl_str)



# %% EXECUTION
# get the paths and filename
path_sto = '../efficiency plots/'
path0 = '../result files calibration/'
lvl = convert_level(level)                      # convert to string
check_if_setup_exists(path0, setup)
check_if_level_exists(path0 + setup + '/', lvl)
filename_plot = define_filename_plot(E, level, setup, radius, path_sto)       # define unique filename for the plot
path = path0 + setup + '/' + lvl + '/'

# define name of file to load efficiency curve
file = get_text_filename(path, lvl)

# load the parameters and other important data
a, b, cov_mat, data_peaks, filename, nuc_list = load_data_file(path, file)

# load the normalized efficiency curve
eff_curve = load_data_efficiency_file(filename, level)

# compute the efficiency and its error
eff, eff_err = get_efficiency_and_error(E, a, b, cov_mat, state_plot=False)
eff_ufloat = ufloat(eff, eff_err)
eff_ufloat *= add_correction_geometrical_factor(radius, level)
eff_ufloat *= add_correction_shape_effciency_curve(E, level, radius, filename)
eff, eff_err = eff_ufloat.nominal_value, eff_ufloat.std_dev
print_efficiency(E, eff, eff_err, lvl)          # print the result

# compute arrays of energy, efficiency and its error for plotting
E_arr, eff_arr, eff_err_arr = compute_efficiency_and_error_plot(a, b, cov_mat)

# plot the data and the fit
plot_label_list = []    # global list
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
plt.errorbar(E, eff, eff_err, fmt='.', color='black', ecolor='black', elinewidth=1, capsize=2, zorder=6,    # plot point of interest
             label='$E_I$ = %.2f keV\n' %(E) + r'$\eta(E_I)$ = (%.3f ± %.3f) $\cdot$ 10$^{%d}$' %(return_result(eff, eff_err)))
for index, line in enumerate(data_peaks):     # plot all data points available
    plot_label = define_plot_label(index, data_peaks[:,3])
    if not plot_label:
        plt.errorbar(line[0], line[1], yerr=line[2], fmt=get_fmt_Nuc(index, data_peaks[:,3]),
                 ecolor='black', elinewidth=1, capsize=2, zorder=5, color=get_color_Nuc(index, data_peaks[:,3]))
    else:
        plt.errorbar(line[0], line[1], yerr=line[2], fmt=get_fmt_Nuc(index, data_peaks[:,3]), label = get_label_Nuc(index, data_peaks[:,3]) ,     # <<-- ADJUST!
                 ecolor='black', elinewidth=1, capsize=2, zorder=5, color=get_color_Nuc(index, data_peaks[:,3]))
plt.plot(E_arr, eff_arr, label = r'Fit: $\eta(E)$ = $a$ + $b$ $f(E)$' + '\n'        # plot the fit
         + r'$a$ = (%.3f ± %.3f) $\cdot$ 10$^{%d}$,' %(return_result(a.nominal_value, a.std_dev)) 
         + '\n' + r'$b$ = (%.3f ± %.3f) $\cdot$ 10$^{%d}$' %(return_result(b.nominal_value, b.std_dev)) 
         + '\n%s' %(return_label_attenuation(filename, level)), color = 'indianred')
plt.fill_between(E_arr, eff_arr - eff_err_arr, eff_arr + eff_err_arr, label=r'1$\sigma$ fit error (68.3% confidence level)',        # plot the error of the fit
                 color = 'lightcoral', alpha=0.4)
ax.set_yscale('log')
plt.legend()
plt.grid()
plt.xlabel('Energy $E$ [keV]')
plt.ylabel(r'Efficiency $\eta$')
plt.title('Efficiency curve on %s' %(lvl))
if store_plot == True:
    plt.savefig(path_sto + filename_plot, dpi=400)
    print("-> Plot with filename '%s' saved!" %(filename_plot))
plt.show()

