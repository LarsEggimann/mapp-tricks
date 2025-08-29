#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Samuel Juillerat
Date: 18.08.2025

Description: This script performs the calibration of the modified peak informaion
             files (peak files of InterSpec which were modified with 'modification-files_Xray-spectrometer.py').

User input:
    Basic:  - setup to calibrate
            - statement for calibrating all levels of the setup automatically [True/False]
            - statement for saving the calibration results [True/False]
    Advanced:   - Statement if the X-rays of the files should be used [True/False]
                - Statement if the gammas of the files should be used [True/False]
                - Statement if the default efficiency curve of Amptek should be used [True/False]
                - Statement if the attenuation of the air should be added [True/False]

Output: Saves a plot and the corresponding parameters of the fit.

Notice: Since actual sources have a finite diameter, all peak areas get scaled
        such that the measured source would correspond to a point source.
        Like that the peak areas can be fitted to different efficiency curves which
        are also computed for point sources.

Additional Information: The default efficiency curve of Amptek is not that precise!
                        With 'state_default = False', efficiency curves computed for
                        all levels are used, taking the source radius into account
                        to provide a more accurate result.
                        Also the attenuation of the air can be included for a even
                        better fit.
"""

# %% USER INPUT
# %%% BASIC
# Setup name for the calibration
setup = 'Gain3-398_PeakingTime25-6us'

# Should all levels be calibrated automatically? (DEFAULT = True)
automatic = False

# Should it be saved? (DEFAULT = True)
store = False

# %%% ADVANCED
# Should the X-rays be used? (DEFAULT = False)
state_Xrays = False

# Should the gammas be used? (DEFAULT = True)
state_Gammas = True

# Want to fit the default efficiency curve of Amptek (prependicular incoming rays w/o attenuation)? (DEFAULT = False)
state_default = False

# Want to add the attenuation of the air? (DEFAULT = True)
state_air = True


# %% FUNCTIONS
import os
import sys
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from uncertainties import umath as um
from utils.functions_geometrical_factor import geometrical_factor_coin, geometrical_factor_point
from utils.functions_normalized_efficiency_detector import efficiency_normalized_accurate, efficiency_normalized_accurate_with_air

EXP = np.vectorize(um.exp)

def _exclude_hidden_files(files0):
    files1 = []
    for file in files0:
        if file[0] != '.':
            files1.append(file)
    return sorted(files1)

def _load_files(path_files0):
    files0 = os.listdir(path_files0)
    return _exclude_hidden_files(files0)

def _load_Source_data():
    path_filename = '../files/Source-data_Xray-spectrometer.csv'
    f = open(path_filename, 'r')
    data_f0 = f.readlines()
    f.close()
    return data_f0

def _return_Source_data_of_file(path_files0, files0, Source_data_raw0):
    n = len(files0)
    m = len(Source_data_raw0)
    Source_names = []
    Source_data0 = []
    for i in range(n):
        Source_names.append(files0[i].split('_')[1])
    for i in range(n):
        for j in range(1,m):
            if Source_names[i] == Source_data_raw0[j].split(',')[0].split(' ')[0]:
                Source_data0.append(Source_data_raw0[j])
    return Source_data0

def _load_Source_data_for_all_files(path_files0, files0):
    Source_data_raw = _load_Source_data()
    return _return_Source_data_of_file(path_files0, files0, Source_data_raw)

def _check_elements_list_equal(List0):
    state = True
    element0 = List0[0]
    for element in List0:
        if element != element0:
            state = False
    if state == False:
        print('-> CAUTION: Files contain different levels!')
        sys.exit()

def _get_level_of_files(files0):
    levels0 = []
    for file in files0:
        if file.split('_')[2][:6] == 'level-':
            levels0.append(int(file.split('_')[2].split('-')[1]))
        elif file.split('_')[2][:5] == 'level':
            levels0.append(int(file.split('_')[2][5:]))
        else:
            print("-> Level not found in '%s'!" %(file))
            levels0.append(-1)
    _check_elements_list_equal(levels0)
    return levels0

def _get_single_level_of_files(files0):
    levels0 = []
    for file in files0:
        if file.split('_')[2][:6] == 'level-':
            levels0.append(int(file.split('_')[2].split('-')[1]))
        elif file.split('_')[2][:5] == 'level':
            levels0.append(int(file.split('_')[2][5:]))
        else:
            print("-> Level not found in '%s'!" %(file))
            levels0.append(-1)
    _check_elements_list_equal(levels0)
    return levels0[0]

def _calculate_time_of_string(time_str0):
    unit = time_str0[-1]
    if unit == 'y':
        fac = 86400*365.2425
    elif unit == 'd':
        fac = 86400
    elif unit == 'h':
        fac =  3600
    elif unit == 'm':
        fac = 60
    elif unit == 's':
        fac = 1
    else:
        print("\n-> Error of the half life unit in the file 'source-data_Xray-spectrometer.csv'!")
        print("   Make sure that it is written in the form '10.551 y'\n")
        sys.exit('Wrong input: %s' %(time_str0))
    number = float(time_str0[:-1])
    return fac*number

def _get_thickness_plastic(plastic, plastic_d, plastic_s_d):
    if plastic == 'False':
        return False, 0
    elif plastic == 'True':
        return True, ufloat(float(plastic_d), float(plastic_s_d))
    else:
        print("\n-> Error of the plastic statement in the file 'source-data_Xray-spectrometer.csv'!")
        print("Only 'True' and 'False' are valid arguments!")
        sys.exit('Wrong input: %s' %(plastic))

def _get_decay_parameters_Source(file_Source0):
    file_Source0_split = file_Source0.split('\n')[0].split(',')
    Nuc = file_Source0_split[1]
    RS = float(file_Source0_split[2])
    Act = float(file_Source0_split[3])
    s_Act = float(file_Source0_split[4])
    date = file_Source0_split[5]
    time = file_Source0_split[6]
    HL_str = file_Source0_split[7]
    s_HL_str = file_Source0_split[8]
    plastic = file_Source0_split[9]
    plastic_d = file_Source0_split[10]
    plastic_s_d = file_Source0_split[11]
    
    A0 = ufloat(Act, s_Act)
    t0_str_0 = datetime.strptime(time + ' ' + date, "%H:%M:%S %d.%m.%Y")
    T12_0 = ufloat(_calculate_time_of_string(HL_str), _calculate_time_of_string(s_HL_str))
    Lambda_0 = np.log(2)/T12_0
    state_plastic, d_plastic = _get_thickness_plastic(plastic, plastic_d, plastic_s_d)
    
    return Nuc, A0, Lambda_0, t0_str_0, RS, [state_plastic, d_plastic]

def _get_month(month_str):
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    m = month_str.lower()
    for index, month in enumerate(months[:8], start=1):
        if m == month:
            return '0%d' %index
    for index, month in enumerate(months[8:], start=10):
        if m == month:
            return '%d' %index

def _convert_date_time_in_duration(date_str0, time_str0, t0_str_0):
    # get time
    h = time_str0.split(':')[0]
    mi = time_str0.split(':')[1]
    sec = time_str0.split(':')[2]
    
    # get date
    d = date_str0.split('-')[0]
    m = date_str0.split('-')[1]
    y = date_str0.split('-')[2]
    
    # transfer string to integer
    mon = _get_month(m)
    
    date_time = '%s:%s:%s %s.%s.%s' %(h, mi, sec, d, mon, y)
    t1_str = datetime.strptime(date_time, "%H:%M:%S %d.%m.%Y")
    
    return (t1_str - t0_str_0).total_seconds()

def _transfer_data(lines0, t0_str_0):
    rows = [0, 1, 2, 8, 9]
    data0 = np.zeros([len(lines0) - 2, 5], dtype=object)
    # data1 = []
    for index, line in enumerate(lines0[2:]):
        for j in range(3):
            data0[index,j] = float(line.split(',')[rows[j]])
        data0[index,3] = line.split(',')[8]
        data0[index,4] = line.split(',')[9].strip()
    
    acc_time0 = float(lines0[2].split(',')[3])
    real_time0 = float(lines0[2].split(',')[4])
    live_time0 = float(lines0[2].split(',')[5])
    date_str = lines0[2].split(',')[6]
    time_str = lines0[2].split(',')[7]
    # sCounts0 = int(lines0[2].split(',')[10])
    fCounts0 = int(lines0[2].split(',')[11])
    pTime0 = float(lines0[2].split(',')[12])
    ftWi0 = float(lines0[2].split(',')[13])
    dt0 = _convert_date_time_in_duration(date_str, time_str, t0_str_0)
    return data0, acc_time0, real_time0, live_time0, dt0, fCounts0, pTime0, ftWi0

def _open_peak_file(path_files0, file0):
    f = open(path_files0 + file0, 'r')
    data0 = f.readlines()
    f.close()
    return data0

def _get_peak_data(path_files0, file0, t0_str_0):
    lines0 = _open_peak_file(path_files0, file0)
    return _transfer_data(lines0, t0_str_0)

def _find_decay_data(Nuc0, path_file0):
    files_dir = _exclude_hidden_files(os.listdir(path_file0))
    for file in files_dir:
        if file.split('_')[0] == Nuc0:
            return file
    print("\n-> Decay data of the nuclide '%s' not found!" %(Nuc0))
    print("   File should be named as '%s_decay_data.csv' in the directory 'files/decay data nuclides/." %(Nuc0))
    sys.exit()

def _calculate_error(value_str, error_str):
    if 'E' in value_str:
        if len(value_str.split('E')[0].split('.')) == 2:
            pot = int(value_str.split('E')[1])
            val = len(value_str.split('E')[0].split('.')[1])
            value = float(value_str.split('E')[0])*10**pot
            error = float(error_str)*10**(pot-val)
        else:
            pot = int(value_str.split('E')[1])
            value = float(value_str.split('E')[0])*10**pot
            error = float(error_str)*10**(pot)
    elif 'e' in value_str:
        if len(value_str.split('e')[0].split('.')) == 2:
            pot = int(value_str.split('e')[1])
            val = len(value_str.split('e')[0].split('.')[1])
            value = float(value_str.split('e')[0])*10**pot
            error = float(error_str)*10**(pot-val)
        else:
            pot = int(value_str.split('e')[1])
            value = float(value_str.split('e')[0])*10**pot
            error = float(error_str)*10**(pot)
    else:
        if len(value_str.split('.')) == 2:
            val = len(value_str.split('.')[1])
            value = float(value_str)
            error = float(error_str)*10**(-val)
        else:
            value = float(value_str)
            error = float(error_str)
    return ufloat(value, error)

def _load_peak_information_file(Nuc0):
    path_file0 = '../files/decay data nuclides/'
    file0 = _find_decay_data(Nuc0, path_file0)
    f = open(path_file0 + file0, 'r')
    data0 = f.readlines()[1:]
    f.close()
    n = len(data0)
    data_peak0 = np.zeros([n,2], dtype=object)
    for index, line in enumerate(data0):
        data_peak0[index,0] = float(line.split(',')[0].split(' ')[0])
        BR0 = line.split('\n')[0].split(',')[1].split('%')
        if not not BR0[1].strip():
            # print(float(BR0[0]), float(BR0[1]))
            data_peak0[index,1] = _calculate_error(BR0[0], BR0[1])
            # data_peak0[index,1] = ufloat_fromstr("%g(%g)" %(float(BR0[0]), float(BR0[1])))*1E-2
        else:
            data_peak0[index,1] = float(BR0[0])*1E-2
    return data_peak0

def _compare_peak_energies_and_return_required_peaks(Nuc0, data0, peak_info0):
    error = False
    data_BR0 = np.zeros([len(data0), 3], dtype=object)
    for index, peak in enumerate(data0):
        photo_peaks = peak[4].split(';')
        found_peak = False
        
        for j in range(len(photo_peaks)):
            photo_peak = float(photo_peaks[j])
            for line in peak_info0:
                photo_peak_raw = line[0]
                if round(photo_peak_raw, 2) == round(photo_peak, 2) or round(photo_peak_raw, 2) == round(photo_peak, 2) + 0.01 or round(photo_peak_raw, 2) == round(photo_peak, 2) - 0.01:
                    if j == 0:
                        data_BR0[index,0] = round(photo_peak, 2)
                        data_BR0[index,1] = line[1]
                        data_BR0[index,2] = peak[3]
                    else:
                        data_BR0[index,1] = data_BR0[index,1] + line[1]
                    found_peak = True
                    break
            if found_peak == False:
                print("-> Peak energy of %g keV not found in '%s_decay-data.csv'!" %(photo_peak, Nuc0))
                error = True
    if error == True:
        print("\n   Check if peak energies are existing!\n")
        sys.exit('Peak energies not existing or deviate more than 0.01 keV!')
    data_BR0[:,1] = data_BR0[:,1]*1E-2
    return data_BR0

def _get_number_of_decays(t0, t1, A0, Lambda):
    return A0*(EXP(-Lambda*t0) - EXP(-Lambda*t1))/Lambda

def _calulate_expected_number_of_decays_per_peak(t0, t1, acc_time0, real_time0, fCounts0, pTime0, ftWi0, peak_info0, A0, Lambda):
    N0 = _get_number_of_decays(t0, t1, A0, Lambda)      # get number of decays during the real time
    t_dead = 1.05*(pTime0 + ftWi0)*1E-6                 # dead time is 1.05 times the peaking time + flat top width, conversion to seconds
    R = fCounts0/acc_time0                              # count rate fast channel
    Rm = R/(np.exp(R*t_dead)*(R*t_dead + 1))            # calculated measured count rate
    f_dead = (R - Rm)/R                                 # dead time fraction
    f_live = 1 - f_dead                                 # live time fraction
    f_real = acc_time0/real_time0                       # scaling factor for accumulated time divided by the real time, should be <=1
    N = N0*f_real*f_live                                # actual number of decays during the system was live
    return peak_info0[:,1]*N                            # returns the total number of decays multiplied by the intensities of each decay

def _pack_ufloat_array(value_row, error_row):
    n = len(value_row)
    array_ufloat = np.zeros(n, dtype=object)
    for i in range(n):
        array_ufloat[i] = ufloat(value_row[i], error_row[i])
    return array_ufloat

def _calculate_attenuation_plastic(E0, d0):
    path0 = '../files/attenuation/'
    filename0 = 'attenuation-coefficient-polyethylene.txt'
    data0 = np.loadtxt(path0 + filename0, skiprows=1)[:,:2]
    data0[:,0] = data0[:,0]*1E+3
    rho_plastic = ufloat(0.935, 0.1)
    f_interpP = interp1d(data0[:,0], data0[:,1], kind='linear')  # or 'cubic', 'quadratic', etc.
    mu_plastic = f_interpP(E0)
    return EXP(-mu_plastic*rho_plastic*d0)

def _add_correction_plastic(data_ufloat0, data0, plastic_attenuation0):
    data_ufloat1 = np.copy(data_ufloat0)
    if plastic_attenuation0[0]:
        d0 = plastic_attenuation0[1]*1E-1       # conversion to [cm]
        for index, E0 in enumerate(data0[:,0]):
            att0 = _calculate_attenuation_plastic(E0, d0)
            data_ufloat1[index] = data_ufloat0[index]/att0
        return data_ufloat1
    else:
        return data_ufloat1

def _return_distance_source_detector(level0, d_err0):
    return ufloat(11.4 - 3 + 10*level0, d_err0)      # [mm]

def _add_correction_geometrical_factor(RS0, level0):
    if RS0 < 1E-3:
        RS0 = 1E-3
    s_dD = 0.5
    dD = _return_distance_source_detector(level0, s_dD)
    fG_coin = geometrical_factor_coin(RS0, 0, dD.nominal_value, dD.std_dev)
    fG_point = geometrical_factor_point(dD.nominal_value, 0)
    fG_coin_ufloat = ufloat(fG_coin[0], fG_coin[1])
    fG_point_ufloat = ufloat(fG_point[0], fG_point[1])
    return fG_point_ufloat/fG_coin_ufloat

def _add_correction_shape_effciency_curve(data0, data_ufloat0, RS0, level0, state_air, state_default):
    if RS0 < 1E-3:
        RS0 = 1E-3
    data_ufloat1 = np.copy(data_ufloat0)
    s_dD = 0.5
    dD = _return_distance_source_detector(level0, s_dD)
    if not state_default:
        if state_air:
            for index, En in enumerate(data0):
                eta_coin = efficiency_normalized_accurate_with_air(En[0], dD.nominal_value, RS0)
                eta_point = efficiency_normalized_accurate_with_air(En[0], dD.nominal_value, 1E-3)
                fac = eta_point/eta_coin
                data_ufloat1[index] = data_ufloat0[index]*fac
        else:
            for index, En in enumerate(data0):
                eta_coin = efficiency_normalized_accurate(En[0], dD.nominal_value, RS0)
                eta_point = efficiency_normalized_accurate(En[0], dD.nominal_value, 1E-3)
                fac = eta_point/eta_coin
                data_ufloat1[index] = data_ufloat0[index]*fac
    return data_ufloat1

def _unpack_ufloat_array(data0):
    n = len(data0)
    data1 = np.zeros([n,2])
    for i in range(n):
        data1[i,0] = data0[i].nominal_value
        data1[i,1] = data0[i].std_dev
    return data1

def _combine_final_data(x, y, yerr, xray_state, list_data, list_eff, list_eff_err, list_Xray):
    list_data.append(x)
    list_eff.append(y)
    list_eff_err.append(yerr)
    return list_data, list_eff, list_eff_err

def _split_array_gamma_xray(data0):
    n = len(data0)
    Xrays, Gammas = [], []
    for i in range(n):
        if 'x-ray' in data0[i,-1]:
            Xrays.append(i)
        else:
            Gammas.append(i)
    nX, nG, m = len(Xrays), len(Gammas), np.size(data0,1)
    
    data_X0, data_G0 = np.zeros([nX, m], dtype=object), np.zeros([nG, m], dtype=object)
    for i in range(nX):
        data_X0[i,:] = data0[Xrays[i],:]
    for i in range(nG):
        data_G0[i,:] = data0[Gammas[i],:]
    return data_X0, data_G0

def _append_Xrays_Gammas_to_list(data_list0, data1):
    data_X0, data_G0 = _split_array_gamma_xray(data1)
    data_list0.append(data_X0)
    data_list0.append(data_G0)
    return data_list0

def _combine_data_to_final_array(data_list0, data0, eff0):
    n = len(data0)
    data1 = np.zeros([n,4], dtype=object)
    for i in range(n):
        data1[i,0] = data0[i,0]
        for j in range(2):
            data1[i,j+1] = eff0[i,j]
        data1[i,3] = data0[i,3]
    return _append_Xrays_Gammas_to_list(data_list0, data1)

def _is_Xray(Nuc0):
    return 'x-ray' in Nuc0

def _get_level_string(val):
    if val == -1:
        return '-1'
    elif val < 10:
        return '0%d' %(val)
    else:
        return '%d' %(val)

def _stack_required_E_Eff_data(data_final0, state_Xrays, state_Gammas):
    Xrays, Gammas = [], []
    for index, arr in enumerate(data_final0):
        if _is_Xray(arr[0,-1]):
            Xrays.append(index)
        else:
            Gammas.append(index)
    if not state_Xrays and not state_Gammas:
        print('-> Neither X-rays nor gammas were selected for fitting!\n')
        sys.exit('Select X-rays or/and gammas for fitting!')
    elif state_Xrays and state_Gammas:
        n = len(data_final0)
        data_stacked0 = data_final0[0]
        for i in range(1,n):
            data_stacked0 = np.vstack((data_stacked0, data_final0[i]))
    elif state_Xrays and not state_Gammas:
        m = len(Xrays)
        if m > 0:
            data_stacked0 = data_final0[Xrays[0]]
            for i in range(1,m):
                data_stacked0 = np.vstack((data_stacked0, data_final0[Xrays[i]]))
    elif not state_Xrays and state_Gammas:
        m = len(Gammas)
        if m > 0:
            data_stacked0 = data_final0[Gammas[0]]
            for i in range(1,m):
                data_stacked0 = np.vstack((data_stacked0, data_final0[Gammas[i]]))
    data_stacked1 = np.zeros([np.size(data_stacked0,0), np.size(data_stacked0,1)-1])
    data_stacked1[:,:] = data_stacked0[:,:-1]
    data_stacked2 = data_stacked0[:,-1]
    return data_stacked1, data_stacked2

def _load_efficiency_curves(level0, state_air, state_default):
    path0 = '../files/SDD Efficiency files/'
    if state_default:
        filename0 = 'SDD_normalized-Efficiency_default.txt'
        data0 = np.loadtxt(path0 + filename0, skiprows=10)
        print('-> Loaded default efficiency file from Amptek (without attenuation of the air).\n')
        E0, A0 = data0[:,0], data0[:,3]
    else:
        if state_air:
            filename0 = 'SDD_normalized-Efficiency_point-source_with-air_all-levels.csv'
            data0 = np.loadtxt(path0 + filename0, skiprows=1, delimiter=',')
            print('-> Loaded modified efficiency file for level %s including the attenuation of the air.\n' %(_get_level_string(level0)))
            E0, A0 = data0[:,0], data0[:,level0+1]
        else:
            filename0 = 'SDD_normalized-Efficiency_point-source_all-levels.csv'
            data0 = np.loadtxt(path0 + filename0, skiprows=1, delimiter=',')
            print('-> Loaded modified efficiency file for level %s without the attenuation of the air.\n' %(_get_level_string(level0)))
            E0, A0 = data0[:,0], data0[:,level0+1]
    return E0, A0, filename0

def _function(E0, a0, b0):
    global eff_curve
    f_interp = interp1d(eff_curve[0], eff_curve[1], kind='linear')  # or 'cubic', 'quadratic', etc.
    return a0 + f_interp(E0)*b0

def _efficiency_raw(E0):
    global eff_curve
    f_interp = interp1d(eff_curve[0], eff_curve[1], kind='linear')  # or 'cubic', 'quadratic', etc.
    return f_interp(E0)

def _efficiency_error(E0, a0, b0, cov_mat0):
    T = np.array([1, _efficiency_raw(E0)])
    return np.sqrt(T @ cov_mat0 @ T)

def _get_efficiency_and_error(E0, popt0, pcov0):
    global eff_curve
    E_low, E_high = eff_curve[0][0], eff_curve[0][-1]
    if E0 >= E_low and E0 <= E_high:
        a0, b0 = popt0[0], popt0[1]
        eff0 = _function(E0, a0, b0)
        eff_err0 = _efficiency_error(E0, a0, b0, pcov0)
    return eff0, eff_err0

def _get_efficiency_and_error_array(En_fit0, popt0, pcov0):
    Eff_fit0, Eff_fit_err0 = np.zeros(len(En_fit0)), np.zeros(len(En_fit0))
    for index, E0 in enumerate(En_fit0):
        Eff_fit0[index], Eff_fit_err0[index] = _get_efficiency_and_error(E0, popt0, pcov0)
    return Eff_fit0, Eff_fit_err0

def _define_plot_label(index0, Nuc_fit0):
    if index0 == 0:
        return True
    else:
        if Nuc_fit0[index0].strip() == Nuc_fit0[index0-1].strip():
            return False
        else:
            return True

def _get_color_Nuc(index0, Nuc_fit0):
    colors = ['tab:blue', 'tab:green', 'tab:purple', 'tab:red', 'tab:orange']
    Nuc_list0 = []
    for Nuc0 in Nuc_fit0:
        Nuc_list0.append(Nuc0.split(' ')[0].strip())
    Nuc_list1 = set(Nuc_list0)
    for index, Nuc1 in enumerate(Nuc_list1):
        if Nuc_fit0[index0].split(' ')[0].strip() == Nuc1:
            return colors[index]

def _get_fmt_Nuc(index0, Nuc_fit0):
    if len(Nuc_fit0[index0].split('x-ray')) == 1:
        return '.'
    else:
        return 'x'

def _get_label_Nuc(index0, Nuc_fit0):
    s = Nuc_fit0[index0].split(' ')[0]
    numbers = ''.join([ch for ch in s if ch.isdigit()])
    letters = ''.join([ch for ch in s if ch.isalpha()])
    if len(Nuc_fit0[index0].split('x-ray')) == 1:
        return r'Data points of $^{%s}$%s, $\gamma$' %(numbers, letters)
    else:
        return r'Data points of $^{%s}$%s, X-ray' %(numbers, letters)

def _return_result(eff0, eff_err0):
    num = 0
    for i in range(0,10):
        if abs(eff0*10**i) >= 1:
            num = i
            break
    return eff0*10**num, eff_err0*10**num, -num

def _return_label_attenuation(filename0, level0):
    lvl_str = _get_level_string(level0)
    if filename0 == 'SDD_normalized-Efficiency_default.txt':
        return '$f(E)$: default efficiency curve'
    elif filename0 == 'SDD_normalized-Efficiency_point-source_all-levels.csv':
        return '$f(E)$: modified efficiency curve\n    for level %s w/o attenuation' %(lvl_str)
    elif filename0 == 'SDD_normalized-Efficiency_point-source_with-air_all-levels.csv':
        return '$f(E)$: modified efficiency curve\n    for level %s w/ attenuation' %(lvl_str)

def _get_decays(Nuc_fit0):
    decay_names = list(set([Nuc0.split(' ')[0].strip() for Nuc0 in Nuc_fit0]))
    for i in range(len(decay_names)):
        s = decay_names[i]
        numbers = ''.join([ch for ch in s if ch.isdigit()])
        letters = ''.join([ch for ch in s if ch.isalpha()])
        decay_names[i] = numbers + letters
    return decay_names

def _define_filename_plot(Nuc_list0, level0, path_sto):
    level_label = _get_level_string(level0)
    decay_names = _get_decays(Nuc_list0)
    files_sto = os.listdir(path_sto)
    files_sto = _exclude_hidden_files(files_sto)
    decay_names_sorted = sorted(list(set(decay_names)))
    filename0 = 'efficiency_'
    for decay in decay_names_sorted:
        filename0 += decay + '-'
    filename0 = filename0[:-1] + '_level-%s_' %(level_label)
    for i in range(1,100):
        state = False
        if i < 10:
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

def _get_decays_string(Nuc_fit0):
    decays0 = _get_decays(Nuc_fit0)
    string = decays0[0]
    for i in range(1, len(decays0)):
        string += ', %s' %(decays0[i])
    return string

def _check_existence_directory(path0, setup0):
    if not os.path.isdir(path0 + setup0):
        setups0 = _exclude_hidden_files(os.listdir(path0))
        print("\n-> Setup '%s' does not exist!" %(setup0))
        print("   The existing setups are:", setups0)
        print()
        sys.exit("Setup not found in the directory '%s'!" %(path0))

def _check_existence_directory_level(path0, level0):
    if not os.path.isdir(path0 + level0):
        folders0 = _exclude_hidden_files(os.listdir(path0))
        print("\n-> Folder '%s' does not exist!" %(level0))
        print("   The existing folders are:", folders0)
        print()
        sys.exit("Folder '%s' not found in the directory '%s'!" %(level0, path0))


def _get_sub_folders(setup0):
    path_raw = '../modified peak files/'
    _check_existence_directory(path_raw, setup0)
    path_1 = path_raw + setup0 + '/'
    folders = _exclude_hidden_files(os.listdir(path_1))
    sub_folders0 = []
    for folder in folders:
        sub_folders0.append(setup + '/' + folder)
    return sub_folders0

def _request_sub_folder(path0, setup0):
    level_folder0 = input("Enter the folder name of the level: ")
    _check_existence_directory_level(path0, level_folder0)
    return setup0 + '/' +  level_folder0

def MAIN_EXECUTION(setup, sub_folder, state_default, state_air, state_Xrays, state_Gammas, store):
    global eff_curve
    # paths
    path_files = '../modified peak files/' + sub_folder + '/'
    path_sto = '../result files calibration/' + sub_folder + '/'
    
    # load and exclude hidden files
    files = _load_files(path_files)
    files_Source = _load_Source_data_for_all_files(path_files, files)
    # levels = _get_level_of_files(files)             # Can may be deleted!
    level = _get_single_level_of_files(files)
    
    # data_final = np.zeros(0, dtype=object)
    data_final = []
    for index, file in enumerate(files):
        # nuclide, initial activity, decay constant, reference date, radius source and plastic attenuation
        Nuc, A0, Lambda, t0_str, RS, plastic_attenuation = _get_decay_parameters_Source(files_Source[index])
        
        # load data and important values of the files and delete marked rows
        data, acc_time, real_time, live_time, dt, fCounts, pTime, ftWi = _get_peak_data(path_files, file, t0_str)
        t_start = dt                # define starting time of the measurement
        t_end = dt + real_time      # define stopping time of the measurement
        
        # load the information of the photo peaks
        peak_info = _load_peak_information_file(Nuc)
        
        # search and return the peak energies, its branching ratios and the decay nuclide
        data_BR = _compare_peak_energies_and_return_required_peaks(Nuc, data, peak_info)
        
        # calculate the expected decays per peak
        N_peaks_expected = _calulate_expected_number_of_decays_per_peak(t_start, t_end, acc_time, real_time, fCounts, pTime, ftWi, data_BR, A0, Lambda)
        
        # pack peak areas in ufloat
        data_ufloat = _pack_ufloat_array(data[:,1], data[:,2])
        
        # corrections for plastic embedded source
        data_ufloat = _add_correction_plastic(data_ufloat, data, plastic_attenuation)
        
        # corrections for the geometrical factor and the shape of the efficiency curve
        data_ufloat *= _add_correction_geometrical_factor(RS, level)
        data_ufloat = _add_correction_shape_effciency_curve(data, data_ufloat, RS, level, state_air, state_default)
        
        # calculate the efficiency
        eff_ufloat = data_ufloat/N_peaks_expected
        
        # unpack ufloats in numpy array
        eff = _unpack_ufloat_array(eff_ufloat)
        
        # append data to list
        data_final = _combine_data_to_final_array(data_final, data, eff)
    
    # load the stored efficiency curve from a file
    eff_curve = _load_efficiency_curves(level, state_air, state_default)
    
    # stack the necessary data into one array
    data_fit, Nuc_fit = _stack_required_E_Eff_data(data_final, state_Xrays, state_Gammas)
    
    # perform a least squares fit
    popt, pcov = curve_fit(_function, data_fit[:,0], data_fit[:,1], sigma=data_fit[:,2], absolute_sigma=True, p0=[0, 0.01])
    perr = np.sqrt(np.diag(pcov))       # error of the parameters
    
    # calculate the root mean squared error of the fit
    rms = np.sqrt(np.mean((data_fit[:,1] - _get_efficiency_and_error_array(data_fit[:,0], popt, pcov)[0])**2))
    print('--> RMS of the fit: %g' %(rms))
    
    
    # define arrays for the energies and the efficiencies computed by the fit
    E_upper = np.max(data_fit[:,0]) + 20
    En_fit = np.linspace(1, E_upper, 400)
    Eff_fit, Eff_fit_unc = _get_efficiency_and_error_array(En_fit, popt, pcov)
    
    # create folder for storage if necessary and define unique file name
    os.makedirs(path_sto, exist_ok=True)
    filename = _define_filename_plot(Nuc_fit, level, path_sto)
    
    # plot data points and fit
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for index, line in enumerate(data_fit):
        plot_label = _define_plot_label(index, Nuc_fit)
        if not plot_label:
            plt.errorbar(line[0], line[1], yerr=line[2], fmt=_get_fmt_Nuc(index, Nuc_fit), color=_get_color_Nuc(index, Nuc_fit), ecolor='black', elinewidth=1, capsize=2, zorder=5)
        else:
            plt.errorbar(line[0], line[1], yerr=line[2], fmt=_get_fmt_Nuc(index, Nuc_fit), color=_get_color_Nuc(index, Nuc_fit), label=_get_label_Nuc(index, Nuc_fit), ecolor='black', elinewidth=1, capsize=2, zorder=5)
    plt.plot(En_fit, Eff_fit, label = r'Fit: $\eta(E)$ = $a$ + $b$ $f(E)$' + '\n'        # plot the fit
             + r'$a$ = (%.3f ± %.3f) $\cdot$ 10$^{%d}$,' %(_return_result(popt[0], perr[0])) 
             + '\n' + r'$b$ = (%.3f ± %.3f) $\cdot$ 10$^{%d}$' %(_return_result(popt[1], perr[1])) 
             + '\n%s' %(_return_label_attenuation(eff_curve[2], level)), color = 'indianred')
    plt.fill_between(En_fit, Eff_fit - Eff_fit_unc, Eff_fit + Eff_fit_unc, label=r'1$\sigma$ fit error (default)',
                       color = 'lightcoral', alpha=0.4)
    ax.set_yscale('log')
    plt.legend()
    plt.grid()
    plt.xlabel('Energy $E$ [keV]')
    plt.ylabel(r'Efficiency $\eta$')
    plt.title("Efficiency on level %s with setup '%s'" %(_get_level_string(level), setup))
    if store == True:
        plt.savefig(path_sto + filename[:-7] + 'log_' + filename[-7:], dpi=400)
    plt.show()
    
    # store the computed parameters to a file
    if store == True:
        f = open(path_sto + filename[:-4] + '.txt', 'w')
        f.write('Used file for the fit: %s\n\n' %(eff_curve[2]))
        f.write('parameters\noffset a\tslope b\n')
        f.write('%g\t%g\n\n' %(popt[0], popt[1]))
        f.write('Covariance matrix\n')
        f.write('%g\t%g\n' %(pcov[0,0], pcov[0,1]))
        f.write('%g\t%g\n\n' %(pcov[1,0], pcov[1,1]))
        f.write('used data points for fitting\n')
        f.write('energy [keV]\tefficiency\tefficiency error\tnuclide\n')
        for index, line in enumerate(data_fit):
            f.write('%g\t%g\t%g\t%s\n' %(line[0], line[1], line[2], Nuc_fit[index]))
        f.close()
        
        print('-> Stored file and plots of the fit with data of %s!' %(_get_decays_string(Nuc_fit)))
        print('   Directory: %s' %(path_sto))
        print('   File name: %s' %(filename[:-4] + '.txt'))
        print('   Plot name: %s' %(filename[:-7] + 'log_' + filename[-7:]))

# %% EXECUTION
path_raw = '../modified peak files/'
if automatic:
    sub_folders = _get_sub_folders(setup)
    for sub_folder in sub_folders:
        print("\nStarting calibration of the folder '%s'!\n" %(sub_folder))
        MAIN_EXECUTION(setup, sub_folder, state_default, state_air, state_Xrays, state_Gammas, store)
else:
    sub_folder = _request_sub_folder(path_raw + setup + '/', setup)
    print("\nStarting single calibration!\n")
    MAIN_EXECUTION(setup, sub_folder, state_default, state_air, state_Xrays, state_Gammas, store)

