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

import os
import sys
import math as m
import numpy as np
from uncertainties import ufloat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from .utils.functions_geometrical_factor import geometrical_factor_coin, geometrical_factor_point
from .utils.functions_normalized_efficiency_detector import efficiency_normalized_accurate, efficiency_normalized_accurate_with_air

class XraySpectrometerEfficiency:
    """
    Class to calculate the efficiency of the X-ray spectrometer for given energies.
    """
    
    def __init__(self, level=0, radius=0, setup='Gain3-398_PeakingTime1us', store_plot=False):
        """
        Initialize the efficiency calculator.
        
        Parameters:
        -----------
        level : int
            Level of the spectrometer {0,1,2,...,14}
        radius : float
            Radius of the source [mm]
        setup : str
            Name of the setup (e.g. 'Gain3-398_PeakingTime1us')
        store_plot : bool
            Whether to store the final plot {True/False}
        """
        self.level = level
        self.radius = radius
        self.setup = setup
        self.store_plot = store_plot
        
        # Initialize data storage
        self.eff_curve = None
        self.data_peaks = None
        self.a = None
        self.b = None
        self.cov_mat = None
        self.filename = None
        self.nuc_list = None
        self.plot_label_list = []

        self.this_file_directory = os.path.dirname(os.path.abspath(__file__))
        
        # Load calibration data
        self._load_calibration_data()

    
    def _load_calibration_data(self):
        """Load calibration data during initialization."""

        path0 = self.this_file_directory + '/result files calibration/'
        lvl = self._convert_level(self.level)  # convert to string
        self._check_if_setup_exists(path0, self.setup)
        self._check_if_level_exists(path0 + self.setup + '/', lvl)
        path = path0 + self.setup + '/' + lvl + '/'

        # Define name of file to load efficiency curve
        file = self._get_text_filename(path, lvl)

        # Load the parameters and other important data
        self.a, self.b, self.cov_mat, self.data_peaks, self.filename, self.nuc_list = self._load_data_file(path, file)

        # Load the normalized efficiency curve
        self.eff_curve = self._load_data_efficiency_file(self.filename, self.level)
    
    def evaluate_efficiency_at_energy(self, E) -> ufloat:
        """
        Calculate and return the efficiency at a given energy.
        
        Parameters:
        -----------
        E : float
            Energy of interest [keV]
            
        Returns:
        --------
        tuple
            (efficiency, efficiency_error) at energy E
        """
        # Compute the efficiency and its error
        eff, eff_err = self._get_efficiency_and_error(E, self.a, self.b, self.cov_mat, state_plot=False)
        eff_ufloat = ufloat(eff, eff_err)
        
        if self.radius != 0:
            eff_ufloat *= self._add_correction_geometrical_factor(self.radius, self.level)
            eff_ufloat *= self._add_correction_shape_effciency_curve(E, self.level, self.radius, self.filename)

        return eff_ufloat

    def plot_efficiency_curve(self, E):
        """
        Plot the efficiency curve with data points and the fit.
        
        Parameters:
        -----------
        E : float
            Energy of interest [keV] to highlight on the plot
        """
        # Get efficiency at energy E
        eff, eff_err = self.evaluate_efficiency_at_energy(E)
        
        # Print the efficiency result
        lvl = self._convert_level(self.level)
        self._print_efficiency(E, eff, eff_err, lvl)
        
        # Compute arrays of energy, efficiency and its error for plotting
        E_arr, eff_arr, eff_err_arr = self._compute_efficiency_and_error_plot(self.a, self.b, self.cov_mat, E)

        # Plot the data and the fit
        self.plot_label_list = []  # reset global list
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        plt.errorbar(E, eff, eff_err, fmt='.', color='black', ecolor='black', elinewidth=1, capsize=2, zorder=6,
                     label='$E_I$ = %.2f keV\n' %(E) + r'$\eta(E_I)$ = (%.3f ± %.3f) $\cdot$ 10$^{%d}$' %(self._return_result(eff, eff_err)))
        
        for index, line in enumerate(self.data_peaks):
            plot_label = self._define_plot_label(index, self.data_peaks[:,3])
            if not plot_label:
                plt.errorbar(line[0], line[1], yerr=line[2], fmt=self._get_fmt_Nuc(index, self.data_peaks[:,3]),
                         ecolor='black', elinewidth=1, capsize=2, zorder=5, color=self._get_color_Nuc(index, self.data_peaks[:,3]))
            else:
                plt.errorbar(line[0], line[1], yerr=line[2], fmt=self._get_fmt_Nuc(index, self.data_peaks[:,3]), 
                         label = self._get_label_Nuc(index, self.data_peaks[:,3]),
                         ecolor='black', elinewidth=1, capsize=2, zorder=5, color=self._get_color_Nuc(index, self.data_peaks[:,3]))
        
        plt.plot(E_arr, eff_arr, label = r'Fit: $\eta(E)$ = $a$ + $b$ $f(E)$' + '\n'
                 + r'$a$ = (%.3f ± %.3f) $\cdot$ 10$^{%d}$,' %(self._return_result(self.a.nominal_value, self.a.std_dev)) 
                 + '\n' + r'$b$ = (%.3f ± %.3f) $\cdot$ 10$^{%d}$' %(self._return_result(self.b.nominal_value, self.b.std_dev)) 
                 + '\n%s' %(self._return_label_attenuation(self.filename, self.level)), color = 'indianred')
        
        plt.fill_between(E_arr, eff_arr - eff_err_arr, eff_arr + eff_err_arr, 
                         label=r'1$\sigma$ fit error (68.3% confidence level)', color = 'lightcoral', alpha=0.4)
        
        ax.set_yscale('log')
        plt.legend()
        plt.grid()
        plt.xlabel('Energy $E$ [keV]')
        plt.ylabel(r'Efficiency $\eta$')
        plt.title('Efficiency curve on %s' %(lvl))
        
        if self.store_plot == True:
            path_sto = self.this_file_directory + '/efficiency plots/'
            filename_plot = self._define_filename_plot(E, self.level, self.setup, self.radius, path_sto)
            plt.savefig(path_sto + filename_plot, dpi=400)
            print("-> Plot with filename '%s' saved!" %(filename_plot))
        
        plt.show()

    def _exclude_hidden_files(self, files0):
        files1 = []
        for file in files0:
            if file[0] != '.':
                files1.append(file)
        return files1

    def _check_if_setup_exists(self, path0, setup0):
        setups0 = self._exclude_hidden_files(os.listdir(path0))
        for s in setups0:
            if s == setup0:
                return 0
        print("-> Setup '%s' not found in the directory '%s'!\n" %(setup0, path0))
        print('   Available setups:', setups0)
        print()
        sys.exit('Setup name not found!')

    def _check_if_level_exists(self, path0, level0):
        levels0 = self._exclude_hidden_files(os.listdir(path0))
        for s in levels0:
            if s == level0:
                return 0
        print("-> Folder '%s' not found in the directory '%s'!\n" %(level0, path0))
        print('   Available levels:', sorted(levels0))
        print()
        sys.exit('Level name not found!')

    def _list_text_files(self, files0):
        files1 = []
        for file in files0:
            if file[-4:] == '.txt':
                files1.append(file)
        return sorted(files1)

    def _get_text_filename(self, path0, level0):       # returns the last text-file of the directory
        files0 = os.listdir(path0)
        files0 = self._exclude_hidden_files(files0)
        files0 = self._list_text_files(files0)
        return files0[-1]

    def _get_data_peaks_nuclides(self, peaks0):
        nuclides0 = []
        data_peaks0 = np.zeros([len(peaks0), 4], dtype=object)
        for index, line in enumerate(peaks0):
            line_ele = line.split('\t')
            for j in range(3):
                data_peaks0[index,j] = float(line_ele[j])
            data_peaks0[index,3] = line_ele[3]
            nuclides0.append(line_ele[3].split('\n')[0])
        return data_peaks0, nuclides0

    def _get_nuclide_number(self, nuclides0):
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

    def _get_cov_mat(self, covariance0):
        return np.array([[ float(covariance0[0].split('\t')[0]), float(covariance0[0].split('\t')[1])],
                             [ float(covariance0[1].split('\t')[0]), float(covariance0[1].split('\t')[1])]])

    def _load_data_file(self, path0, filename0):
        f = open(path0 + filename0, 'r')
        data = f.readlines()
        f.close()
        
        filename1 = data[0][23:].split('\n')[0]
        
        cov_mat0 = self._get_cov_mat(data[7:9])
        
        parameters0 = data[4].split('\t')
        a0, b0 = ufloat(float(parameters0[0]), np.sqrt(cov_mat0[0,0])), ufloat(float(parameters0[1]), np.sqrt(cov_mat0[1,1]))
        
        data_peaks0, nuclides0 = self._get_data_peaks_nuclides(data[12:])
        # data_peaks0[:,3], nuc_list0 = get_nuclide_number(nuclides0)
        
        return a0, b0, cov_mat0, data_peaks0, filename1, []

    def _load_data_efficiency_file(self, filename1, level0):
        path = self.this_file_directory + '/files/SDD Efficiency files/'
        if filename1 == 'SDD_normalized-Efficiency_default.txt':
            data = np.loadtxt(path + filename1, skiprows=10)
            return np.transpose(np.vstack((data[:,0], data[:,3])))
        else:
            data = np.loadtxt(path + filename1, skiprows=1, delimiter=',')
            return np.transpose(np.vstack((data[:,0], data[:,level0+1])))

    def _interpolate(self, X, Y, X_new):
        interp_func = interp1d(X, Y, kind='linear')  # or 'cubic', 'quadratic'
        return interp_func(X_new)

    def _efficiency(self, E0, a0, b0):
        return a0.nominal_value + self._interpolate(self.eff_curve[:,0], self.eff_curve[:,1], E0)*b0.nominal_value

    def _efficiency_raw(self, E0):
        return self._interpolate(self.eff_curve[:,0], self.eff_curve[:,1], E0)

    def _efficiency_error(self, E0, a0, b0, cov_mat0):
        T = np.array([1, self._efficiency_raw(E0)])
        return np.sqrt(T @ cov_mat0 @ T)

    def _get_efficiency_and_error(self, E0, a0, b0, cov_mat0, state_plot):
        if state_plot == False:
            if E0 < self.eff_curve[0,0] or E0 > self.eff_curve[-1,0]:
                print('-> Energy of %.2f keV is out of range!' %(E0))
                sys.exit()
        eff0 = self._efficiency(E0, a0, b0)
        eff_err0 = self._efficiency_error(E0, a0, b0, cov_mat0)
        # print(eff, eff_err)
        return eff0, eff_err0

    def _compute_efficiency_and_error_plot(self, a0, b0, cov_mat0, E):
        E_upper = np.max([np.max(self.data_peaks[:,0]), E]) + 20
        E_arr0 = np.linspace(1, E_upper, 200)
        eff_arr0 = np.zeros(200,)
        eff_err_arr0 = np.zeros(200,)
        
        for i in range(200):
            eff_arr0[i], eff_err_arr0[i] = self._get_efficiency_and_error(E_arr0[i], a0, b0, cov_mat0, True)
        return E_arr0, eff_arr0, eff_err_arr0

    def _get_nuclide_color(self, nuc0):
        colors_list = ['tab:blue', 'tab:green', 'tab:purple', 'tab:olive', 'tab:cyan']
        for index, color in enumerate(colors_list):
            if nuc0 == index:
                return color
        return 'tab:gray'

    def _get_nuclide_label(self, nuc0, nuc_list0):
        for num_label in self.plot_label_list:
            if nuc0 == num_label:
                return None
        self.plot_label_list.append(nuc0)
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

    def _return_result(self, eff0, eff_err0):
        num = 0
        for i in range(0,10):
            if abs(eff0*10**i) >= 1:
                num = i
                break
        return eff0*10**num, eff_err0*10**num, -num

    def _return_result_parameters(self, a0, b0):
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

    def _return_power_of_value(self, integer):
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

    def _return_2numbers(self, integer):
        if integer < 10 and integer >= 0:
            return '0%d' %(integer)
        else:
            return '%d' %(integer)

    def _return_3numbers(self, integer):
        if integer < 10 and integer >= 0:
            return '00%d' %(integer)
        elif integer >= 10 and integer < 100:
            return '0%d' %(integer)
        else:
            return '%d' %(integer)

    def _print_efficiency(self, E0, eff0, eff_err0, level0):
        eff1, eff_err1, num = self._return_result(eff0, eff_err0)
        result = '(%.3f ± %.3f)%s' %(eff1, eff_err1, self._return_power_of_value(num))
        if E0 < 25:
            print('CAUTION! Energy of %.2f keV is below lowest measured data point!' %(E0))
        print('-> The efficiency at E = %.2f keV on %s is %s.\n' %(E0, level0, result))

    def _convert_pTime_in_number(self, string):
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

    def _convert_pTime_list_in_numbers(self, pTime_list0):
        pTime_list1 = []
        for pTime0 in pTime_list0:
            pTime_list1.append(self._convert_pTime_in_number(pTime0))
        return sorted(pTime_list1)

    def _convert_peakingTime(self, value_pTime):
        if value_pTime == 1:
            return '1 us'
        elif value_pTime == 25.6:
            return '25-6 us'
        else:
            path = self.this_file_directory + '/result files calibration/'
            pTime_list = os.listdir(path)
            pTime_list = self._exclude_hidden_files(pTime_list)
            pTime_list = self._convert_pTime_list_in_numbers(pTime_list)
            print('-> Peaking time of %g µs does not exist!' %(value_pTime))
            print('   Available peaking times:', pTime_list, 'µs\n')
            sys.exit()

    def _convert_level(self, level0):
        if level0 >= 0 and level0 <= 14 and int(level0) == level0:
            if level0 < 10:
                return 'level 0%d' %(int(level0))
            else:
                return 'level %d' %(int(level0))
        else:
            print('-> Level %g does not exist!' %(level0))
            print('   Existing levels: {0, 1, 2, ..., 14}\n')
            sys.exit()

    def _return_float_with_minus(self, value):
        string0 = '%g' %(value)
        string1 = string0.split('.')
        if len(string1) == 2:
            return  '%s-%s' %(string1[0], string1[1])
        else:
            return string1[0]

    def _define_filename_plot(self, E0, level0, setup0, radius0, path_sto):
        files_sto = os.listdir(path_sto)
        files_sto = self._exclude_hidden_files(files_sto)
        filename0 = 'efficiency_Level%s' %(self._return_2numbers(level0)) + '_%s' %(setup0) + '_Radius%smm' %(self._return_float_with_minus(radius0)) + '_Energy%s' %(self._return_float_with_minus(float(E0))) + '_'
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

    def _get_label_Nuc(self, index0, Nuc_fit0):
        s = Nuc_fit0[index0].split(' ')[0]
        numbers = ''.join([ch for ch in s if ch.isdigit()])
        letters = ''.join([ch for ch in s if ch.isalpha()])
        if len(Nuc_fit0[index0].split('x-ray')) == 1:
            return r'Data points of $^{%s}$%s, $\gamma$' %(numbers, letters)
        else:
            return r'Data points of $^{%s}$%s, X-ray' %(numbers, letters)

    def _define_plot_label(self, index0, Nuc_fit0):
        if index0 == 0:
            return True
        else:
            if Nuc_fit0[index0].strip() == Nuc_fit0[index0-1].strip():
                return False
            else:
                return True

    def _get_color_Nuc(self, index0, Nuc_fit0):
        colors = ['tab:blue', 'tab:green', 'tab:purple', 'tab:red', 'tab:orange']
        Nuc_list0 = []
        for Nuc0 in Nuc_fit0:
            Nuc_list0.append(Nuc0.split(' ')[0].strip())
        Nuc_list1 = set(Nuc_list0)
        for index, Nuc1 in enumerate(Nuc_list1):
            if Nuc_fit0[index0].split(' ')[0].strip() == Nuc1:
                return colors[index]

    def _get_fmt_Nuc(self, index0, Nuc_fit0):
        if len(Nuc_fit0[index0].split('x-ray')) == 1:
            return '.'
        else:
            return 'x'

    def _pack_ufloat_array(self, value_row, error_row):
        n = len(value_row)
        array_ufloat = np.zeros(n, dtype=object)
        for i in range(n):
            array_ufloat[i] = ufloat(value_row[i], error_row[i])
        return array_ufloat

    def _return_distance_source_detector(self, level0, d_err0):
        return ufloat(11.4 - 3 + 10*level0, d_err0)      # [mm]

    def _add_correction_geometrical_factor(self, radius0, level0):
        if radius0 < 1E-3:
            radius0 = 1E-3
        s_dD = 0.5
        dD = self._return_distance_source_detector(level0, s_dD)
        fG_coin = geometrical_factor_coin(radius0, 0, dD.nominal_value, dD.std_dev)     # <<-- Error of distance necessary?!?
        fG_point = geometrical_factor_point(dD.nominal_value, 0)
        fG_coin_ufloat = ufloat(fG_coin[0], fG_coin[1])
        fG_point_ufloat = ufloat(fG_point[0], fG_point[1])
        return fG_coin_ufloat/fG_point_ufloat

    def _add_correction_shape_effciency_curve(self, E0, level0, radius0, filename0):
        if radius0 < 1E-3:
            radius0 = 1E-3
        s_d0 = 0.5
        d0 = self._return_distance_source_detector(level0, s_d0)
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

    def _get_level_string(self, val):
        if val == -1:
            return '-1'
        elif val < 10:
            return '0%d' %(val)
        else:
            return '%d' %(val)

    def _return_label_attenuation(self, filename0, level0):
        lvl_str = self._get_level_string(level0)
        if filename0 == 'SDD_normalized-Efficiency_default.txt':
            return '$f(E)$: default efficiency curve'
        elif filename0 == 'SDD_normalized-Efficiency_point-source_all-levels.csv':
            return '$f(E)$: modified efficiency curve\n    for level %s w/o attenuation' %(lvl_str)
        elif filename0 == 'SDD_normalized-Efficiency_point-source_with-air_all-levels.csv':
            return '$f(E)$: modified efficiency curve\n    for level %s w/ attenuation' %(lvl_str)

