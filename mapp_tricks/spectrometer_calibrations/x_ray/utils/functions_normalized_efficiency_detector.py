#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Samuel Juillerat
Date: 11.08.2025

Description:    This script contains the main functions for the computation of
                the normalized attenuation of the X-ray spectormeter at a given 
                energy. Also the attenuation of the air is included.
"""

from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import dblquad

files_path = Path(__file__).resolve().parent.parent / 'calibration-data'

# %% basic functions
def _load_NIST_data_and_return_file():
    path = files_path / 'attenuation'
    filename_Be = 'attenuation-coefficient-Be_NIST.txt'
    filename_Si = 'attenuation-coefficient-Si_NIST.txt'
    
    data_Be = np.loadtxt(path / filename_Be, skiprows=4)
    data_Si = np.loadtxt(path / filename_Si, skiprows=4)
    
    E_Be, mu_Be = data_Be[:,0], data_Be[:,1]
    E_Si, mu_Si = data_Si[:,0], data_Si[:,1]
    return E_Be, mu_Be, E_Si, mu_Si

def _load_original_data():
    path = files_path / 'SDD Efficiency files'
    filename = 'SDD_normalized-Efficiency_default.txt'
    data = np.loadtxt(path / filename, skiprows=10)
    return data[:,0], data[:,3]

def _get_density_air():
    T = 295.15          # temperature 22 °C
    R = 8.314462618     # gas constant [J/(mol K)]
    M = 28.96           # molar mass of the air [g/mol]
    p0 = 101325         # normal pressure [Pa]
    return p0*M/(R*T)*1E-3      # [kg/m^3]

def _barometric_formula(rho0, h):
    p0 = 101325         # normal pressure [Pa]
    g = 9.80597         # gravity [m/s^2]
    return rho0*np.exp(-rho0*g/p0*h)*1E-3      # return density of the air [g/cm^3]

def _load_air_data_and_return_file():
    path = files_path / 'attenuation'
    filename = 'attenuation-coefficient-air.txt'
    
    h_bern = 537         # height [m] above sea level
    rho0_air_20 = _get_density_air()
    rho0_air_20_bern = _barometric_formula(rho0_air_20, h_bern)
    
    data = np.loadtxt(path / filename, skiprows=1)
    E_air = data[:,0]*1E+3
    mu_air = data[:,1]*rho0_air_20_bern
    return E_air, mu_air

def _interpolate(X, Y, X_new, order):
    interp_func = interp1d(X, Y, kind=order)
    return interp_func(X_new)

def _calculate_attenuation(E_fun, mu_fun, E_int, d, order):
    mu_int = _interpolate(E_fun, mu_fun, E_int, order)
    att = np.exp(-mu_int*d)
    return att, 1 - att

# global variables for the follwoing function
E_Be, mu_Be, E_Si, mu_Si = _load_NIST_data_and_return_file()
E_air, mu_air = _load_air_data_and_return_file()

def _efficiency_stretched(fac, E):
    d_Be = 12.7E-6*100          # thickness of the Be window [cm]
    d_Si_0 = 0.15E-6*100        # thickness of the dead Si layer [cm]
    d_Si_1 = 500E-6*100*0.9     # thickness of the active detector volume [cm] (assumption that 50 µm at the end of the detector are inactive [source missing])
    
    d_Be *= fac             # applying the stretching factor
    d_Si_0 *= fac           # applying the stretching factor
    d_Si_1 *= fac           # applying the stretching factor
    
    trans_Be = _calculate_attenuation(E_Be, mu_Be, E, d_Be, 'quadratic')[0]          # transmission Be window
    trans_Si = _calculate_attenuation(E_Si, mu_Si, E, d_Si_0, 'quadratic')[0]        # transmission dead Si layer
    atten_Si = _calculate_attenuation(E_Si, mu_Si, E, d_Si_1, 'quadratic')[1]        # attenuation active volume
    return trans_Be*trans_Si*atten_Si

# %% default calcuation
def efficiency_normalized_default(E):
    """
    Function for calculating the normalized default efficiency at a given energy.
    It computes which fraction of incoming photons are undergoing transmission through the Berylium window and the dead Silicon layer
    and attenuation in the active Silicon detector volume.
    This is computed for perpendicular photons, i.e. photons with an incient angle of 90°.

    Parameters
    ----------
    E : Float
        Energy for the requested normalized default efficiency in keV.

    Returns
    -------
    Float
        Normalized default efficiency at the given energy.

    """
    d_Be = 12.7E-6*100          # thickness of the Be window [cm]
    d_Si_0 = 0.15E-6*100        # thickness of the dead Si layer [cm]
    d_Si_1 = 500E-6*100*0.9     # thickness of the active detector volume [cm] (assumption that 50 µm at the end of the detector are inactive [source missing])
    
    trans_Be = _calculate_attenuation(E_Be, mu_Be, E, d_Be, 'quadratic')[0]          # transmission Be window
    trans_Si = _calculate_attenuation(E_Si, mu_Si, E, d_Si_0, 'quadratic')[0]        # transmission dead Si layer
    atten_Si = _calculate_attenuation(E_Si, mu_Si, E, d_Si_1, 'quadratic')[1]        # attenuation active volume
    return trans_Be*trans_Si*atten_Si


# %% simple calculation
def _calculate_factor(r, d):
    return np.sqrt(r**2/d**2 + 1)

def _integrand_simple(r, d, E):
    fac = _calculate_factor(r, d)
    return r * _efficiency_stretched(fac, E)

def efficiency_normalized_approximation(E, d, RS):
    """
    Function for calculating the normalized efficiency at a given energy, distance and source radius.
    It computes which fraction of incoming photons are undergoing transmission through the Berylium window and the dead Silicon layer
    and attenuation in the active Silicon detector volume.
    This is computed for a finite source radius.
    IMPORTANT: Only an approximation and can deviate strongly for certain cases!

    Parameters
    ----------
    E : Float
        Energy for the requested normalized efficiency in keV.
    d : Float
        Distance between source and silicon of the detector (detector front end + 1.4 mm) in Millimeters.
    RS : Float
        Radius of the source in Millimeters.

    Returns
    -------
    Float
        Normalized efficiency at the given energy.

    """
    pre_fac = 2/RS**2
    integral, error = quad(_integrand_simple, 0, RS, args=(d, E))
    return pre_fac*integral


# %% advanced calculation (without air)
# calculation of major semi-angle
def _theta_a_function(r, d, RC):
    return np.arctan(RC/np.sqrt(r**2 + d**2))

# calculation of minor semi-angle
def _theta_b_function(r, d, RC):
    return np.arccos((r**2 + d**2 - RC**2)/np.sqrt((RC**2 - r**2)**2 + 2*(RC**2 + r**2)*d**2 + d**4))/2

# function theta(phi) for the collimator
def _theta_max(phi, theta_aC, theta_bC):
    return 1/np.sqrt((np.cos(phi)/theta_aC)**2 + (np.sin(phi)/theta_bC)**2)

# Define the integrand
def _integrand_sphere_1(theta, phi, alpha, E):
    fac0 = 1/np.cos(np.sqrt((theta*np.cos(phi))**2 + (theta*np.sin(phi) + alpha)**2))
    return np.sin(theta) * _efficiency_stretched(fac0, E)

def _integrand_sphere_2(theta, phi):
    return np.sin(theta)

def _efficiency_integration_sphere(E, alpha, theta_aC, theta_bC):
    result1, error1 = dblquad(func = lambda theta, phi: _integrand_sphere_1(theta, phi, alpha, E),
                              a = 0,
                              b = 2*np.pi,                  # phi limits
                              gfun = 0,
                              hfun = lambda phi: _theta_max(phi, theta_aC, theta_bC),
                              epsabs=1e-10, epsrel=1e-10)
    
    result2, error2 = dblquad(func = lambda theta, phi: _integrand_sphere_2(theta, phi),
                              a = 0,
                              b = 2*np.pi,
                              gfun = 0,
                              hfun = lambda phi: _theta_max(phi, theta_aC, theta_bC),
                              epsabs=1e-10, epsrel=1e-10)
    return result1/result2

def _integrand_accurate(r, d, E, RC):
    alpha = np.arctan(r/d)
    theta_aC = _theta_a_function(r, d, RC)
    theta_bC = _theta_b_function(r, d, RC)
    return r * _efficiency_integration_sphere(E, alpha, theta_aC, theta_bC)

def efficiency_normalized_accurate(E, d, RS):
    """
    Function for calculating the normalized efficiency at a given energy, distance and source radius.
    It computes which fraction of incoming photons are undergoing transmission through the Berylium window and the dead Silicon layer
    and attenuation in the active Silicon detector volume.
    This is computed for a finite source radius.
    Efficiency is computed precisely by also integrating over the active detector volume.
    IMPORTANT: Takes some time for calculating the effiency due to several numerical integrations!

    Parameters
    ----------
    E : Float
        Energy for the requested normalized efficiency in keV.
    d : Float
        Distance between source and silicon of the detector (detector front end + 1.4 mm) in Millimeters.
    RS : Float
        Radius of the source in Millimeters.

    Returns
    -------
    Float
        Normalized efficiency at the given energy.

    """
    RC = 2.33           # radius collimator [mm]
    pre_fac = 2/RS**2   # pre factor of the integral
    integral, error = quad(_integrand_accurate, 0, RS, args=(d, E, RC), epsabs=1e-10, epsrel=1e-10)      # integration
    return pre_fac*integral

# %% advanced calculation (with air)
def _efficiency_stretched_with_air(fac, E, d_air):
    d_Be = 12.7E-6*100          # thickness of the Be window [cm]
    d_Si_0 = 0.15E-6*100        # thickness of the dead Si layer [cm]
    d_Si_1 = 500E-6*100*0.9     # thickness of the active detector volume [cm] (assumption that 50 µm at the end of the detector are inactive [source missing])
    
    d_Be *= fac             # applying the stretching factor
    d_Si_0 *= fac           # applying the stretching factor
    d_Si_1 *= fac           # applying the stretching factor
    d_air = (0.1*d_air - d_Be)*fac      # conversion to [cm], subtraction of the Be window and applying the stretching factor
    
    trans_air = _calculate_attenuation(E_air, mu_air, E, d_air, 'quadratic')[0]         # transmission air
    trans_Be = _calculate_attenuation(E_Be, mu_Be, E, d_Be, 'quadratic')[0]             # transmission Be window
    trans_Si = _calculate_attenuation(E_Si, mu_Si, E, d_Si_0, 'quadratic')[0]           # transmission dead Si layer
    atten_Si = _calculate_attenuation(E_Si, mu_Si, E, d_Si_1, 'quadratic')[1]           # attenuation active volume
    return trans_air*trans_Be*trans_Si*atten_Si

def _integrand_sphere_1_with_air(theta, phi, alpha, E, d):
    fac0 = 1/np.cos(np.sqrt((theta*np.cos(phi))**2 + (theta*np.sin(phi) + alpha)**2))
    return np.sin(theta) * _efficiency_stretched_with_air(fac0, E, d)

def _efficiency_integration_sphere_with_air(E, alpha, theta_aC, theta_bC, d):
    result1, error1 = dblquad(func = lambda theta, phi: _integrand_sphere_1_with_air(theta, phi, alpha, E, d),
                              a = 0,
                              b = 2*np.pi,                  # phi limits
                              gfun = 0,
                              hfun = lambda phi: _theta_max(phi, theta_aC, theta_bC),
                              epsabs=1e-10, epsrel=1e-10)
    
    result2, error2 = dblquad(func = lambda theta, phi: _integrand_sphere_2(theta, phi),
                              a = 0,
                              b = 2*np.pi,
                              gfun = 0,
                              hfun = lambda phi: _theta_max(phi, theta_aC, theta_bC),
                              epsabs=1e-10, epsrel=1e-10)
    return result1/result2

def _integrand_accurate_with_air(r, d, E, RC):
    alpha = np.arctan(r/d)
    theta_aC = _theta_a_function(r, d, RC)
    theta_bC = _theta_b_function(r, d, RC)
    return r * _efficiency_integration_sphere_with_air(E, alpha, theta_aC, theta_bC, d)

def efficiency_normalized_accurate_with_air(E, d, RS):
    """
    Function for calculating the normalized efficiency at a given energy, distance and source radius.
    It computes which fraction of incoming photons are undergoing transmission through the Berylium window and the dead Silicon layer
    and attenuation in the active Silicon detector volume.
    This is computed for a finite source radius.
    Efficiency is computed precisely by also integrating over the active detector volume.
    IMPORTANT: Takes some time for calculating the effiency due to several numerical integrations!

    Parameters
    ----------
    E : Float
        Energy for the requested normalized efficiency in keV.
    d : Float
        Distance between source and silicon of the detector (detector front end + 1.4 mm) in Millimeters.
    RS : Float
        Radius of the source in Millimeters.

    Returns
    -------
    Float
        Normalized efficiency at the given energy.

    """
    RC = 2.33           # radius collimator [mm]
    pre_fac = 2/RS**2   # pre factor of the integral
    integral, error = quad(_integrand_accurate_with_air, 0, RS, args=(d, E, RC), epsabs=1e-10, epsrel=1e-10)      # integration
    return pre_fac*integral
