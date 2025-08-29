#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Samuel Juillerat
Date: 11.08.2025

Description:    This script contains the main functions for the computation of
                the geometrical factor.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares
from uncertainties import umath as um
from uncertainties import ufloat


epsabs0, epsrel0, limit0= 1e-7, 1e-7, 100

# %% basic functions

# calculation of major semi-angle
def theta_a_function(r, d, RC):
    return np.arctan(RC/np.sqrt(r**2 + d**2))

# calculation of minor semi-angle
def theta_b_function(r, d, RC):
    return np.arccos((r**2 + d**2 - RC**2)/np.sqrt((RC**2 - r**2)**2 + 2*(RC**2 + r**2)*d**2 + d**4))/2

# function theta(phi) for the collimator
def theta_C(phi, theta_aC, theta_bC):
    return 1/np.sqrt((np.cos(phi)/theta_aC)**2 + (np.sin(phi)/theta_bC)**2)

# function theta(phi) for the detector
def theta_D(phi, theta_aD, theta_bD):
    return 1/np.sqrt((np.cos(phi)/theta_aD)**2 + (np.sin(phi)/theta_bD)**2)

# function theta(phi) for a line
def theta_L(phi, theta_0):
    return theta_0/np.sin(phi)

# calculates value which will be needed for solving the equations (intersections of collimator/detector ellipse) numerical
def delta_function(r, d, RC, tC, theta_aC, theta_bC, theta_aD, theta_bD):
    return (np.arctan((RC + r)/(d - tC)) - np.arctan((RC + r)/(d))) - (theta_C(np.pi/2, theta_aC, theta_bC) - theta_D(np.pi/2, theta_aD, theta_bD))

# function of the equations to solve numerically
def equations(vars, theta_aC, theta_bC, theta_aD, theta_bD, delta_val):
    phi_D, phi_C = vars
    eq1 = theta_C(phi_C, theta_aC, theta_bC)*np.sin(phi_C) - delta_val - theta_D(phi_D, theta_aD, theta_bD)*np.sin(phi_D)
    eq2 = theta_C(phi_C, theta_aC, theta_bC)*np.cos(phi_C) - theta_D(phi_D, theta_aD, theta_bD)*np.cos(phi_D)
    return [eq1, eq2]

# calculation of the angle theta for the line integration (theta_0 = theta(pi))
def theta_012(gamma_1, gamma_2, theta_aC, theta_bC, theta_aD, theta_bD):
    return abs(theta_D(gamma_1, theta_aD, theta_bD)*np.sin(gamma_1)), abs(theta_C(gamma_2, theta_aC, theta_bC)*np.sin(gamma_2))

# %% coin simple integration

# integrand for the ellipse of the detector
def integrand_total_ellipse(phi, theta_aD, theta_bD):
    theta = 1/np.sqrt((np.cos(phi)/theta_aD)**2 + (np.sin(phi)/theta_bD)**2)
    return 1 - np.cos(theta)      # return the integrand

# performs first integration and returns the fractional area of the ellipse relative to the sphere
def fraction_ellipse_area_simple(theta_aD, theta_bD):
    integral, _ = quad(integrand_total_ellipse, 0, 2*np.pi, args=(theta_aD, theta_bD), epsabs=1e-10, epsrel=1e-10)
    return integral/(4*np.pi)       # return the fraction of the surface area of the ellipse divided by the whole sphere

# calculates both semi-angles and return the integrand of the final integration
def integrand_coin_simple(r, dD, RC):
    theta_aD = theta_a_function(r, dD, RC)
    theta_bD = theta_b_function(r, dD, RC)
    return r * fraction_ellipse_area_simple(theta_aD, theta_bD)      # return the integrand of the final integral

# final integration
def integration_coin_simple(RS, RC, dD):     # perform final integration
    integral, _ = quad(integrand_coin_simple, 0, RS, args=(dD, RC), epsabs=1e-10, epsrel=1e-10)
    return 2*integral/RS**2        # return result

# %% coin advanced integration

# integrand for the ellipse of the detector
def integrand_ellipse_D(phi, theta_aD, theta_bD):
    theta = 1/np.sqrt((np.cos(phi)/theta_aD)**2 + (np.sin(phi)/theta_bD)**2)
    return 1 - np.cos(theta)      # return the integrand

# integrand for the ellipse of the collimator
def integrand_ellipse_C(phi, theta_aC, theta_bC):
    theta = 1/np.sqrt((np.cos(phi)/theta_aC)**2 + (np.sin(phi)/theta_bC)**2)
    return 1 - np.cos(theta)      # return the integrand

# integrand for the line integration of the detector
def integrand_line_D(phi, theta_01):
    theta = theta_01/np.sin(phi)
    return 1 - np.cos(theta)      # return the integrand

# integrand for the line integration of the collimator
def integrand_line_C(phi, theta_02):
    theta = theta_02/np.sin(phi)
    return 1 - np.cos(theta)      # return the integrand

# performs first partly integration of the detector and returns the fractional area of the ellipse relative to the sphere
def fraction_partly_ellipse_area_D(theta_aD, theta_bD, gamma_1):
    integral, _ = quad(integrand_ellipse_D, -gamma_1, np.pi + gamma_1, args=(theta_aD, theta_bD), epsabs=epsabs0, epsrel=epsrel0, limit=limit0)
    return integral/(4*np.pi)       # return the fraction of the surface area of the ellipse divided by the whole sphere

# performs first partly line integration of the detector and returns the fractional area of the ellipse relative to the sphere
def fraction_line_ellipse_area_D(gamma_1, theta_01):
    integral, _ = quad(integrand_line_D, np.pi + gamma_1, 2*np.pi - gamma_1, args=(theta_01), epsabs=epsabs0, epsrel=epsrel0, limit=limit0)
    return integral/(4*np.pi)       # return the fraction of the surface area of the ellipse divided by the whole sphere

# performs first partly integration of the collimator and returns the fractional area of the ellipse relative to the sphere
def fraction_partly_ellipse_area_C(theta_aC, theta_bC, gamma_2):
    integral, _ = quad(integrand_ellipse_C, np.pi + gamma_2, 2*np.pi - gamma_2, args=(theta_aC, theta_bC), epsabs=epsabs0, epsrel=epsrel0, limit=limit0)
    return integral/(4*np.pi)       # return the fraction of the surface area of the ellipse divided by the whole sphere

# performs first partly line integration of the collimator and returns the fractional area of the ellipse relative to the sphere
def fraction_line_ellipse_area_C(gamma_2, theta_02):
    integral, _ = quad(integrand_line_C, np.pi + gamma_2, 2*np.pi - gamma_2, args=(theta_02), epsabs=epsabs0, epsrel=epsrel0, limit=limit0)
    return integral/(4*np.pi)       # return the fraction of the surface area of the ellipse divided by the whole sphere

# calculates all semi-angles, solves the numerical equations and returns the corresponding integrand
def integrand_coin_advanced(r, dD, RC, tC):
    theta_aC = theta_a_function(r, dD - tC, RC)
    theta_bC = theta_b_function(r, dD - tC, RC)
    theta_aD = theta_a_function(r, dD, RC)
    theta_bD = theta_b_function(r, dD, RC)
    delta = delta_function(r, dD, RC, tC, theta_aC, theta_bC, theta_aD, theta_bD)
    
    fun = lambda vars: equations(vars, theta_aC, theta_bC, theta_aD, theta_bD, delta)   # function for the least_squares function
    lower_bounds = [-np.pi/2, -np.pi/2]     # lower bounds for the results
    upper_bounds = [np.pi/2, np.pi/2]       # upper bounds for the results
    gamma_1, gamma_2 = least_squares(fun, [0.1, 0.1], bounds=(lower_bounds, upper_bounds)).x    # solve equations
    theta_01, theta_02 = theta_012(gamma_1, gamma_2, theta_aC, theta_bC, theta_aD, theta_bD)    # calculate theta_0 values for the detector/collimator
    # seperate two cases of intersections
    if gamma_1 >= 0:
        return r * (fraction_partly_ellipse_area_D(theta_aD, theta_bD, gamma_1) + fraction_line_ellipse_area_D(gamma_1, theta_01)
                    + fraction_partly_ellipse_area_C(theta_aC, theta_bC, gamma_2) - fraction_line_ellipse_area_C(gamma_2, theta_02))
    else:
        return r * (fraction_partly_ellipse_area_D(theta_aD, theta_bD, gamma_1) - fraction_line_ellipse_area_D(abs(gamma_1), theta_01)
                    + fraction_partly_ellipse_area_C(theta_aC, theta_bC, gamma_2) - fraction_line_ellipse_area_C(gamma_2, theta_02))

# performs the final integration, using the simple and the advanced integrand
def integration_coin_advanced(RS, RC, dD, tC):     # perform final integration
    integral1, _ = quad(integrand_coin_simple, 0, RC, args=(dD, RC), epsabs=epsabs0, epsrel=epsrel0, limit=limit0)
    integral2, _ = quad(integrand_coin_advanced, RC, RS, args=(dD, RC, tC), epsabs=epsabs0, epsrel=epsrel0, limit=limit0)
    return (integral1 + integral2)*2/RS**2        # return result

# final function which returns the geometrical factor of a finite measured coin
def integration_coin_advanced_final(RS, RC, dD, tC):
    if RS <= RC:
        return integration_coin_simple(RS, RC, dD)
    elif RS > RC:
        return integration_coin_advanced(RS, RC, dD, tC)

# function for calculating the error of the final function on the source radius
def error_integration_coin_advanced(RS, s_RS, RC, dD, s_dD, tC):
    d_RS = 1E-2
    d_dD = 1E-2
    s_fG_RS2 = ( ( ( integration_coin_advanced_final(RS + d_RS, RC, dD, tC) - integration_coin_advanced_final(RS - d_RS, RC, dD, tC) ) / (2*d_RS) )*s_RS )**2
    s_fG_dD2 = ( ( ( integration_coin_advanced_final(RS, RC, dD + d_dD, tC) - integration_coin_advanced_final(RS, RC, dD - d_dD, tC) ) / (2*d_dD) )*s_dD )**2
    return np.sqrt(s_fG_RS2 + s_fG_dD2)

# final function which returns the geometrical factor of a point source
def geometrical_factor_point(dD, s_dD):
    """
    Function for calculating the geometrical factor of a point source centered under the detector in a distance dD.

    Parameters
    ----------
    dD : float
        Distance between source and silicon of the detector (detector front end + 1.4 mm) in Millimeters.
    s_dD : float
        Error of the distance between the source and the detector in Millimeters.

    Returns
    -------
    float, float
        Geometrical factor and its error, which is the area of the opening angle divided by the area of the whole sphere.

    """
    COS = np.vectorize(um.cos)
    ARCTAN = np.vectorize(um.atan)
    
    dD1 = ufloat(dD, s_dD)
    RC = 2.33       # radius collimator [mm]
    theta = ARCTAN(RC/dD1)        # opening angle from source to collimator
    result = (1 - COS(theta))/2
    return result.nominal_value, result.std_dev

# final function which returns the geometrical factor of a coin source
def geometrical_factor_coin(RS, s_RS, dD, s_dD):
    """
    Function for calculating the geometrical factor of a coin source, with radius RS, centered under the detector in a distance dD.

    Parameters
    ----------
    RS : float
        Coin radius in Millimeters.
    s_RS : float
        Error of the coin radius in Millimeters.
    dD : float
        Distance between source and  silicon of the detector (detector front end + 1.4 mm) in Millimeters.
    s_dD : float
        Error of the distance between source and detector in Millimeters.

    Returns
    -------
    float, float
        Geometrical factor and its error, which is in simple terms the area of the opening angle divided by the area of the whole sphere.

    """
    RC = 2.33       # radius collimator [mm]
    tC = 0.38       # thickness collimator [mm]
    RS_max = dD - RC
    if RS > RS_max:
        print('Radius of the source is too large for an accurate measurement!')
    return integration_coin_advanced_final(RS, RC, dD, tC), error_integration_coin_advanced(RS, s_RS, RC, dD, s_dD, tC)
