"""
Module to generate development data for self-improving data augmentation.
This creates diverse spectrum patterns to test the model's ability to handle
out-of-distribution data.
"""
import sys
sys.path.append('../optogpt')
import os
import numpy as np
import torch
from core.datasets.sim import load_materials, inc_tmm
import math

# Materials for multilayer thin film structures
MATERIALS = ['Al', 'Ag', 'Al2O3', 'AlN', 'Ge', 'HfO2', 'ITO', 'MgF2', 'MgO', 
            'Si', 'Si3N4', 'SiO2', 'Ta2O5', 'TiN', 'TiO2', 'ZnO', 'ZnS', 
            'ZnSe', 'Glass_Substrate']

# Thickness range for materials (in nm)
THICKNESSES = [str(i) for i in range(10, 505, 10)]

# Wavelength range for simulations (in micrometers)
LAMBDA_LOW = 0.4
LAMBDA_HIGH = 1.1
WAVELENGTHS = np.arange(LAMBDA_LOW, LAMBDA_HIGH+1e-3, 0.01)

# Load material refractive index data
NK_DICT = load_materials(all_mats=MATERIALS, wavelengths=WAVELENGTHS, DATABASE='../optogpt/nk')

pi = math.pi

def spectrum(materials, thickness, pol='s', theta=0, wavelengths=WAVELENGTHS, 
             nk_dict=NK_DICT, substrate='Glass_Substrate', substrate_thick=500000):
    """
    Calculate reflection and transmission spectra for multilayer thin film structure.
    
    Args:
        materials: List of material names
        thickness: List of layer thicknesses in nm
        pol: Polarization ('s' or 'p')
        theta: Incidence angle in degrees
        wavelengths: Array of wavelengths in micrometers
        nk_dict: Dictionary of refractive indices
        substrate: Substrate material name
        substrate_thick: Substrate thickness in nm
        
    Returns:
        List of reflection and transmission values at each wavelength
    """
    degree = pi/180
    theta = theta * degree
    wavess = (1e3 * wavelengths).astype('int')
        
    thickness = [np.inf] + thickness + [substrate_thick, np.inf]

    R, T = [], []
    inc_list = ['i'] + ['c']*len(materials) + ['i', 'i']
    for i, lambda_vac in enumerate(wavess):
        n_list = [1] + [nk_dict[mat][i] for mat in materials] + [nk_dict[substrate][i], 1]
        res = inc_tmm(pol, n_list, thickness, inc_list, theta, lambda_vac)
        R.append(res['R'])
        T.append(res['T'])

    return R + T

def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))

def smooth_pulse_function(start, end=0, steepness=50, reverse=False, wavelength=WAVELENGTHS):
    """
    Create a smooth pulse function using sigmoid.
    
    Args:
        start: Starting wavelength
        end: Ending wavelength (not used in current implementation)
        steepness: Steepness of the sigmoid transition
        reverse: Whether to invert the function
        wavelength: Array of wavelengths
        
    Returns:
        Array of spectral values
    """
    # Sigmoid rise
    rising_edge = sigmoid(wavelength, start, steepness)
    
    if reverse:
        return 1 - rising_edge # return the reverse of the smooth pulse function
    return rising_edge

def gaussian_spec(center, std, peak, b=0, if_reverse=False, wavelength=WAVELENGTHS):
    """
    Generate a Gaussian-shaped spectrum.
    
    Args:
        center: Center wavelength of the Gaussian
        std: Standard deviation (width) of the Gaussian
        peak: Peak amplitude
        b: Baseline value
        if_reverse: Whether to invert the spectrum
        wavelength: Array of wavelengths
        
    Returns:
        Array of spectral values
    """
    spec = []
    for i in range(len(wavelength)):
        temp = np.round(peak * np.exp(-0.5*((center-wavelength[i])/std)**2), 3)
        temp = max(temp, b)
        if if_reverse:
            spec.append(1 - temp)
        else:
            spec.append(temp)
    return spec

def double_gaussian_spec(center1, std1, peak1, center2, std2, peak2, 
                         b=0, if_reverse=False, wavelength=WAVELENGTHS):
    """
    Generate a spectrum with two Gaussian peaks.
    
    Args:
        center1, center2: Center wavelengths of the two Gaussians
        std1, std2: Standard deviations of the two Gaussians
        peak1, peak2: Peak amplitudes of the two Gaussians
        b: Baseline value
        if_reverse: Whether to invert the spectrum
        wavelength: Array of wavelengths
        
    Returns:
        Array of spectral values
    """
    # Initialize the spectrum array
    spec = np.zeros(len(wavelength))
    
    # Calculate the first Gaussian curve
    for i in range(len(wavelength)):
        temp1 = peak1 * np.exp(-0.5 * ((center1 - wavelength[i]) / std1) ** 2)
        temp1 = max(temp1, b)  # Ensure the value is not below the baseline
        spec[i] += temp1
    
    # Add the second Gaussian curve
    for i in range(len(wavelength)):
        temp2 = peak2 * np.exp(-0.5 * ((center2 - wavelength[i]) / std2) ** 2)
        temp2 = max(temp2, b)  # Ensure the value is not below the baseline
        spec[i] += temp2

    # If reverse is true, invert the spectrum
    if if_reverse:
        spec = 1 - spec

    # Round the spectrum values
    spec = np.round(spec, 3)
    return spec
    
def generate_dbr_spectra(all_spec):
    """
    Generate spectra for Distributed Bragg Reflector (DBR) structures.
    
    Args:
        all_spec: List to append the generated spectra to
        
    Returns:
        Updated list with DBR spectra added
    """
    # Center wavelength ranges (nm)
    centers = [550, 850, 1050]  # Different center wavelengths
    
    # Material combinations for DBR structures
    dbr_combinations = [
        # Material pairs and thickness ratios for 550nm center
        ('ZnO', 'Al2O3', 60, 80),
        ('ZnO', 'MgF2', 60, 100),
        ('ZnO', 'SiO2', 60, 90),
        ('TiO2', 'Al2O3', 50, 80),
        ('TiO2', 'MgF2', 50, 100),
        ('TiO2', 'SiO2', 50, 90),
        ('ZnS', 'Al2O3', 50, 80),
        ('ZnS', 'MgF2', 50, 100),
        ('ZnS', 'SiO2', 50, 90),
        ('ZnSe', 'Al2O3', 50, 80),
        ('ZnSe', 'MgF2', 50, 100),
        ('ZnSe', 'SiO2', 50, 90)
    ]
    
    # Scale factors for different center wavelengths (550nm -> 1x, 850nm -> ~1.5x, 1050nm -> ~1.9x)
    scale_factors = [1.0, 1.5, 1.9]
    
    for center_idx, scale in enumerate(scale_factors):
        for mat1, mat2, thick1, thick2 in dbr_combinations:
            # Scale thicknesses according to center wavelength
            scaled_thick1 = int(thick1 * scale)
            scaled_thick2 = int(thick2 * scale)
            
            unit_struct = [mat1, mat2]
            unit_thick = [scaled_thick1, scaled_thick2]
            
            # Create DBR structures with 5-10 repeated unit cells
            for rep in range(5, 11):
                dbr_struct = unit_struct * rep
                dbr_thick = unit_thick * rep
                all_spec.append(np.array(spectrum(dbr_struct, dbr_thick)))
    
    return all_spec

def generate_dev_data():
    """
    Generate diverse spectrum patterns for testing.
    
    Returns:
        List of spectra for testing
    """
    all_spec = []
    
    # Generate Gaussian spectra
    centers = [0.5, 0.6, 0.7, 0.8, 0.9]
    stds = [0.02, 0.03, 0.04, 0.05]
    peaks = [1, 0.95, 0.9]
    
    for c in centers:
        for s in stds:
            for p in peaks:
                # Four variations: normal, inverted, normal with zero T, and zero R with normal T
                all_spec.append(np.concatenate((gaussian_spec(c, s, p), gaussian_spec(c, s, p, 0, True))))
                all_spec.append(np.concatenate((gaussian_spec(c, s, p, 0, True), gaussian_spec(c, s, p))))
                all_spec.append(np.concatenate((gaussian_spec(c, s, p), np.zeros(71))))
                all_spec.append(np.concatenate((np.zeros(71), gaussian_spec(c, s, p))))

    # Generate Double Gaussian spectra
    centers1 = [0.4, 0.5]
    centers2 = [0.8, 0.9]
    stds1 = [0.04, 0.06]
    stds2 = [0.04, 0.06]
    peaks1 = [1, 0.9]
    peaks2 = [1, 0.9]
    
    for c1 in centers1:
        for c2 in centers2:
            for s1 in stds1:
                for s2 in stds2:
                    for p1 in peaks1:
                        for p2 in peaks2:
                            # Four variations: normal, inverted, normal with zero T, and zero R with normal T
                            all_spec.append(np.concatenate((
                                double_gaussian_spec(c1, s1, p1, c2, s2, p2),
                                double_gaussian_spec(c1, s1, p1, c2, s2, p2, 0, True)
                            )))
                            all_spec.append(np.concatenate((
                                double_gaussian_spec(c1, s1, p1, c2, s2, p2, 0, True),
                                double_gaussian_spec(c1, s1, p1, c2, s2, p2)
                            )))
                            all_spec.append(np.concatenate((
                                double_gaussian_spec(c1, s1, p1, c2, s2, p2),
                                np.zeros(71)
                            )))
                            all_spec.append(np.concatenate((
                                np.zeros(71),
                                double_gaussian_spec(c1, s1, p1, c2, s2, p2)
                            )))

    # Generate smooth pulse functions
    starts = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    smoothness_values = [35, 40, 45, 50, 55, 60]
    
    for s in starts:
        for sm in smoothness_values:
            # Four variations: normal with zero T, zero R with normal T, normal with inverted T, and inverted R with normal T
            all_spec.append(np.concatenate((
                smooth_pulse_function(s, 0, sm),
                np.zeros(71)
            )))
            all_spec.append(np.concatenate((
                np.zeros(71),
                smooth_pulse_function(s, 0, sm)
            )))
            all_spec.append(np.concatenate((
                smooth_pulse_function(s, 0, sm),
                smooth_pulse_function(s, 0, sm, reverse=True)
            )))
            all_spec.append(np.concatenate((
                smooth_pulse_function(s, 0, sm, reverse=True),
                smooth_pulse_function(s, 0, sm)
            )))

    # Add DBR (Distributed Bragg Reflector) spectra
    all_spec = generate_dbr_spectra(all_spec)

    return all_spec
