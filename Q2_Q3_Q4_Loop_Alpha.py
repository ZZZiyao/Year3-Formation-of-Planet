# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# Constants
AU = 1.5e11  # meters, equal to 1 AU
kb = 1.38e-23  # J/K, Boltzmann constant
mH = 1.6735575e-27  # kg, mass of hydrogen atom
G = 6.6743e-11  # m^3 kg^–1 s^–2, gravitational constant
M = 2e30  # kg, mass of the Sun
rhoin = 1000  # kg/m^3, internal density of dust particles
size_a = 1e-6  # meters, size of dust particles (1 μm)
 
# Fixed parameters at R = 100 AU
R = 100  # AU
R_ref = R * AU
T_ref = 264 * (R**(-1/2))  # Temperature at distance R
cs_ref = (kb * T_ref / (2 * mH))**0.5  # Speed of sound in the disc
omega_ref = (G * M / R_ref**3)**0.5  # Angular velocity
H_ref = cs_ref / omega_ref  # Scale height in meters
rhog_ref = 5e-7 * R**(-3)  # Gas density at distance R in kg/m^3
ts_ref = 1 / omega_ref  # Settling timescale in seconds
uz_ref = cs_ref
D_ref = cs_ref**2 / omega_ref  # Diffusion coefficient
 
def uz(z):
    k = np.sqrt(np.pi / 8) * (omega_ref * rhoin * size_a / (rhog_ref * cs_ref))
    return -k * z * np.exp(z**2 / 2)
 
# Space parameters
max_z = 3  # Maximum z (in scale heights)
Nz = 300  # Number of grid points in space
z = np.linspace(0, max_z, Nz)  # z grid
dz = z[1] - z[0]  # Space step size
 
uz_max = abs(uz(max_z))
 
# Gas density profile
rho_g = np.exp(-z**2 / 2)
 
rho_g_derivative = -z * rho_g
 
rho_g_derivative_max = max(abs(rho_g_derivative))
 
# Initial dust density profile
rho_d_init = np.concatenate([0.01*rho_g.copy()[:-1], [0]])
 
# Alpha values to study
alphas = [1e-5, 1e-4, 1e-3]
 
# Plot setup
plt.figure(figsize=(10, 6))
 
for alpha in alphas:
    print(f"Processing alpha = {alpha}")
   
    # Time parameters
    dt1 = dz / (2*uz_max)  # Time step size
    dt2 = dz**2/(2*alpha)
    dt = min (dt1,dt2)
    print(dt)
    t_max = 3e6 * 365 * 24 * 3600 / ts_ref  # Simulation time in settling timescales
    Nt = int(t_max / dt)  # Number of time steps
    print(Nt)
 
    rho_d = rho_d_init.copy()  # Reset initial dust density
    times=[]
    max_ratios=[]
 
    # Time evolution
    for n in range(Nt):
        #print(n)
        # Calculate velocity at each z
        u = uz(z)
 
        # Advection calculation (physical grid only)
        adv_flux = u * rho_d
        adv_flux_gradient = (adv_flux[2:] - adv_flux[1:-1]) / dz
       
 
        # Diffusion calculation using rho_g_virtual and rho_d_virtual
        diff_contribution = np.zeros_like(rho_d)
 
        # Compute rho_d / rho_g
        rho_ratio = rho_d / rho_g
 
        # First derivative (central difference)
        #rho_ratio_gradient = (rho_ratio[2:] - rho_ratio[:-2]) / (2 * dz)
 
        # Second derivative (central difference)
        diff_contribution[1:-1] = alpha * ((((rho_g[1:-1]+rho_g[2:])/2 * (rho_ratio[2:] - rho_ratio[1:-1])) -
        ((rho_g[1:-1]+rho_g[0:-2])/2 * (rho_ratio[1:-1] - rho_ratio[:-2])) )/ (dz**2))
 
 
        # Update rho_d
        rho_d[1:-1] += dt * (-adv_flux_gradient
                             + diff_contribution[1:-1]
                             )
        rho_d[-1]=0
 
        rho_d[0] += dt* ((-2*(adv_flux[1]-adv_flux[0])/dz) + (alpha * (
        ((rho_g[0]+rho_g[1])/2 * (rho_ratio[1] - rho_ratio[0]) / dz -
        (rho_g[0]+rho_g[1])/2 * (rho_ratio[0] - rho_ratio[1]) / dz) / dz))) # Bottom boundary: symmetry condition
 
 
        # Store maximum ratio at each time step
        times.append(n * dt)
        max_ratios.append(max(rho_d / rho_g))
        
    times = np.array(times)
   
    # Plot for this alpha
    plt.plot((times * ts_ref) / (365 * 24 * 3600), max_ratios, marker='.', linestyle='-', label=f"\u03b1 = {alpha}")
 
# Final plot formatting
plt.xlabel("Time (years)")
plt.ylabel("Max $\\rho_d / \\rho_g$")
plt.title("Maximum Dust-to-Gas Density Ratio vs. Time for Different \u03b1")
plt.legend()
plt.grid()
plt.show()
