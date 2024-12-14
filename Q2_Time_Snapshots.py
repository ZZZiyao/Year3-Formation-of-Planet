# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:49:15 2024

@author: zx2222
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
    k = np.sqrt(np.pi/8)*(omega_ref*rhoin*size_a/(rhog_ref*cs_ref))
    return -k * z * np.exp(z**2/2)
 
# Space parameters
max_z = 3  # Maximum z (in scale heights)
Nz = 300  # Number of grid points in space
z = np.linspace(0, max_z, Nz)  # z grid
dz = z[1] - z[0]  # Space step size
uz_max = abs(uz(max_z))
 
# Gas density profile
rho_g = np.exp(-z**2 / 2)
 
# Initial dust density profile
rho_d_init = np.concatenate([0.01*rho_g.copy()[:-1], [0]])

# Alpha values to study
alphas = [0]
 
# Dictionary to store results: results[alpha][time_index] = (z, rho_d/rho_g)
results = {}
 
for alpha in alphas:
    print(f"Processing alpha = {alpha}")
    # Calculate time step and total simulation time
    dt1 = dz / (2 * uz_max)  # Time step size based on advection CFL
    dt2 = dz**2/(2*alpha)      # Time step size based on diffusion stability
    dt = min(dt1, dt2)
    t_max = 1.5e6 * 365 * 24 * 3600 / ts_ref  # Simulation time in settling timescales
    Nt = int(t_max / dt)  # Number of time steps
    print(Nt)
    # Define output times now that we know dt and Nt
    output_times = [
        0,
        dt * 2000,
        dt * 35000,
        t_max * 0.25,
        t_max * 0.75,
        (Nt-1)*dt
    ]
    # Ensure all times are within the simulation range
    for i, t_ in enumerate(output_times):
        if t_ > (Nt-1)*dt:
            output_times[i] = (Nt-1)*dt
    # Convert output times to indices
    output_indices = [int(t_ / dt) for t_ in output_times]
 
    rho_d = rho_d_init.copy()  # Reset initial dust density
    results[alpha] = {}
 
    # We will store initial condition as well
    results[alpha][0] = rho_d.copy()
 
    # Time evolution
    for n in range(1, Nt):
        # Calculate velocity at each z
        u = uz(z)
 
        # Advection flux and gradient
        adv_flux = u * rho_d
        adv_flux_gradient = (adv_flux[2:] - adv_flux[1:-1]) / dz
 
        # Diffusion calculation
        diff_contribution = np.zeros_like(rho_d)
        rho_ratio = rho_d / rho_g
 
        # Calculate diffusion using a central difference approach
        # Here we use a finite-difference formula for second derivative of (rho_d/rho_g)*rho_g
        diff_contribution[1:-1] = alpha * ( ((rho_g[1:-1]+rho_g[2:])/2 * (rho_ratio[2:] - rho_ratio[1:-1])) 
                                           -((rho_g[1:-1]+rho_g[0:-2])/2 * (rho_ratio[1:-1] - rho_ratio[:-2])) ) / (dz**2)
 
        # Update rho_d
        # Interior points
        rho_d[1:-1] += dt * (-adv_flux_gradient + diff_contribution[1:-1])
 
        # Top boundary (z = max_z)
        rho_d[-1] = 0
 
        rho_d[0] += dt * (
            (-2*(adv_flux[1]-adv_flux[0])/dz) 
            + alpha * (
                ((rho_g[0]+rho_g[1])/2 * (rho_ratio[1] - rho_ratio[0]) / dz - 
                 (rho_g[0]+rho_g[1])/2 * (rho_ratio[0] - rho_ratio[1]) / dz) / dz)
        )
 
        # Store results if n is in output_indices
        if n in output_indices:
            results[alpha][n] = rho_d.copy()

fig, axs = plt.subplots(3, 2, figsize=(10, 12))
axs = axs.flatten()

plot_times = output_times
 
for i, time_ in enumerate(plot_times):
    ax = axs[i]
    idx = output_indices[i]
    # Plot each alpha line
    for alpha in alphas:
        if idx in results[alpha]:
            rho_d_plot = results[alpha][idx]
        else:
            # If for some reason index not stored (should not happen), skip
            continue
    
        # Dust-to-gas ratio
        ratio = rho_d_plot / rho_g
        # Label only once for each alpha in each subplot
        ax.plot(z*H_ref/R_ref, ratio, label=f"α={alpha}")
    if i==0:
        ax.legend()
 
    # Formatting
    # Convert index back to actual time for a nice title
    current_time = idx * dt
    ax.set_title(f"t = {current_time*ts_ref/(365*24*3600):.2e} years")
    ax.set_xlabel(f"Multiples of 100 AU")
    ax.set_ylabel(r"$\rho_d / \rho_g$")
    ax.grid(True)
    
 
handles, labels = axs[-1].get_legend_handles_labels()
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()