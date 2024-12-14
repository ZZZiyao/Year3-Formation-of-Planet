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
 
# Dust velocity function
def uz(z):
    k = np.sqrt(np.pi / 8) * (omega_ref * rhoin * size_a / (rhog_ref * cs_ref))
    return -k * z * np.exp(z**2 / 2)
 
# Gas density profile
rho_g_full = lambda z: np.exp(-z**2 / 2)
 
# Alpha values to study
alphas = [0, 1e-7, 1e-8, 1e-9]
Nz_values = [10, 50, 100, 200, 300]
 
# General plot for all alpha and dz combinations
plt.figure(figsize=(12, 8))
 
# Loop for alpha values with Nz = 300
for alpha in alphas:
    print(f"Processing alpha = {alpha} with Nz = 300")
    max_z = 3  # Maximum z (in scale heights)
    Nz = 300  # Fixed grid points for alpha values
    z = np.linspace(0, max_z, Nz)
    dz = z[1] - z[0]
    rho_g = rho_g_full(z)
    rho_d_init = np.concatenate([0.01*rho_g.copy()[:-1], [0]])
    rho_d = rho_d_init.copy()
 
    # Time step calculation
    dt1 = dz / (2 * abs(uz(max_z)))
    if alpha > 0:
        dt2 = dz**2 / (2 * alpha)
        dt = min(dt1, dt2)
    else:
        dt = dt1  # Only advection for alpha = 0
 
    # Time parameters
    t_max = 3e6 * 365 * 24 * 3600 / ts_ref
    Nt = int(t_max / dt)
 
    times = []
    max_ratios = []
 
    # Time evolution
    for n in range(Nt):
        u = uz(z)
        adv_flux = u * rho_d
        adv_flux_gradient = (adv_flux[2:] - adv_flux[1:-1]) / dz
 
        rho_ratio = rho_d / rho_g
        diff_contribution = np.zeros_like(rho_d)
 
        if alpha > 0:
            diff_contribution[1:-1] = alpha * ((((rho_g[1:-1] + rho_g[2:]) / 2 * (rho_ratio[2:] - rho_ratio[1:-1])) -
                                               ((rho_g[1:-1] + rho_g[0:-2]) / 2 * (rho_ratio[1:-1] - rho_ratio[:-2]))) / (dz**2))
 
        # Update rho_d
        rho_d[1:-1] += dt * (-adv_flux_gradient + diff_contribution[1:-1])
        rho_d[-1] = 0
        rho_d[0] += dt * ((-2 * (adv_flux[1] - adv_flux[0]) / dz) +
                          (alpha * (((rho_g[0] + rho_g[1]) / 2 * (rho_ratio[1] - rho_ratio[0]) / dz -
                                     (rho_g[0] + rho_g[1]) / 2 * (rho_ratio[0] - rho_ratio[1]) / dz) / dz)))
 
        times.append(n * dt)
        max_ratios.append(max(rho_d / rho_g))
 
    times = np.array(times)
    max_ratios = np.array(max_ratios)
 
    # Plot results
    plt.plot((times * ts_ref) / (365 * 24 * 3600), max_ratios, marker='.', linestyle='-',
             label=f"alpha = {alpha}, Nz = 300")
 
# Special case for alpha = 1e-4 with varying dz
print("Processing alpha = 1e-4 with varying dz values")
alpha = 1e-4
for Nz_value in Nz_values:
    max_z = 3
    Nz = Nz_value
    z = np.linspace(0, max_z, Nz)
    dz = z[1] - z[0]
    rho_g = rho_g_full(z)
    rho_d_init = 0.01 * rho_g.copy()
    rho_d = rho_d_init.copy()
    dt1 = dz / (2 * abs(uz(max_z)))
    dt2 = dz**2 / (2 * alpha)
    dt = min(dt1, dt2)
    t_max = 3e6 * 365 * 24 * 3600 / ts_ref
    Nt = int(t_max / dt)
    times = []
    max_ratios = []
    for n in range(Nt):
        u = uz(z)
        adv_flux = u * rho_d
        adv_flux_gradient = (adv_flux[2:] - adv_flux[1:-1]) / dz
        rho_ratio = rho_d / rho_g
        diff_contribution = np.zeros_like(rho_d)
        diff_contribution[1:-1] = alpha * ((((rho_g[1:-1] + rho_g[2:]) / 2 * (rho_ratio[2:] - rho_ratio[1:-1])) -
                                           ((rho_g[1:-1] + rho_g[0:-2]) / 2 * (rho_ratio[1:-1] - rho_ratio[:-2]))) / (dz**2))
        rho_d[1:-1] += dt * (-adv_flux_gradient + diff_contribution[1:-1])
        rho_d[-1] = 0
        rho_d[0] += dt * ((-2 * (adv_flux[1] - adv_flux[0]) / dz) +
                          (alpha * (((rho_g[0] + rho_g[1]) / 2 * (rho_ratio[1] - rho_ratio[0]) / dz -
                                    (rho_g[0] + rho_g[1]) / 2 * (rho_ratio[0] - rho_ratio[1]) / dz) / dz)))
        times.append(n * dt)
        max_ratios.append(max(rho_d / rho_g))
    times = np.array(times)
    max_ratios = np.array(max_ratios)
    plt.plot((times * ts_ref) / (365 * 24 * 3600), max_ratios, marker='.', linestyle='--',
             label=f"alpha = 1e-4, Nz = {Nz_value}")
 
# Plot setup
plt.xlabel("Time (years)")
plt.ylabel("Max $\\rho_d / \\rho_g$")
plt.title("Maximum Dust-to-Gas Density Ratio for Different Alpha and dz Values")
plt.legend(loc='upper left')
plt.grid()
plt.show()