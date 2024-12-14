import numpy as np
import matplotlib.pyplot as plt

# Constants
kb = 1.38e-23  # J/K, Boltzmann constant
mH = 1.6735575e-27  # kg, mass of hydrogen atom
G = 6.6743e-11  # m^3 kg^–1 s^–2, gravitational constant
M = 2e30  # kg, mass of the Sun
rhoin = 1000  # kg/m^3, internal density of dust particles
size_a = 0.218  # meters, size of dust particles (1 μm)

# Fixed parameters at R = 100 AU
R = 5.2  # AU
AU = 1.496e11  # m, astronomical unit
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

# Initial dust density profile
rho_d_init = np.concatenate([0.01*rho_g.copy()[:-1], [0]])

# Alpha values to study
alpha = 1e-4  # Different turbulence parameters

# Time parameters
dt1 = dz / (2 * uz_max)  # Time step size estimate from advection
dt2 = dz**2 / (2 * alpha)  # Time step size estimate from diffusion
dt = min(dt1, dt2)

# 1 Myr in seconds
one_myr = 0.5e6 * 365 * 24 * 3600
# Convert 1 Myr in terms of local settling timescale
t_max = one_myr / ts_ref 
Nt = int(t_max / dt)  # Number of time steps

rho_d = rho_d_init.copy()  # Reset initial dust density

# Arrays to store results
times = []
max_ratios = []

prev_max_ratios = [0, 0, 0,0,0]  # Initialize the previous 5 maximum ratios

for n in range(Nt):
    # Calculate velocity at each z
    u = uz(z)

    # Advection calculation (physical grid only)
    adv_flux = u * rho_d
    adv_flux_gradient = (adv_flux[2:] - adv_flux[1:-1]) / dz

    # Diffusion calculation using rho_g and rho_d
    diff_contribution = np.zeros_like(rho_d)

    # Compute rho_d / rho_g
    rho_ratio = rho_d / rho_g

    # Diffusion term
    diff_contribution[1:-1] = alpha * (((rho_g[1:-1] + rho_g[2:]) / 2 * (rho_ratio[2:] - rho_ratio[1:-1]) -
                                        (rho_g[1:-1] + rho_g[0:-2]) / 2 * (rho_ratio[1:-1] - rho_ratio[:-2])) / dz**2)

    # Update rho_d
    rho_d[1:-1] += dt * (-adv_flux_gradient + diff_contribution[1:-1])
    rho_d[-1] = 0
    
    rho_d[0] += dt * ((-2 * (adv_flux[1] - adv_flux[0]) / dz) + alpha * (
        ((rho_g[0] + rho_g[1]) / 2 * (rho_ratio[1] - rho_ratio[0]) / dz -
         (rho_g[0] + rho_g[1]) / 2 * (rho_ratio[0] - rho_ratio[1]) / dz) / dz))

    # Calculate max ratio at this time
    current_ratio = rho_d / rho_g
    max_ratio = np.max(current_ratio)

    # Store time and max ratio
    time_for_R = n * dt * ts_ref / (365 * 24 * 3600)  # Convert to years
    times.append(time_for_R)
    max_ratios.append(max_ratio)

    # Check if the change in max ratio is below threshold for three consecutive iterations
    prev_max_ratios.pop(0)
    prev_max_ratios.append(max_ratio)
    if all(abs(prev_max_ratios[i] - prev_max_ratios[i-1]) < 1e-8 for i in range(1, 5)):
        print(f"Convergence reached at iteration {n}, time = {time_for_R:.3f} years")
        break

# Plot max ratio vs time
plt.plot(times, max_ratios, label='a=0.001 m')

plt.xlabel("Time (years)")
plt.ylabel("Maximum Dust-to-Gas Ratio")
plt.title("Maximum Dust-to-Gas Ratio vs. Time for Different a")
plt.legend()
plt.grid()
plt.show()

