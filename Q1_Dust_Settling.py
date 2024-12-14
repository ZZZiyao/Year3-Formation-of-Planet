import numpy as np
import matplotlib.pyplot as plt
import os
# Constants
AU = 1.5e11  # meters, equal to 1 AU
kb = 1.38e-23  # J/K, Boltzmann constant
mH = 1.6735575e-27  # kg, mass of hydrogen atom
G = 6.6743e-11  # m^3 kg^–1 s^–2, gravitational constant
M = 2e30  # kg, mass of the Sun
rhoin = 1000  # kg/m^3, internal density of dust particles
size_a = 1e-6  # meters, size of dust particles (1 μm)

# Initialize arrays for results
time = []
distance = np.linspace(1, 100, 50)  # distances from 10 AU to 300 AU

# Iterate over distances
for R in distance:
    print(R)
    # Reference values at R (in AU)
    R_ref = R * AU
    T_ref = 264 * (R**(-1/2))  # Temperature at distance R
    cs_ref = (kb * T_ref / (2 * mH))**0.5  # Speed of sound in the disc
    omega_ref = (G * M / R_ref**3)**0.5  # Angular velocity
    H_ref = cs_ref / omega_ref  # Scale height in meters
    rhog_ref = 5e-7 * R**(-3)  # Gas density at distance R in kg/m^3
    ts_ref = 1 / omega_ref  # Settling timescale in seconds
    uz_ref = (omega_ref**2) * ts_ref * H_ref  # Dust velocity in m/s

    # Function to calculate vertical dust velocity uz(z)
    def uz(z):
        k = np.sqrt(np.pi / 8) * (omega_ref * rhoin * size_a / (rhog_ref * cs_ref))
        return -k * z * np.exp(z**2 / 2)

    # Space parameters
    max_z = 3  # Maximum z (in scale heights)
    Nz = 300  # Number of grid points in space
    z = np.linspace(0, max_z, Nz)  # z grid
    dz = z[1] - z[0]  # Space step size

    # Gas density profile
    rho_g = np.exp(-z**2 / 2)

    # Initial dust density profile
    rho_d = np.concatenate([0.01*rho_g.copy()[:-1], [0]])

    # Time parameters
    uz_max = abs(uz(max_z))  # Maximum downward velocity
    dt = dz / (2*uz_max)  # Time step size
    print(dt)
    t_max = 10e6 * 365 * 24 * 3600 / ts_ref  # Simulation time in settling timescales
    Nt = int(t_max / dt)  # Number of time steps
    print(Nt)

    # Time evolution
    for n in range(Nt):
        u = uz(z)  # Calculate velocity at each z

        # Calculate flux and its gradient
        flux = u * rho_d
        flux_gradient = (flux[2:] - flux[1:-1]) / dz

        # Update dust density
        rho_d[1:-1] -= dt * flux_gradient

        # Boundary conditions
        rho_d[-1] = 0  # Top boundary: set dust density to 0
        rho_d[0] -= dt*2*(flux[1]-flux[0])/dz  # Bottom boundary: symmetry condition

        # Check if local dust-to-gas ratio exceeds 1
        if np.any(rho_d / rho_g > 1):
            time_for_R = n * dt * ts_ref / (365 * 24 * 3600)  # Convert to years
            time.append(time_for_R)
            print(f"Dust-to-gas ratio > 1 reached at R = {R} AU, time = {time_for_R:.3f} years")
            break
# Define the output file path
output_dir = "DustEvolutionResults"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
output_file = os.path.join(output_dir, "dust_evolution_results.txt")

# Save results to a text file
with open(output_file, "w") as f:
    f.write("Distance (AU)\tTime (years)\n")  # Write headers
    for r, t in zip(distance, time):
        f.write(f"{r:.2f}\t{t:.3f}\n")  # Write each pair of distance and time

print(f"Results saved to {output_file}")

# Visualization
plt.figure(figsize=(8, 5))
plt.plot(distance, time, label='Time for local dust-to-gas ratio > 1')
plt.xlabel('Distance (AU)')
plt.ylabel('Time (years)')
plt.title('Dust Density Evolution Across Different Distances')
plt.grid()
plt.legend()
plt.show()
