README

Q1_Dust_Settling: This simulates the pure advection situation, outputs the graph of required time for local density ratio>1 against R, also automatically saves the data file.
Q2_Time_Snapshots: This creates 6 time snapshots for different alpha values at R=100AU. Alpha values can be changed by inputting interested alphas as an array in alphas=[]. R can be changed by inputting a float/integer in AU to R=... Timescale of simulation can be changed by  inputting an integer/float representing number of years in t_max, i.e.t_max = ?* 365 * 24 * 3600 / ts_ref.
Q2_Q3_Q4_Loop_Alpha: This loops different alpha values and plot the evolution of maximum density ratio against a given period of time at R=100AU. Alpha values can be changed by inputting interested alphas as an array in alphas=[]. Timescale and R can also be changed (same way as in Q2_Time_Snapshots)
Q3_Size: Outputs the maximum density ratio against time, where it only simulates within the time where stablization is jest reached. Can change radial distances: R and partical size: size_a to simulate different situations and look for critical value of a for planet formation.
Q4_Loop_Nt_and_alpha: This simulates numerical diffusion. It loops over different alphas where Nz=300 and and different Nz where alpha=1e-4. Nz and alpha values can be changed by inputting an array of interested values into Nz_values=[] and alphas=[]. Timescale and R can also be changed (same way as in Q2_Time_Snapshots)

