#Import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import TwoSlopeNorm


#Path of the project
path = "/home/melle/OneDrive/Master/Year1/ComplexSystems/Project3/"

#Constants in the code
dK = 0.2
Kmax = 10
K_arr = np.arange(0,Kmax+dK,dK)

#Function to fit
def f(p, a, b):
    return a+b/p

#Lists of the values
p_list = np.arange(0, 1.05, 0.05)
kc_list = []
p_plot_list = [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

#Function for determining the critical K
def determine_kc(arr):
    for i in range(len(arr)):
        if arr[i] > 0.3: break
    return K_arr[i]-dK/2

#Load in the data
data = np.loadtxt(path+'Data/Bonus/data.txt')

#Determine kc and plot
plt.figure()
for i in range(len(data)):
    data_p = data[i]
    if p_list[i] in p_plot_list:
        plt.plot(K_arr, data_p, label=str(p_list[i]))
    kc = determine_kc(data_p)
    kc_list.append(kc)

#Make the plots pretty
plt.grid()
plt.ylabel("r")
plt.xlabel("K")
plt.title("Order parameter versus K")
plt.legend()
plt.savefig(path+"Plots/Bonus/plot_r_k.pdf")


# Perform fit
popt, pcov = curve_fit(f, p_list[1:], kc_list[1:])

# For plotting color map
X, Y = np.meshgrid(p_list[1:], K_arr[1:])
Z = np.array([data[i,1:] for i in range(len(data))])[1:]
norm = TwoSlopeNorm(vmin=0., vcenter=0.3, vmax=1)

# Plot Kc versus p
plt.figure()
plt.scatter(p_list[1:], kc_list[1:], label="data")
plt.plot(p_list[1:], f(p_list[1:], *popt), color='black', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
plt.pcolormesh(X, Y, Z.transpose(), cmap='coolwarm', norm=norm)
plt.colorbar()
plt.ylabel("Kc")
plt.xlabel("p")
plt.xlim(p_list[1], p_list[-1])
plt.title("Estimated relation between p and Kc")
plt.legend()
plt.savefig(path+"Plots/Bonus/plot_kc_p.pdf")
