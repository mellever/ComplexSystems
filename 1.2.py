#Import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, odeint
from scipy.optimize import fsolve
import networkx as nx
from numba import njit

#Path of the project
path = "/home/melle/OneDrive/Master/Year1/ComplexSystems/Project3/"

#Function which represents the RHS of differential equation (1)
def f(theta_t, K, w):
    theta_i, theta_j = np.meshgrid(theta_t, theta_t) #Smart way to subtract angles without for loop
    return w + K/N * np.sum(np.sin(theta_j - theta_i), axis=0) #Sum over angles and return result

#Forward euler scheme
def scheme(K, theta, w):
    #Loop over all timesteps
    for i in range(len(t)-1):
       theta[:,i+1] = theta[:,i]+dt*f(theta[:,i], K, w) #Forward euler
    return theta #Return the result

#Function for retrieving order parameter
def order_param(theta_t):
    sum_t = np.sum(np.exp(1j *(theta_t - np.mean(theta_t))))
    return abs(sum_t / len(theta_t))

#Function that plots the order parameter
def plot_order_t(theta, K, normal):
    if normal:
        colors = ["tab:orange", "tab:purple"]
        plt.plot(t, [order_param(theta_t) for theta_t in theta.T], label="K = "+str(K), c=colors[int(K)-1])
    else: plt.plot(t, [order_param(theta_t) for theta_t in theta.T])
    plt.title("Order parameter versus time")
    plt.xlabel("t")
    plt.ylabel("r")
    plt.ylim(0,1)
    if normal: plt.legend()

#Function that computes and saves the data. Run only once, since this takes a lot of time
def generate_data(debug, theta):
    if debug: print("Computation started")
    for K in np.arange(0, K_max+dK, dK):
        theta = scheme(K, theta, w)
        np.savetxt(path_data+"theta_K="+str(round(K,4))+".txt", theta)
        if debug: print("Computing... " + str(round(K/K_max*100,4))+"%")
    if debug: print("Computation finished")

#Function that plots the last value of the order parameter r(T)
def plot_order_k(theoretical):
    K_list = np.arange(0, K_max+dK, dK)
    order_list = []
    for K in K_list:
        theta_T = np.loadtxt(path_data+"theta_K="+str(np.round(K,4))+".txt", usecols=-1) #Load the data of the last column
        order_list.append(order_param(theta_T)) #Compute order parameter  
    plt.scatter(K_list, order_list, color="tab:purple", label="Numerical") #Plot the order parameters for the last entry
    if theoretical: plt.plot(K_list_theo, r_list_theo, color="tab:green", label="Theoretical") #Plot solution of the consistency equation
    plt.vlines(K_c, 0, 1, linestyles='--', color='tab:orange', label=r'Theoretical $K_c$')
    plt.title("Order parameter versus coupling constant")
    plt.ylabel("r")
    plt.xlabel("K")
    plt.legend()
    plt.grid()
    plt.savefig(path+"/Plots/order_vs_k.pdf")
    plt.show()

#Define the normal distribution
def g_normal(w):
    return np.exp(-w**2/2)/np.sqrt(2*np.pi)

#Define the consistency equation
def consist_eq(r, K):
    integrand = lambda theta, r: np.cos(theta)**2 * g_normal(K*r*np.sin(theta))
    integral, error = quad(integrand, -np.pi/2, np.pi/2, args=(r))
    return 1 - K*integral

#Function for solving the consistency equation. Can this only be solved for K >= K_c?
def consist_eq_solver():
    dK = 0.01
    K_list = np.arange(K_c, K_max+dK, dK)
    r_list = []
    for K in K_list:
        r_list.append(fsolve(consist_eq, 0.5, args=(K))[0])
    return np.array(r_list), K_list

def generate_data_Q2(debug):
    w = np.random.uniform(-gamma, gamma, N) #Fix frequencies
    for i in range(10):
        if debug: print(i) #Counter to see progress
        theta0 = np.random.uniform(-np.pi, np.pi, N) #Vary initial conditions
        theta = np.empty((N, len(t)))
        theta[:,0] = theta0
        theta = scheme(K, theta, w)
        np.savetxt(path_data+"theta_fixed_w_"+str(i)+".txt", theta)

def generate_data_Q3(debug):
    theta0 = np.random.uniform(-np.pi, np.pi, N) #Fix initial conditions
    for i in range(10):
        if debug: print(i) #Counter to see progress
        w = np.random.uniform(-gamma, gamma, N) #Vary frequencies
        theta = np.empty((N, len(t)))
        theta[:,0] = theta0
        theta = scheme(K, theta, w)
        np.savetxt(path_data+"theta_fixed_initial_"+str(i)+".txt", theta)


#______________________________________QUESTION 1.2.1______________________________________
#Numerical parameters
dt = 1e-2
T = 100 #Time -> should be 100
t = np.arange(0, T+dt, dt) #create array that contains all timepoints

#System parameters 
N = 1000 #population size -> should be 1000
theta0 = np.random.uniform(-np.pi, np.pi, N) #Draw 1000 samples from Unif(-pi,pi)
w = np.random.normal(0, 1, N) #Draw 1000 frequencies from N(0,1)
K_c = np.sqrt(8/np.pi) #Theoretical order parameter for N(0,1)
dK = 0.2 #Steps for coupling constant -> should be 0.2
K_max = 5 #Maximal coupling constant -> should be 5

#Array for saving all the solutions. 
theta = np.empty((N, len(t))) #rows represents theta_i and columns represent timesteps
theta[:,0] = theta0 #initialize solutions

#Path for saving the data
path_data = path+"/Data/Normal/"

#Question 1
#generate_data(True, theta) #Run this once, since this takes quite a lot of time
r_list_theo, K_list_theo = consist_eq_solver()
plot_order_k(True)

#Question 2
for K in [1.0,2.0]:
    theta = np.loadtxt(path_data+"theta_K="+str(np.round(K,2))+".txt")
    plot_order_t(theta, K, True)
plt.grid()
plt.savefig(path+"/Plots/order_vs_t.pdf")
plt.show()

#______________________________________QUESTION 1.2.2______________________________________
#Numerical parameters
dt = 0.05
T = 200 #Time -> should be 200
t = np.arange(0, T+dt, dt) #create array that contains all timepoints

#System parameters 
N = 2000 #population size -> should be 2000
gamma = 1/2 #Defining parameter uniform distribution
theta0 = np.random.uniform(-np.pi, np.pi, N) #Draw 2000 samples from Unif(-pi,pi)
w = np.random.uniform(-gamma, gamma, N) #Draw 2000 frequencies from Unif(-1/2, 1/2)
K_c = 2/np.pi #Theoretical order parameter for Unif(-1/2, 1/2)
dK = 0.03 #Steps for coupling constant -> should be 0.03
K_max = 1.5 #Maximal coupling constant -> should be 1.5

#Array for saving all the solutions. 
theta = np.empty((N, len(t))) #rows represents theta_i and columns represent timesteps
theta[:,0] = theta0 #initialize solutions

#Path for saving the data
path_data = path+"/Data/Normal/"

#Question 1
path_data = path+"/Data/Uniform/Q1/" #Path for saving the data
#generate_data(True, theta) #Run this once since this take quite a lot of time
plot_order_k(False)

#Question 2
K = 1
path_data = path+"/Data/Uniform/Q2/" #Path for saving the data
#generate_data_Q2(True) #Run this once
for i in range(10):
    theta = np.loadtxt(path_data+"theta_fixed_w_"+str(i)+".txt")
    plot_order_t(theta, K, False)
plt.grid()
plt.savefig(path+"/Plots/order_vs_t_fixed_w.pdf")
plt.show()

#Question 3
K = 1
path_data = path+"/Data/Uniform/Q3/" #Path for saving the data
generate_data_Q3(True) #Run this once
for i in range(10):
    theta = np.loadtxt(path_data+"theta_fixed_initial_"+str(i)+".txt")
    plot_order_t(theta, K, False)
plt.grid()
plt.savefig(path+"/Plots/order_vs_t_fixed_initial.pdf")
plt.show()

