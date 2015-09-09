# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:38:51 2015

@author: ZhouBo
"""

"""
compute the optimal policy for one MBS and two SBSs using PIA
general user set, cache set
"""

import time
import copy
import scipy.io as sio
import numpy as np
from scipy.special import comb


def PIeva_1MBS_2SBS(mu, N, Bmax, P, power, Num_queue):
    """
    relative value iteration method to compute the linear system of equations in policy evaluation step
    
    input:
        mu: given policy
        N: queue size vector
        Bmax: maximal arrival of B (depends on the number of users)
        P: arrival probability of B
        power: transmission power of BSs
        Num_queue: number of request queues

    output:
        Vnew: value function for given policy mu
    """
    Vnew = np.ones(N+1)
    Vold = np.zeros(N+1)
        
    epsilon = 0.5
    maxits = 50
    numits = 0
#    while np.amax(np.fabs(np.subtract(Vnew,Vold)))/np.amax(np.fabs(Vnew))>epsilon and numits<maxits:
    while np.amax(np.fabs(np.subtract(Vnew, Vold))) > epsilon and numits < maxits:

        Vold = copy.deepcopy(Vnew)
        
        #  enumerate all possible request queue states    
        for ind in np.arange(0, 1 + np.ravel_multi_index(N, N+1)):  
            #  given index, obtain the corresponding subscript
            sub = np.unravel_index(ind,N+1)
    
            u = np.array([mu[0][sub],mu[1][sub],mu[2][sub]])
            
            Vnew[sub] = cost_action_1MBS_2SBS(u,Vold,sub,N,Bmax,P,power,Num_queue)
        
        # relative value
        sub = np.unravel_index(0,N+1)  # reference state (0,0,0)
        Vnew = Vnew - Vnew[sub] 
         
        numits = numits+1
    print("iteration in PIeva:", numits)
    return Vnew


def cost_action_1MBS_2SBS(u,Vold,sub,N,Bmax,P,power,Num_queue):
    """
    compute the R.H.S. of each equation in policy evalution
    
    input:
        u: given control action
        Vold: previous value function
        sub: given request queue state
        N: queue size
        Bmax: maximal arrival of B (depends on the number of users)
        P: arrival probability of B
        power: transmission power of BSs
        Num_queue: number of request queues

    output:
        Vnew: value of the R.H.S. in policy evalution
    """
#    start_cost=time.time()
#    print("current cost_action function")  
    sub_up = [0 for n in np.arange(0,Num_queue)]
    temp = 0
    #  enumerate all possible request arrivals 
    for ind in np.arange(0,1+np.ravel_multi_index(Bmax,Bmax+1)):
        #  subscript of an index
        sub_B = np.unravel_index(ind,Bmax+1)
        #  different queue update for Q0, Q1 and Q2
        for n in np.arange(0,Num_queue):
            #  for Q0
            if n < M0:
                sub_up[n] = min((u[0] != n+1)*sub[n] + sub_B[n],N[n])
            #  for Q1
            elif n < M0 + M1:
                sub_up[n] = min((u[0] != M1_set[n - M0]+1)*(u[1] != M1_set[n - M0]+1)*sub[n] + sub_B[n],N[n])
            #  for Q2
            else:
                sub_up[n] = min((u[0] != M2_set[n - M0 - M1]+1)*(u[2] != M2_set[n - M0 - M1]+1)*sub[n] + sub_B[n],N[n])

#       one approach of \prod P, may be computationally complex
#       P_all=np.array([P[n][sub_B[n]] for n in np.arange(0,Num_queue)])
#       temp = temp + np.prod(P_all)*Vold[tuple(sub_up)]
        
        P_prod = 1
        
        for n in np.arange(0,Num_queue):
            P_prod = P_prod*P[n][sub_B[n]]
            
        temp = temp + P_prod*Vold[tuple(sub_up)]
    
#       power_all = [power[n] for n in np.arange(0,2) if u[n]!=0]

#    t_cost = time.time() - start_cost
#    print("one time of cost_action function:", t_cost)
    return temp + sum(sub) + np.dot(power,u)
    

def theta_cost(mu,N,Bmax,P,power,Num_queue):
    """
    compute average cost for given policy
    
    input:
        mu: given policy
        N: queue size
        Bmax: maximal arrival of B (depends on the number of users)
        P: arrival probability of B
        power: transmission power of BSs
        Num_queue: number of request queues

    output:
        theta_cost: value of average cost
    """
    V = PIeva_1MBS_2SBS(mu_o,N,Bmax,P,power,N_queue)
    
    sub = np.unravel_index(2,N+1)
    u = mu[0][sub],mu[1][sub],mu[2][sub]

    RHS = cost_action_1MBS_2SBS(u,V,sub,N,Bmax,P,power,Num_queue)

    return RHS - V[sub]


def PIimp_1MBS_2SBS(V,sub,N,Bmax,P,power,Num_queue,u_set):
    """
    policy improvement step
    
    input:
        V: given value function
        sub: given request request state
        N: queue size
        Bmax: maximal arrival of B (depends on the number of users)
        P: arrival probability of B
        power: transmission power of BSs
        Num_queue: number of request queues
        u_set: feasible action space

    output:
        u_set[u_ind_min]:  optimal value in current iteration step
    """
    sub_up = [0 for n in np.arange(0,Num_queue)]
    vtemp = [0 for n in np.arange(0,len(u_set))]

#    power_all=[0 for n in np.arange(0,len(u_set))]
        
    #  enumerate all possible actions   
    for u_ind, u in enumerate(u_set):
        #  enumerate all possible request arrivals   
        for ind in np.arange(0,1+np.ravel_multi_index(Bmax,Bmax+1)):
            sub_B = np.unravel_index(ind,Bmax+1)
            #  different queue update for Q0 and Q1
            for n in np.arange(0,Num_queue):
                if n < M0:
                    sub_up[n] = min((u[0] != n+1)*sub[n] + sub_B[n],N[n])
                #  for Q1
                elif n < M0 + M1:
                    sub_up[n] = min((u[0] != M1_set[n - M0]+1)*(u[1] != M1_set[n - M0]+1)*sub[n] + sub_B[n],N[n])
                #  for Q2
                else:
                    sub_up[n] = min((u[0] != M2_set[n - M0 - M1]+1)*(u[2] != M2_set[n - M0 - M1]+1)*sub[n] + sub_B[n],N[n])

#           one approach of \prod P, may be computationally complex
#           P_all=np.array([P[n][sub_B[n]] for n in np.arange(0,Num_queue)])
#           vtemp[u_ind] = vtemp[u_ind] + np.prod(P_all)*V[tuple(sub_up)]           
           
            P_prod = 1
            for n in np.arange(0,Num_queue):
                P_prod = P_prod*P[n][sub_B[n]]
            
            vtemp[u_ind] = vtemp[u_ind] + P_prod*V[tuple(sub_up)]
            
#        power_all[u_ind] = [power[n] for n in np.arange(0,2) if u[n]!=0]
        
#        vtemp[u_ind] = vtemp[u_ind]+ sum(sub) + sum(power_all[u_ind])
        vtemp[u_ind] = vtemp[u_ind] + sum(sub) + np.dot(power,u)
    #  obtain the optimal action    
    u_ind_min = np.argmin(vtemp)
    return u_set[u_ind_min]    


# system parameter

# number of BSs
N_BS = 3 

# contents
M = 2
M0 = M
M1 = 1
M2 = 1

# content set:  start with 0 in accordance with request indeices
M0_set = [n for n in range(0,M0)]
M1_set = [n for n in range(0,M1)]
M2_set = [n for n in range(1,M2+1)]

# action set
U0_set = [n for n in range(1,M0+1)]
U0_set.insert(0,0)
U1_set = [n for n in range(1,M1+1)]
U1_set.insert(0,0)
U2_set = [n for n in range(2,M2+1+1)]
U2_set.insert(0,0)

# all possible feasible control actions
u_set = [[U0_set[u0], U1_set[u1], U2_set[u2]] for u0 in range(len(U0_set)) for u1 in range(len(U1_set)) for u2 in range(len(U2_set)) if U0_set[u0] * (U1_set[u1] + U2_set[u2]) == 0]

# set of SBSs caching content m, i.e., Nm
Nm_set = []
# 1: SBS1, 2: SBS2

for m in range(0,M):
    temp = [1,2]
    if m not in M1_set:
        temp.remove(1)
    if m not in M2_set:
        temp.remove(2)
    Nm_set.append(temp)
        
# number of queues
N_queue = M0 + M1 + M2

# number of users
N_user = 3
K0_set = [1]
K1_set = [2]
K2_set = [3]

K0 = len(K0_set)
K1 = len(K1_set)
K2 = len(K2_set)

K = [K0, K1, K2]

# can set queue size according to cache state and number of users 
N = np.zeros(N_queue,int)
for n in range(0,N_queue):
    if n < M0:     
        #N[n] = 4 + (2-len(Nm_set[n]))
        N[n] = 3
    elif n < M0 + M1:
        N[n] = 3
    else:
        N[n] = 3
        
# power cost, power[0]: MBS; power[1]: SBS1; power[2]: SBS2
power = np.array([2,1,1])

# popularity profile: Zipf 
para = 0.6
Popularity = np.array([1/m**para for m in np.arange(1,M+1)])
Popularity = Popularity / np.sum(Popularity)

# request arrval
Amax = 1  # each user request at most one content

# construct B based on A, see request queue dynamics
""" 
    Bmax[n] n (0--M0-1): request for centent n can only be served by MBS
    Bmax[n] n (M0--M0+M1-1): request for conent m can  be served by MBS and SBS1
    Bmax[n] n (M0+M1,M0+M1+M2-1):  request for conent m can  be served by MBS and SBS2
"""
Bmax = np.zeros(N_queue,int)
for n in range(0,N_queue):
    if n < M0:     
        Bmax[n] = K0 + K1 * (2-len(Nm_set[n]))
    elif n < M0 + M1:
        Bmax[n] = K1
    else:
        Bmax[n] = K2

# calculate probality for B
P = []
for n in np.arange(0,N_queue):
    P.append(np.zeros(Bmax[n]+1))
    
for n in np.arange(0,N_queue):
    # mapping the index to to the content index in the cache 
    
    if n < M0:
        idx_content = n
    elif n < M0 + M1:
        idx_content = M1_set[n - M0]
    else:
        idx_content = M2_set[n - M0 - M1]
    
    for m in np.arange(0,Bmax[n]+1):            
        P[n][m] = comb(Bmax[n],m)*(Popularity[idx_content]**m)*((1-Popularity[idx_content])**(Bmax[n]-m))
  
num_of_iter = 1

t = []

for num_avg in np.arange(0,num_of_iter):
    mu_n_0 = np.zeros(N+1,np.int)  # MBS
    mu_n_1 = np.zeros(N+1,np.int)  # SBS1
    mu_n_2 = np.zeros(N+1,np.int)  # SBS2
    
    mu_o_0 = np.ones(N+1,np.int)  # MBS
    mu_o_1 = np.ones(N+1,np.int)  # SBS1
    mu_o_2 = np.ones(N+1,np.int)  # SBS2

    mu_n = np.array([mu_n_0, mu_n_1,  mu_n_2])  
    mu_o = np.array([mu_o_0, mu_o_1,  mu_n_2])
    
    V = np.zeros(N+1)
    
    theta_o = 0
    theta_n = 1
    num = 0

    while np.any(np.not_equal(mu_n,mu_o)) and abs(theta_o - theta_n) > 0.01:
        theta_o = theta_cost(mu_o,N,Bmax,P,power,N_queue)
        mu_o = copy.deepcopy(mu_n)

        #  policy evaluation        
        print("PI evaulation")
        start_eva = time.time()     
        V = PIeva_1MBS_2SBS(mu_o,N,Bmax,P,power,N_queue)
        t_eva = time.time()-start_eva
        print("one iteration of PIeva:", t_eva)    
        
        #  policy improvement
        print("PI improvement")  
        start_imp = time.time()
        #  enumerate all possible request queue states    
        for ind in np.arange(0,1+np.ravel_multi_index(N,N+1)):
            sub = np.unravel_index(ind,N+1)
            mu_n_0[sub],mu_n_1[sub],mu_n_2[sub] = PIimp_1MBS_2SBS(V,sub,N,Bmax,P,power,N_queue, u_set)
        
        mu_n = np.array([mu_n_0, mu_n_1, mu_n_2])  
        diff = np.count_nonzero(mu_n-mu_o)
        print("number of diffences:", diff)
        t_imp = time.time()-start_imp
        print("one iteration of PIimp:", t_imp)  
        num = num+1
        print("num:", num)    
        
        # compare the average costs of the two policies in adjacent iterations
        theta_n = theta_cost(mu_n,N,Bmax,P,power,N_queue)
        print("theta_o:", theta_o)
        print("theta_n:", theta_n)       
        
        t.append(t_eva+t_imp)
        
print("total time:", sum(t))

file_name = ["PIA_M0_",str(M0), "_M1_",str(M1), "_M2_", str(M2), "_SBS_2_N", str(N[0]),time.strftime("_%Y%m%d_%H_%M"), ".mat"]
file_name = "".join(file_name)
sio.savemat(file_name,{"mu_PIA0":mu_n_0,"mu_PIA1":mu_n_1, "mu_PIA2":mu_n_2, "V_PIA":V,"t_PIA":t, "num_PIA": num})
