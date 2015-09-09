# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:26:04 2015

@author: ZhouBo
"""

"""
Compute the optimal policy using structured policy iteration
"""


import time
import copy
import scipy.io as sio
import numpy as np
from scipy.special import comb


#from numbapro import vectorize

#@vectorize(['float64(int32, int, int32, list, int32, int)'], target='cpu')
def PIeva_M0_2_M1_1_SBS_1(mu, N, Bmax, P, power, Num_queue):
    """
    relative value iteration method to compute the linear system of equations in policy evaluation step
    
    input:
        mu: given policy
        N: queue size
        Bmax: maximal arrival of B (depends on the number of users)
        P: arrival probability of B
        power: transmission power of BSs
        Num_queue: number of request queues

    output:
        Vnew: value function for given policy mu
    """
    Vnew = np.ones((N+1, N+1, N+1))
    Vold = np.zeros((N+1, N+1, N+1))
        
    epsilon = 0.1
    maxits = 50
    numits = 0
    
#    while np.amax(np.fabs(np.subtract(Vnew,Vold)))/np.amax(np.fabs(Vnew))>epsilon and numits<maxits:
    while np.amax(np.fabs(np.subtract(Vnew, Vold))) > epsilon and numits < maxits:

        Vold = copy.deepcopy(Vnew)
        
        #  enumerate all possible request queue states    
        for ind in np.arange(0, 1 + np.ravel_multi_index([N, N, N], [N+1,N+1,N+1])):  
            #  given index, obtain the corresponding subscript
            sub = np.unravel_index(ind,[N+1,N+1,N+1])
    
            u = np.array([mu[0][sub],mu[1][sub]])
            
            Vnew[sub] = cost_action_M0_2_M1_1_SBS_1(u,Vold,sub,N,Bmax,P,power,Num_queue)
        
        # relative value
        sub = np.unravel_index(0,[N+1,N+1,N+1])  # reference state (0,0,0)
        Vnew = Vnew - Vnew[sub] 
         
        numits = numits+1

    print("iteration in PIeva:", numits)
    return Vnew

    
#@vectorize(['float64(int32, float64, list, int, int32, list, int32, int)'], target='cpu')
def cost_action_M0_2_M1_1_SBS_1(u,Vold,sub,N,Bmax,P,power,Num_queue):
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
    sub_up = [0 for n in np.arange(0,Num_queue)]
    temp = 0
    #  enumerate all possible request arrivals 
    for ind in np.arange(0,1+np.ravel_multi_index(Bmax,Bmax+1)):
        #  subscript of an index
        sub_B = np.unravel_index(ind,Bmax+1)
        #  different queue update for Q0 and Q1
        for n in np.arange(0,Num_queue):
            #  for Q0
            if n != 2:
                sub_up[n] = min((u[0] != n+1)*sub[n] + sub_B[n],N)
            #  for Q1
            else:
                sub_up[n] = min((u[0] != 1)*(u[1] != 1)*sub[n] + sub_B[n],N)

#       one approach of \prod P, may be computationally complex
#       P_all=np.array([P[n][sub_B[n]] for n in np.arange(0,Num_queue)])
#       temp = temp + np.prod(P_all)*Vold[tuple(sub_up)]
        
        P_prod = 1
        
        for n in np.arange(0,Num_queue):
            P_prod = P_prod*P[n][sub_B[n]]
            
        temp = temp + P_prod*Vold[tuple(sub_up)]
    
#       power_all = [power[n] for n in np.arange(0,2) if u[n]!=0]
    
    return temp + sum(sub) + np.dot(power,u)
            

#@vectorize(['float32(float32, float32)'], target='cpu')
def PIimp_M0_2_M1_1_SBS_1(V,sub,N,Bmax,P,power,Num_queue):
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

    output:
        u_set[u_ind_min]:  optimal value in current iteration step
    """

    #  all possible feasible actions
    u_set = [[u0,u1] for u0 in np.arange(0,3) for u1 in np.arange(0,2) if u0*u1 == 0]

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
                if n != 2:
                    sub_up[n] = min((u[0] != n+1)*sub[n] + sub_B[n],N)
                else:
                    sub_up[n] = min((u[0] != 1)*(u[1] != 1)*sub[n] + sub_B[n],N)

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

# contents
M = 2
M0 = 2
M1 = 1

# number of queues
Num_queue = M0 + M1

# number of users
K = 4
K0 = 2
K1 = 2

# queue size
N = 5

# power cost, power[0]: MBS; power[1]: SBS
power = np.array([4,2])

# popularity profile: Zipf 
para = 0.75
Popularity = np.array([1/m**para for m in np.arange(1,M+1)])
Popularity = Popularity / np.sum(Popularity)

# request arrval
Amax = 1  # each user request at most one content

Bmax = np.array([K0,K,K1])
# Bmax[1]: users in K0 for centent 1 & can only be served by MBS
# Bmax[2]: users in K for content 2 & can only be served by MBS
# Bmax[3]: users in K1 for content 1 [can be served by both MBS and SBS]

P = []
for n in np.arange(0,Num_queue):
    P.append(np.zeros(Bmax[n]+1))
    
for n in np.arange(0,Num_queue):
    for m in np.arange(0,Bmax[n]+1):    
        # content_index can be computed based on n and m
        # mapping the index to to the content index in the cache 
        
        # construct B using A according the request queue model
        if (n+1) % 2 != 0:
            idx_content = (n+1) % 2-1
        else:
            idx_content = 2-1
            
        P[n][m] = comb(Bmax[n],m)*(Popularity[idx_content]**m)*((1-Popularity[idx_content])**(Bmax[n]-m))
        
num_of_iter = 1

t = []

for num_avg in np.arange(0,num_of_iter):

    mu_n_0 = np.zeros((N+1,N+1,N+1),np.int)  # MBS
    mu_n_1 = np.zeros((N+1,N+1,N+1),np.int)  # SBS
    
    mu_o_0 = np.ones((N+1,N+1,N+1),np.int)   # MBS
    mu_o_1 = np.ones((N+1,N+1,N+1),np.int)   # SBS

    mu_n = np.array([mu_n_0, mu_n_1])    
    mu_o = np.array([mu_o_0, mu_o_1])  
    
    V = np.zeros((N+1,N+1,N+1))
    num = 0
    while np.any(np.not_equal(mu_n,mu_o)):
        start = time.time()
        mu_o = copy.deepcopy(mu_n)

        #  policy evaluation        
        V = PIeva_M0_2_M1_1_SBS_1(mu_o,N,Bmax,P,power,Num_queue)

        #  structured policy improvement
        #  enumerate all possible request queue states    
        for ind in np.arange(0,1+np.ravel_multi_index([N,N,N],[N+1,N+1,N+1])):
            sub = np.unravel_index(ind,[N+1,N+1,N+1])
            flag = 1
            for n in np.arange(0,Num_queue):
                if sub[n] > 0:
                    sub_de = np.subtract(sub, np.eye(Num_queue,dtype=int)[n,:]) 
                    
                    if n < M0:
                        idx_BS = 0
                        idx_content = n
                    else:
                        idx_BS = 1
                        idx_content = 0
                    
                    if mu_o[idx_BS][tuple(sub_de)] == idx_content + 1:
                        mu_n_0[sub],mu_n_1[sub] = mu_o[0][tuple(sub_de)], mu_o[1][tuple(sub_de)] 
                        flag = 0
                        break
                    
            if flag == 1:
                mu_n_0[sub],mu_n_1[sub] = PIimp_M0_2_M1_1_SBS_1(V,sub,N,Bmax,P,power,Num_queue)
        
        mu_n = np.array([mu_n_0, mu_n_1])    
        print ("diffence number", np.count_nonzero(mu_n-mu_o))
        num = num+1
        print("num:", num)
        
        t.append(time.time()-start)
        
print ("total time:", sum(t))

file_name = ["M0_",str(M0), "_M1_",str(M1), "_SBS_1_N", str(N),"_SPIA_",time.strftime("_%Y%m%d_%H_%M"), ".mat"]
file_name = "".join(file_name)
sio.savemat(file_name,{"mu_SPIA":mu_n_0,"mu_SPIA2":mu_n_1,"V_SPIA":V,"t_SPIA":t})
