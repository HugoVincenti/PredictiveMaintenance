# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 00:21:57 2022

@author: hugov
"""

from math import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from time import time


sensitivity_tab = \
[\
#["scale", 25,35,45],\
#["shape",10,15,20],\
#["z_lim",0.2,0.3,0.4],\
#["one_sensor_investment",1000,2500,5000],\
#["preventive_cost_pu",0.3,0.5,0.7]\
["clients_per_line",1000,2000,4000]
]

number_of_variables = len(sensitivity_tab)
#%% 
plt.close('all')
a=time()
for variable_i in range (number_of_variables):
    d = time()
    chosenVariable = sensitivity_tab[variable_i][0]
    tab_testedValues = sensitivity_tab[variable_i][1:]
    print(chosenVariable, tab_testedValues)
    '''Useful constants'''
 #%%

    y2h = 365.25*24 #number of hours in a year
    y2d = 365
    h2y = 1/y2h #multiply by this to transform hours into years
    simulationLength = int(55*y2d) #number of steps in our algorithm
    simulationSteps = np.linspace(0,55,simulationLength)
    
    corrective_cost_pu = 1 #By definition
    corrective_ENS_pu = 1 #By definition
    
    '''Default Parameters'''
    
    z_lim = 0.3
    preventive_cost_pu = 0.5
    preventive_ENS_pu = 0.5
    one_sensor_investment = 2500 #in €
    sensors_annual_cost = one_sensor_investment*0.05 #
    scale = 45
    shape = 15
    clients_per_line = 2000
    Q_std_dev = 0.5 #percent of Q
    IR = 0.07 #discount rate = 7%
    ref_OPEX = 15e3 #One reparation costs 15,000€
    ref_ENS = 1 #for ONE client : 12kWh a day on average: two hours lost = 1 kWh lost
    cost_ENS = 10 #according to Enedis : 10€/kwh 
    
#%%
    ''' Chosen parameter for sensibility analysis '''
    for testedValue in tab_testedValues:
        exec("%s = %f" % (chosenVariable,testedValue))
        print(chosenVariable + " = " + str(testedValue))
        ''' Functions '''
        
        pdfFailure = scipy.stats.weibull_min.pdf(simulationSteps,shape,scale=scale)
        reliability = 1 - scipy.stats.weibull_min.cdf(simulationSteps,shape,scale=scale)
        failureRate = np.divide(pdfFailure,reliability)
        
        def PD_to_HI(PD):
            if PD<250:
                HI = (PD-0)*(1-0.75)/(250-0)
            elif PD<350:
                HI = 0.25+(PD-250)*(0.75-0.5)/(350-250)
            else:
                HI = 0.5+(PD-350)*(0.5-0.25)/(500-350)
            return HI
            
        def HI_to_z(HI):
            A = 0.01976
            B = 3.4295969
        #    C = -0.009756098
            C=-A
            z = A*exp(B*HI)+C
            return z
        
        def z_to_HI(z):
            A = 0.01976
            B = 3.4295969
        #    C = -0.009756098
            C=-A
            HI = log((z-C)/A)/B
            return HI
        
        def HI_to_PD(HI):
            if HI<0.25:
                PD = HI*(250-0)/(1-0.75)
            elif HI<0.5:
                PD = 250+(HI-0.25)*(350-250)/(0.75-0.5)
            else:
                PD = 350+(HI-0.5)*(500-350)/(0.5-0.25)
            return PD
        
        '''Retro-engineered to get the "generated PD" goal" Q_obj'''
        
        z_obj = np.zeros(simulationLength)
        HI_obj = np.zeros(simulationLength)
        PD_obj = np.zeros(simulationLength)
        Q_obj = np.zeros(simulationLength)
        
        Q_obj[0] = 0 
        for t in range (1,simulationLength):
            z_obj[t] = failureRate[t]
            HI_obj[t] = z_to_HI(z_obj[t])
            PD_obj[t] = HI_to_PD(HI_obj[t])
            Q_obj[t] = PD_obj[t]-PD_obj[t-1]
        
        '''Simulation'''
        
        N=10 #Number of lines for the simulation
    
        MC_iterations = 10
        CM_all_iterations = np.zeros((MC_iterations,simulationLength))     
        PM_all_iterations = np.zeros((MC_iterations,simulationLength))
        for iteration in range (MC_iterations):        
            '''Initialisation of tables'''
        
            OPEX_CM_pu = np.zeros(simulationLength)
            OPEX_PM_pu = np.zeros(simulationLength)
            ENS_CM_pu = np.zeros(simulationLength)
            ENS_PM_pu = np.zeros(simulationLength)
        
            actualised_cost_CM = np.zeros((N,simulationLength))
            actualised_cost_PM = np.zeros((N,simulationLength))
    
            '''Initialisation of the ages of the line '''
        
            Tline_init = np.linspace(0, scale*y2d, num = N, dtype=int)
            PD_init = np.zeros(N)
            for i in range(N):
                for T_init in range (Tline_init[i]):
                    Q = max(0,np.random.normal(Q_obj[T_init],Q_std_dev*Q_obj[T_init]))
                    PD_init[i]+=Q
            '''Scenario 1: no predictive maintenance'''
            time_of_failure_no_PM = np.zeros(N)
            for i in range (N):
                Tsim = 0
                while(Tsim<simulationLength):
                    if Tsim == 0:
                        PD = PD_init[i]
                        Tline = Tline_init[i]
                    else:
                        PD = 0
                        Tline = 0
                    alive = True
                    while (Tsim<simulationLength and alive):
                        Q = max(0,np.random.normal(Q_obj[Tline],Q_std_dev*Q_obj[Tline])) #Step 1
                        PD+=Q
                        HI = PD_to_HI(PD) #Step 2
                        z = HI_to_z(HI) #Step 3
                        rdm = np.random.random()
                        cost = 0
                        if (rdm<z*(1/y2d)): #probability of failure between t and t+dt is z(t)*dt
                            alive = False
                            time_of_failure_no_PM[i] = Tline
                            OPEX_CM_pu[i] = corrective_cost_pu
                            ENS_CM_pu[i] = corrective_ENS_pu
                            cost = corrective_cost_pu*ref_OPEX+\
                            corrective_ENS_pu*ref_ENS*cost_ENS*clients_per_line
                        actualised_cost_CM[i,Tsim]=actualised_cost_CM[i,Tsim-1]+\
                        (cost)/(1+IR)**int(Tsim/y2d)
                        Tsim+=1
                        Tline+=1
            
            'Plot all the lines'
            total_cost_CM = np.sum(actualised_cost_CM, axis=0)
        #    plt.figure('100 lines CM cost over time')
        #    for i in range (N):
        #        plt.plot(actualised_cost_CM[i,:])
        #    plt.plot(total_cost_CM)
        #    plt.show()
        #            
            '''Scenario 2: predictive maintenance'''
            z_for_plot = np.zeros((N,simulationLength))
            time_of_failure_PM = np.zeros(N)
            time_of_PM = np.zeros(N)
            for i in range (N):
                Tsim = 0
                while (Tsim<simulationLength):
                    if Tsim == 0:
                        PD = PD_init[i]
                        Tline = Tline_init[i]
                    else:
                        PD = 0
                        Tline = 0
                    alive = True
                    while (Tsim<simulationLength and alive):
                        Q = max(0,np.random.normal(Q_obj[Tline],Q_std_dev*Q_obj[Tline])) #Step 1
                        PD+=Q
                        HI = PD_to_HI(PD) #Step 2
                        z = HI_to_z(HI) #Step 3
                        z_for_plot[i,Tsim] = z
                        cost = sensors_annual_cost/y2d
                        if (z>z_lim): #Preventive action is taken
                            alive = False
                            time_of_PM[i] = Tline
                            OPEX_PM_pu[i] = preventive_cost_pu
                            ENS_PM_pu[i] = preventive_ENS_pu
                            cost += preventive_cost_pu*ref_OPEX+\
                            preventive_ENS_pu*ref_ENS*cost_ENS*clients_per_line
                        rdm = np.random.random()
                        if (alive and rdm<z*(1/y2d)): #in this case, we have not predicted the failure, so corrective parameters apply
                            alive = False
                            time_of_failure_PM[i] = Tline
                            OPEX_PM_pu[i] = corrective_cost_pu
                            ENS_PM_pu[i] = corrective_ENS_pu
                            cost += corrective_cost_pu*ref_OPEX \
                            +corrective_ENS_pu*ref_ENS*cost_ENS*clients_per_line
                        actualised_cost_PM[i,Tsim] = actualised_cost_PM[i,Tsim-1] +\
                        cost/(1+IR)**int(Tsim/y2d)
                        if Tsim == 0:
                            actualised_cost_PM[i,Tsim] = one_sensor_investment
                        Tsim+=1
                        Tline+=1
                
            'Plot all the lines'
            total_cost_PM = np.sum(actualised_cost_PM, axis=0)
            CM_all_iterations[iteration] = total_cost_CM
            PM_all_iterations[iteration] = total_cost_PM
        'Comparison between PM and CM'
        CM_averaged_iterations = np.average(CM_all_iterations,axis=0)
        PM_averaged_iterations = np.average(PM_all_iterations,axis=0)
        
        plt.figure('Comparison between CM and PM for ' + chosenVariable + ' = ' + str(testedValue))
        plt.plot(simulationSteps,CM_averaged_iterations/1000,label="Corrective Maintenance")
        plt.plot(simulationSteps,PM_averaged_iterations/1000, label="Predictive Maintenance")
        plt.xlabel('Time (years)')
        plt.ylabel('Cost of maintenance (k€)')
        plt.ylim([-5,250])
        plt.legend(loc="lower right")
        plt.title(chosenVariable+"="+str(testedValue))
        plt.savefig("Images_Report/Results_python/"+chosenVariable+str(testedValue)+".svg")

    c  = time()
    print("Done for ",chosenVariable,"in ",c-d,"s")
b=time()
print("Done: running everything took",b-a, "s")