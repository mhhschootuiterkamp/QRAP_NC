'''
Implementation of several algorithms for quadratic resource allocation problems with nested constraints

Contains implementations of the algorithms of:
    T. van der KLauw, M.E.T. Gerards, J.L. Hurink, "Resource allocation problems in decentralized energy management", OR Spectrum, 39:749-773, 2017
    T. Vidal, D. Gribel, P. Jaillet, "Separable convex optimization with nested lower and upper constraints", INFORMS Journal on Optimization, 1(1):71-90, 2019
    M.H.H. Schoot Uiterkamp, J.L. Hurink, M.E.T. Gerards, "A fast algorithm for quadratic resource allocation problems with nested constraints", arXiv:2009.03880, 2020


Copyright (c) 2021 Martijn Schoot Uiterkamp

'''


import math
import numpy
from collections import deque
import statistics
import heapq



def QRAP_median(objective_par,lower_bound,upper_bound,resource_value):
    #Solve an instance of QRAP using a meidan-find approach
    
    #Initialization
    num_var = len(objective_par)
    lower_breakpoints = [lower_bound[j]/objective_par[j] for j in range(0,num_var)]
    upper_breakpoints = [upper_bound[j]/objective_par[j] for j in range(0,num_var)] 
    current_breakpoints = lower_breakpoints + upper_breakpoints
    num_current_breakpoints = 2*num_var
    undecided_variables_indices = [j for j in range(0,num_var)]
    num_undecided_variables = num_var
    lower_breakpoint_bound = min(current_breakpoints)
    upper_breakpoint_bound = math.inf
    fixed = 0
    free = 0
    
    #Find consecutive breakpoints
    while num_undecided_variables > 0 and num_current_breakpoints > 0:
        #Find median breakpoint
        candidate_breakpoint = statistics.median(current_breakpoints)
        
        #Compute candidate sum:
        SummedVariables = 0
        for var_index in undecided_variables_indices:
            if upper_breakpoints[var_index] <= candidate_breakpoint:
                SummedVariables += upper_bound[var_index]
            elif lower_breakpoints[var_index] >= candidate_breakpoint:
                SummedVariables += lower_bound[var_index]
            else:
                SummedVariables += objective_par[var_index]*candidate_breakpoint
        SummedVariables += fixed + free*candidate_breakpoint
        
        #Determine relation between variable sum and resource value
        if abs(SummedVariables - resource_value) <= 10**(-6):
            opt_multiplier = candidate_breakpoint
            result = [max(lower_bound[j],min(upper_bound[j],objective_par[j]*opt_multiplier)) for j in range(0,num_var)]
            return result 
        elif SummedVariables > resource_value:
            upper_breakpoint_bound = candidate_breakpoint
            current_breakpoints = [BP for BP in current_breakpoints if BP < candidate_breakpoint]
            num_current_breakpoints = len(current_breakpoints)
        else:
            lower_breakpoint_bound = candidate_breakpoint
            current_breakpoints = [BP for BP in current_breakpoints if BP > candidate_breakpoint]
            num_current_breakpoints = len(current_breakpoints)
        
        #Determine value of newly decided variables
        new_undecided_variables = []
        for var_index in undecided_variables_indices:
            if upper_breakpoints[var_index] <= lower_breakpoint_bound:
                fixed += upper_bound[var_index]
                num_undecided_variables -= 1
            elif lower_breakpoints[var_index] <= lower_breakpoint_bound and upper_breakpoint_bound <= upper_breakpoints[var_index]:
                free += objective_par[var_index]
                num_undecided_variables -= 1
            elif lower_breakpoints[var_index] >= upper_breakpoint_bound:
                fixed += lower_bound[var_index]
                num_undecided_variables -= 1
            else:
                new_undecided_variables.append(var_index)
        undecided_variables_indices = new_undecided_variables
    
    #Compute optimal multiplier
    if free == 0:
        opt_multiplier = lower_breakpoint_bound
    else:
        opt_multiplier = (resource_value - fixed)/free
        
    result = [max(lower_bound[j],min(upper_bound[j],objective_par[j]*opt_multiplier)) for j in range(0,num_var)]
    return result  



def QRAP_NC_decomposition(objective_par,lower_bound,upper_bound,lower_nested,upper_nested):
    #Solves QRAP_NC using the algorithm from Vidal et al. (2019)
    
    #Define decomposition function MDA
    num_var = len(objective_par)
    result = numpy.zeros((2,2,num_var))
    
    def MDA(start_index,end_index,result):

        if start_index == end_index:
            #Only one variable!
            #Compute solutions!
            if start_index == 0:
                result[0][0][start_index] = lower_nested[start_index] 
                result[0][1][start_index] = upper_nested[start_index] 
                result[1][0][start_index] = lower_nested[start_index] 
                result[1][1][start_index] = upper_nested[start_index] 
            else:
                result[0][0][start_index] = lower_nested[start_index] - lower_nested[start_index - 1]
                result[0][1][start_index] = upper_nested[start_index] - lower_nested[start_index - 1]
                result[1][0][start_index] = lower_nested[start_index] - upper_nested[start_index - 1]
                result[1][1][start_index] = upper_nested[start_index] - upper_nested[start_index - 1]
        else:
            new_index = math.floor((start_index + end_index)/2)
            #Do decomposition steps
            MDA(start_index,new_index,result)
            MDA(new_index + 1, end_index,result)
            option_list = [[0,0],[0,1],[1,0],[1,1]]
            for subcase in option_list:
                #Determine new variable bounds
                current_lower_bounds = [result[subcase[0]][0][j] for j in range(start_index,new_index + 1)] + [result[1][subcase[1]][j] for j in range(new_index+1,end_index+1)]
                current_upper_bounds = [result[subcase[0]][1][j] for j in range(start_index,new_index + 1)] + [result[0][subcase[1]][j] for j in range(new_index+1,end_index+1)]
                alternative_lower_bounds = [0]*(end_index + 1 - start_index)
                alternative_upper_bounds = [0]*(end_index + 1 - start_index)
                for j in range(0,end_index + 1 - start_index):
                    if current_upper_bounds[j] < lower_bound[j + start_index]:
                        alternative_lower_bounds[j] = current_upper_bounds[j]
                    elif current_lower_bounds[j] > lower_bound[j + start_index]:
                        alternative_lower_bounds[j] = current_lower_bounds[j]
                    else:
                        alternative_lower_bounds[j] = lower_bound[j + start_index]
                        
                    if current_lower_bounds[j] > upper_bound[j + start_index]:
                        alternative_upper_bounds[j] = current_lower_bounds[j]
                    elif current_upper_bounds[j] < upper_bound[j + start_index]:
                        alternative_upper_bounds[j] = current_upper_bounds[j]
                    else:
                        alternative_upper_bounds[j] = upper_bound[j + start_index]

                #Determine new target resource value
                if start_index == 0:
                    if subcase[1] == 0:
                        target_value = lower_nested[end_index]
                    else:
                        target_value = upper_nested[end_index]
                else:
                    if subcase == [0,0]:
                        target_value = lower_nested[end_index] - lower_nested[start_index - 1]
                    elif subcase == [0,1]:
                        target_value = upper_nested[end_index] - lower_nested[start_index - 1]
                    elif subcase == [1,0]:
                        target_value = lower_nested[end_index] - upper_nested[start_index -1]
                    else:
                        target_value = upper_nested[end_index] - upper_nested[start_index - 1]
                
                sum_lower = sum(alternative_lower_bounds)
                sum_upper = sum(alternative_upper_bounds)
                if target_value < sum_lower:
                    sum_lower_original = sum(current_lower_bounds)      
                    HELP_sum = (target_value - sum_lower)/(sum_lower_original - sum_lower)                  
                    result[subcase[0]][subcase[1]][start_index:end_index + 1] = [alternative_lower_bounds[j] + HELP_sum*(current_lower_bounds[j] - alternative_lower_bounds[j]) for j in range(0,end_index - start_index +1)]
                elif target_value > sum_upper:
                    sum_upper_original = sum(current_upper_bounds)
                    HELP_sum = (target_value - sum_upper)/(sum_upper_original - sum_upper)
                    result[subcase[0]][subcase[1]][start_index:end_index + 1] = [alternative_upper_bounds[j] + HELP_sum*(current_upper_bounds[j] - alternative_upper_bounds[j]) for j in range(0,end_index - start_index+1)]
                else:       
                    #Solve QRAP subproblem
                    result[subcase[0]][subcase[1]][start_index:end_index + 1] = QRAP_median(objective_par[start_index:end_index+1], alternative_lower_bounds, alternative_upper_bounds, target_value)
        return result
    
    main_result = MDA(0,num_var-1,result)
    return main_result[0][1]


    
def QRAP_NC_seq(objective_par,lower_bound,upper_bound,lower_nested,upper_nested):
    #Solves QRAP-NC using approach of Schoot Uiterkamp et al. (2020)
    
    #Initialization of parameters:
    num_var = len(objective_par)      
    lower_nested[0] = max(lower_nested[0],lower_bound[0])
    upper_nested[0] = min(upper_nested[0],upper_bound[0])
    lower_bound[0] = max(lower_nested[0],lower_bound[0])
    upper_bound[0] = min(upper_nested[0],upper_bound[0])
    for j in range(1,num_var):
        if lower_nested[j-1] + lower_bound[j] > lower_nested[j]:
           lower_nested[j] = lower_nested[j-1] + lower_bound[j]
        if upper_nested[j-1] + upper_bound[j] < upper_nested[j]:
            upper_nested[j] = upper_nested[j-1] + upper_bound[j]
    
    #Initialization of breakpoints
    lower_breakpoints = [lower_bound[j]/objective_par[j] for j in range(0,num_var)]
    upper_breakpoints = [upper_bound[j]/objective_par[j] for j in range(0,num_var)]    
    
    #Initialization of optimal multiplier lists
    lower_opt_multiplier = [0]*num_var
    upper_opt_multiplier = [0]*num_var
    lower_opt_multiplier[0] = lower_breakpoints[0]
    upper_opt_multiplier[0] = upper_breakpoints[0]
    
    #Initialization of multiplier heaps (breakpoint sets to be updated throughout the algorithm)
    lower_breakpoints_heap_min = []
    upper_breakpoints_heap_min = []
    lower_breakpoints_heap_max = []
    upper_breakpoints_heap_max = []
    num_lower_breakpoints_heap = 0
    num_upper_breakpoints_heap = 0
    lower_opt_multiplier_deque = deque([0])
    num_lower_opt_multiplier_deque = 1
    upper_opt_multiplier_deque = deque([0])
    num_upper_opt_multiplier_deque = 1
    
    #Initialize flag paramters
    Removed_lower_BP = [0]*num_var
    Removed_upper_BP = [0]*num_var
    
    #Initialize bookkeeping parameters
    lower_fixed = [0]*num_var
    upper_fixed = [0]*num_var
    lower_free = [0]*num_var
    upper_free = [0]*num_var   
    lower_free[0] = objective_par[0]
    upper_free[0] = objective_par[0]
    
    #Start the sequential procedures!
    for j in range(1,num_var):

        #Initialize flags for adding new initial breakpoints
        FLAG_lower = 0
        FLAG_upper = 0
        
        #Apply initial checks for lower problem:
        Help_par = lower_nested[j-1] + max(lower_bound[j],min(upper_bound[j],objective_par[j]*lower_opt_multiplier[j-1]))
        if Help_par == lower_nested[j]:
            lower_opt_multiplier[j] = lower_opt_multiplier[j-1]
            if num_lower_opt_multiplier_deque > 0:
                if lower_opt_multiplier_deque[0] == j-1:
                    lower_opt_multiplier_deque[0] = j
            lower_fixed[j] = lower_fixed[j-1]
            lower_free[j] = lower_free[j-1]
            if lower_opt_multiplier[j] < lower_breakpoints[j]:
                lower_fixed[j] += lower_bound[j]
            elif lower_opt_multiplier[j] > upper_breakpoints[j]:
                lower_fixed[j] += upper_bound[j]
            else:
                lower_free[j] += objective_par[j]
        elif Help_par > lower_nested[j]:
            lower_opt_multiplier[j] = (lower_nested[j] - lower_nested[j-1])/objective_par[j]
            lower_fixed[j] = lower_nested[j-1]
            lower_free[j] = objective_par[j]
            lower_opt_multiplier_deque.appendleft(j)
            num_lower_opt_multiplier_deque += 1
        else:
            FLAG_lower = 1
            if num_lower_opt_multiplier_deque > 0:
                if lower_opt_multiplier_deque[0] == j-1:
                    lower_opt_multiplier_deque.popleft()
                    num_lower_opt_multiplier_deque -= 1
            lower_fixed_help = lower_fixed[j-1]
            lower_free_help = lower_free[j-1]
        
        #Apply initial checks for upper problem:
        Help_par = upper_nested[j-1] + max(lower_bound[j],min(upper_bound[j],objective_par[j]*upper_opt_multiplier[j-1]))
        if Help_par == upper_nested[j]:
            upper_opt_multiplier[j] = upper_opt_multiplier[j-1]
            if num_upper_opt_multiplier_deque > 0:
                if upper_opt_multiplier_deque[num_upper_opt_multiplier_deque - 1] == j-1:
                    upper_opt_multiplier_deque[num_upper_opt_multiplier_deque - 1] = j
            upper_fixed[j] = upper_fixed[j-1]
            upper_free[j] = upper_free[j-1]
            if upper_opt_multiplier[j] > upper_breakpoints[j]:
                upper_fixed[j] += upper_bound[j]
            elif upper_opt_multiplier[j] < lower_breakpoints[j]:
                upper_fixed[j] += lower_bound[j]
            else:
                upper_free[j] += objective_par[j]
        elif Help_par < upper_nested[j]:
            upper_opt_multiplier[j] = (upper_nested[j] - upper_nested[j-1])/objective_par[j]
            upper_fixed[j] = upper_nested[j-1]
            upper_free[j] = objective_par[j]
            upper_opt_multiplier_deque.append(j)
            num_upper_opt_multiplier_deque += 1
        else:
            FLAG_upper = 1
            if num_upper_opt_multiplier_deque > 0:
                if upper_opt_multiplier_deque[num_upper_opt_multiplier_deque - 1] == j-1:
                    upper_opt_multiplier_deque.pop()
                    num_upper_opt_multiplier_deque -= 1
            upper_fixed_help = upper_fixed[j-1]
            upper_free_help = upper_free[j-1]

        #If needed: add lower and upper breakpoints for j:
        #First: create search bounds for BPs:
        if FLAG_lower == 0:
            Threshold_lower = lower_opt_multiplier[j]
        else:
            Threshold_lower = lower_opt_multiplier[j-1]
        if FLAG_upper == 0:
            Threshold_upper = upper_opt_multiplier[j]
        else:
            Threshold_upper = upper_opt_multiplier[j-1]

        if Threshold_lower < lower_breakpoints[j] - 10**(-6) and lower_breakpoints[j] <= Threshold_upper - 10**(-6):
            num_lower_breakpoints_heap +=1
            heapq.heappush(lower_breakpoints_heap_min, [lower_breakpoints[j],j])
            heapq.heappush(lower_breakpoints_heap_max, [-lower_breakpoints[j],j])
        if Threshold_lower <= upper_breakpoints[j] - 10**(-6) and upper_breakpoints[j] < Threshold_upper - 10**(-6):
            num_upper_breakpoints_heap += 1
            heapq.heappush(upper_breakpoints_heap_min, [upper_breakpoints[j],j])
            heapq.heappush(upper_breakpoints_heap_max, [-upper_breakpoints[j],j])

        #If required: prepare for breakpoint search for the lower subproblem
        if FLAG_lower == 1:
            if lower_breakpoints[j] > lower_opt_multiplier[j-1]:
                lower_fixed_help = lower_fixed[j-1] + lower_bound[j]
            elif upper_breakpoints[j] > lower_opt_multiplier[j-1]:
                lower_free_help = lower_free[j-1] + objective_par[j]
            else:
                lower_fixed_help = lower_fixed[j-1] + upper_bound[j]
            
            #Start breakpoint search for lower problem!
            FLAG_found = 0
            Type_next_breakpoint = 0
            while FLAG_found == 0:
                #Determine next candidate breakpoint
                current_minimum_value = math.inf
                if num_lower_breakpoints_heap > 0:
                    FLAG_find_01 = 0
                    while FLAG_find_01 == 0:
                        candidate = lower_breakpoints_heap_min[0]
                        if Removed_lower_BP[candidate[1]] == 1:
                            heapq.heappop(lower_breakpoints_heap_min)
                        else:
                            candidate_breakpoint_index = candidate[1]
                            candidate_breakpoint_value = candidate[0]
                            FLAG_find_01 = 1                        
                    if candidate_breakpoint_value < current_minimum_value:
                        current_minimum_value = candidate_breakpoint_value
                        current_index = candidate_breakpoint_index
                        Type_next_breakpoint = 1
                if num_upper_breakpoints_heap > 0:
                    FLAG_find_01 = 0
                    while FLAG_find_01 == 0:
                        candidate = upper_breakpoints_heap_min[0]
                        if Removed_upper_BP[candidate[1]] == 1:
                            heapq.heappop(upper_breakpoints_heap_min)
                        else:
                            candidate_breakpoint_index = candidate[1]
                            candidate_breakpoint_value = candidate[0]
                            FLAG_find_01 = 1
                    if candidate_breakpoint_value < current_minimum_value:
                        current_minimum_value = candidate_breakpoint_value
                        current_index = candidate_breakpoint_index
                        Type_next_breakpoint = 2                    
                if num_lower_opt_multiplier_deque > 0:
                    candidate_breakpoint_index = lower_opt_multiplier_deque[0]
                    candidate_breakpoint_value = lower_opt_multiplier[candidate_breakpoint_index]
                    if candidate_breakpoint_value < current_minimum_value:
                        current_minimum_value = candidate_breakpoint_value
                        current_index = candidate_breakpoint_index
                        Type_next_breakpoint = 3                
                if num_upper_opt_multiplier_deque > 0:
                    candidate_breakpoint_index = upper_opt_multiplier_deque[0]
                    candidate_breakpoint_value = upper_opt_multiplier[candidate_breakpoint_index]
                    if candidate_breakpoint_value < current_minimum_value:
                        current_minimum_value = candidate_breakpoint_value
                        current_index = candidate_breakpoint_index
                        Type_next_breakpoint = 4                
                
                #Establish value of candidate breakpoint
                candidate_breakpoint = current_minimum_value
                
                #Go through possible cases for breakpoint values:
                if candidate_breakpoint == math.inf:
                    lower_opt_multiplier[j] = (lower_nested[j] - lower_fixed_help)/lower_free_help
                    lower_fixed[j] = lower_fixed_help
                    lower_free[j] = lower_free_help
                    lower_opt_multiplier_deque.appendleft(j)
                    num_lower_opt_multiplier_deque += 1
                    FLAG_found = 1
                else:
                    candidate_breakpoint_index = current_index
                    Summed_variables = lower_fixed_help + lower_free_help*candidate_breakpoint
                    if abs(Summed_variables - lower_nested[j]) <= 10**(-6):
                        lower_opt_multiplier[j] = candidate_breakpoint
                        lower_opt_multiplier_deque.appendleft(j)
                        num_lower_opt_multiplier_deque += 1
                        FLAG_found = 1
                        lower_fixed[j] = lower_fixed_help
                        lower_free[j] = lower_free_help
                    elif Summed_variables > lower_nested[j]:
                        lower_opt_multiplier[j] = (lower_nested[j] - lower_fixed_help)/lower_free_help
                        lower_fixed[j] = lower_fixed_help
                        lower_free[j] = lower_free_help
                        lower_opt_multiplier_deque.appendleft(j)
                        num_lower_opt_multiplier_deque += 1
                        FLAG_found = 1
                    else:
                        #Depending of type of breakpoint: update helper variables and delete breakpoint from corresponding heap or deque:
                        if Type_next_breakpoint == 1:
                            #Lower initial breakpoint
                            lower_fixed_help -= lower_bound[candidate_breakpoint_index]
                            lower_free_help += objective_par[candidate_breakpoint_index]
                            Removed_lower_BP[candidate_breakpoint_index] = 1
                            heapq.heappop(lower_breakpoints_heap_min)
                            num_lower_breakpoints_heap -= 1                                                                                               
                        elif Type_next_breakpoint == 2:
                            #Upper initial breakpoint
                            lower_fixed_help += upper_bound[candidate_breakpoint_index]
                            lower_free_help -= objective_par[candidate_breakpoint_index]   
                            Removed_upper_BP[candidate_breakpoint_index] = 1
                            heapq.heappop(upper_breakpoints_heap_min)
                            num_upper_breakpoints_heap -= 1                    
                        elif Type_next_breakpoint == 3:
                            #lower optimal multiplier breakpoint
                            lower_fixed_help -= lower_free[candidate_breakpoint_index]*candidate_breakpoint
                            lower_free_help += lower_free[candidate_breakpoint_index]
                            lower_opt_multiplier_deque.popleft()
                            num_lower_opt_multiplier_deque -= 1
                        else:
                            #Upper optimal multiplier breakpoint
                            lower_fixed_help += upper_free[candidate_breakpoint_index]*candidate_breakpoint
                            lower_free_help -= upper_free[candidate_breakpoint_index]
                            upper_opt_multiplier_deque.popleft()
                            num_upper_opt_multiplier_deque -= 1
        
        #If required: prepare for breakpoint search for upper subproblem
        if FLAG_upper == 1:
            if upper_breakpoints[j] < upper_opt_multiplier[j-1]:
                upper_fixed_help = upper_fixed[j-1] + upper_bound[j]
            elif lower_breakpoints[j] < upper_opt_multiplier[j-1]:
                upper_free_help = upper_free[j-1] + objective_par[j]
            else:
                upper_fixed_help = upper_fixed[j-1] + lower_bound[j]
            
            #Start breakpoint search for upper problem!
            FLAG_found = 0
            Type_next_breakpoint = 0
            while FLAG_found == 0:
                current_maximum_value = -1*math.inf
                #Determine candidate breakpoint value
                if num_lower_breakpoints_heap > 0:
                    FLAG_find_01 = 0
                    while FLAG_find_01 == 0:
                        candidate = lower_breakpoints_heap_max[0]
                        if Removed_lower_BP[candidate[1]] == 1:
                            heapq.heappop(lower_breakpoints_heap_max)
                        else:
                            candidate_breakpoint_index = candidate[1]
                            candidate_breakpoint_value = -candidate[0]
                            FLAG_find_01 = 1
                    if candidate_breakpoint_value > current_maximum_value:
                        current_maximum_value = candidate_breakpoint_value
                        current_index = candidate_breakpoint_index
                        Type_next_breakpoint = 1
                if num_upper_breakpoints_heap > 0:       
                    FLAG_find_01 = 0
                    while FLAG_find_01 == 0:
                        candidate = upper_breakpoints_heap_max[0]
                        if Removed_upper_BP[candidate[1]] == 1:
                            heapq.heappop(upper_breakpoints_heap_max)
                        else:
                            candidate_breakpoint_index = candidate[1]
                            candidate_breakpoint_value = -candidate[0]
                            FLAG_find_01 = 1
                    if candidate_breakpoint_value > current_maximum_value:
                        current_maximum_value = candidate_breakpoint_value
                        current_index = candidate_breakpoint_index
                        Type_next_breakpoint = 2                    
                if num_lower_opt_multiplier_deque > 0:
                    candidate_breakpoint_index = lower_opt_multiplier_deque[num_lower_opt_multiplier_deque-1]
                    candidate_breakpoint_value = lower_opt_multiplier[candidate_breakpoint_index]
                    if candidate_breakpoint_value > current_maximum_value:
                        current_maximum_value = candidate_breakpoint_value
                        current_index = candidate_breakpoint_index
                        Type_next_breakpoint = 3                
                if num_upper_opt_multiplier_deque > 0:
                    candidate_breakpoint_index = upper_opt_multiplier_deque[num_upper_opt_multiplier_deque - 1]
                    candidate_breakpoint_value = upper_opt_multiplier[candidate_breakpoint_index]
                    if candidate_breakpoint_value > current_maximum_value:
                        current_maximum_value = candidate_breakpoint_value
                        current_index = candidate_breakpoint_index
                        Type_next_breakpoint = 4                
                
                #Establish candidate breakpoint value
                candidate_breakpoint = current_maximum_value

                #Go through possible options for breakpoint value:
                if candidate_breakpoint == -math.inf:
                    upper_opt_multiplier[j] = (upper_nested[j] - upper_fixed_help)/upper_free_help
                    upper_fixed[j] = upper_fixed_help
                    upper_free[j] = upper_free_help
                    upper_opt_multiplier_deque.append(j)
                    num_upper_opt_multiplier_deque += 1
                    FLAG_found = 1
                else:
                    candidate_breakpoint_index = current_index
                    Summed_variables = upper_fixed_help + upper_free_help*candidate_breakpoint
                    if abs(Summed_variables - upper_nested[j]) <= 10**(-6):
                        upper_opt_multiplier[j] = candidate_breakpoint
                        upper_opt_multiplier_deque.append(j)
                        num_upper_opt_multiplier_deque += 1
                        FLAG_found = 1
                        upper_fixed[j] = upper_fixed_help
                        upper_free[j] = upper_free_help
                    elif Summed_variables < upper_nested[j]:
                        upper_opt_multiplier[j] = (upper_nested[j] - upper_fixed_help)/upper_free_help
                        upper_fixed[j] = upper_fixed_help
                        upper_free[j] = upper_free_help
                        upper_opt_multiplier_deque.append(j)
                        num_upper_opt_multiplier_deque += 1
                        FLAG_found = 1
                    else:
                        #Depending of type of breakpoint: update helper variables and delete breakpoint from corresponding heap or deque:
                        if Type_next_breakpoint == 1:
                            #lower initial breakpoint
                            upper_fixed_help += lower_bound[candidate_breakpoint_index]
                            upper_free_help -= objective_par[candidate_breakpoint_index]
                            Removed_lower_BP[candidate_breakpoint_index] = 1
                            num_lower_breakpoints_heap -= 1
                            heapq.heappop(lower_breakpoints_heap_max)                                                                                                      
                        elif Type_next_breakpoint == 2:
                            #Upper initial breakpoint
                            upper_fixed_help -= upper_bound[candidate_breakpoint_index]
                            upper_free_help += objective_par[candidate_breakpoint_index]
                            Removed_upper_BP[candidate_breakpoint_index] = 1
                            num_upper_breakpoints_heap -= 1
                            heapq.heappop(upper_breakpoints_heap_max)
                        elif Type_next_breakpoint == 3:
                            #lower optimal multiplier breakpoint
                            upper_fixed_help += lower_free[candidate_breakpoint_index]*candidate_breakpoint
                            upper_free_help -= lower_free[candidate_breakpoint_index]
                            lower_opt_multiplier_deque.pop()
                            num_lower_opt_multiplier_deque -= 1
                        else:
                            #upper optimal multiplier breakpoint
                            upper_fixed_help -= upper_free[candidate_breakpoint_index]*candidate_breakpoint
                            upper_free_help += upper_free[candidate_breakpoint_index]
                            upper_opt_multiplier_deque.pop()
                            num_upper_opt_multiplier_deque -= 1           

    #Initialize placeholders for intermediate multiplier values (chi)
    lower_opt_multiplier_new = [0]*num_var
    upper_opt_multiplier_new = [0]*num_var

    lower_opt_multiplier_new[num_var-1] = lower_opt_multiplier[num_var-1]
    upper_opt_multiplier_new[num_var-1] = upper_opt_multiplier[num_var-1]
    
    #Initialize placeholders for optimal solutions   
    opt_lower = [0]*num_var
    opt_upper = [0]*num_var
    opt_lower[num_var-1] = max(lower_bound[num_var-1],min(upper_bound[num_var-1],objective_par[num_var-1]*lower_opt_multiplier_new[num_var-1]))               
    opt_upper[num_var-1] = max(lower_bound[num_var-1],min(upper_bound[num_var-1],objective_par[num_var-1]*upper_opt_multiplier_new[num_var-1]))
    
    #Compute intermediate multiplier values 
    for j in range(num_var - 2,-1,-1):
        if upper_opt_multiplier[j] <= lower_opt_multiplier_new[j+1]:
            lower_opt_multiplier_new[j] = upper_opt_multiplier[j]
        elif lower_opt_multiplier[j] >= lower_opt_multiplier_new[j+1]:
            lower_opt_multiplier_new[j] = lower_opt_multiplier[j]
        else:
            lower_opt_multiplier_new[j] = lower_opt_multiplier_new[j+1]
            
        if upper_opt_multiplier[j] <= upper_opt_multiplier_new[j+1]:
            upper_opt_multiplier_new[j] = upper_opt_multiplier[j]
        elif lower_opt_multiplier[j] >= upper_opt_multiplier_new[j+1]:
            upper_opt_multiplier_new[j] = lower_opt_multiplier[j]
        else:
            upper_opt_multiplier_new[j] = upper_opt_multiplier_new[j+1]        
        
        if lower_bound[j] > objective_par[j]*lower_opt_multiplier_new[j]:
            opt_lower[j] = lower_bound[j]
        elif upper_bound[j] < objective_par[j]*lower_opt_multiplier_new[j]:
            opt_lower[j] = upper_bound[j]
        else:
            opt_lower[j] = objective_par[j]*lower_opt_multiplier_new[j]

        if lower_bound[j] > objective_par[j]*upper_opt_multiplier_new[j]:
            opt_upper[j] = lower_bound[j]
        elif upper_bound[j] < objective_par[j]*upper_opt_multiplier_new[j]:
            opt_upper[j] = upper_bound[j]
        else:
            opt_upper[j] = objective_par[j]*upper_opt_multiplier_new[j]
                        
    #Compute optimal solution
    if lower_nested[num_var - 1] == upper_nested[num_var - 1]:
        return opt_lower
    else:
        opt = [max(opt_lower[j],min(opt_upper[j],objective_par[j]*0)) for j in range(0,num_var)]
        return opt



def QRAP_NC_infeasible(objective_par,lower_bound,upper_bound,lower_nested,upper_nested):
    #Compute optimal solution to QRAP_NC via infeasibility-guided algorithm of Van der Klauw et al. (2017)

    #Initialize parameters    
    num_var = len(objective_par)

    lower_nested2 = [lower_nested[i] for i in range(0,num_var)]
    upper_nested2 = [upper_nested[i] for i in range(0,num_var)]
    
    Sol_vec = [0]*num_var
    HELP_num = 0
    HELP_num_2 = 0
    
    #The main algorithm
    if num_var == 1:
        Sol_vec = [upper_nested2[0]]
    else:
        #Solve relaxation
        Sol_vec = QRAP_median(objective_par, lower_bound, upper_bound, upper_nested2[num_var-1])
        
        #Search for most violated constraint
        VAR_sum = 0
        Max_index = -1
        Violation_value = 0
        Violation_type = -1
        for i in range(0,num_var):
            VAR_sum += Sol_vec[i]
            if VAR_sum > upper_nested2[i] + pow(10,-5):
                if VAR_sum - upper_nested2[i] > Violation_value:
                    Violation_value = VAR_sum - upper_nested2[i]
                    Violation_type = 1
                    Max_index = i
            elif VAR_sum < lower_nested2[i] - pow(10,-5):
                if lower_nested2[i] - VAR_sum > Violation_value:
                    Violation_value = lower_nested2[i] - VAR_sum
                    Violation_type = 0
                    Max_index = i
                    
        #Set new lower and upper nested bound according to most violated constraint
        if Violation_type == 1:
            lower_nested2[Max_index] = upper_nested2[Max_index]
            for i in range(Max_index+1,num_var):
                lower_nested2[i] -= upper_nested2[Max_index]
                upper_nested2[i] -= upper_nested2[Max_index]
        elif Violation_type == 0:
            upper_nested2[Max_index] = lower_nested2[Max_index]
            for i in range(Max_index+1,num_var):
                lower_nested2[i] -= lower_nested2[Max_index]
                upper_nested2[i] -= lower_nested2[Max_index]
        
        #Do decomoposition and solve two new instances!
        if Violation_type != -1:
            Sol_vec[0:Max_index+1] = QRAP_NC_infeasible(objective_par[0:Max_index+1], lower_bound[0:Max_index+1], upper_bound[0:Max_index+1], lower_nested2[0:Max_index+1], upper_nested2[0:Max_index+1])
            Sol_vec[Max_index+1:num_var] = QRAP_NC_infeasible(objective_par[Max_index+1 :num_var], lower_bound[Max_index+1:num_var], upper_bound[Max_index+1:num_var], lower_nested2[Max_index+1:num_var], upper_nested2[Max_index+1:num_var])
    return Sol_vec