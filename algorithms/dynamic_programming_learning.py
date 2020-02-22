#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 09:09:51 2020

@author: jiyewang
"""

### 1 top-down Dynamic Programming ####

### 1.1 exponential-time recursive program ##

S = ['A','B','A','Z','D','C']
T = ['B','A','C','B','A','D']
n = 5
m = 5

#def LCS(S,n,T,m):
#    if n == 0 or m == 0:
#        return 0;
#    if S[n] == T[m]:
#        result = 1 + LCS(S,n-1,T,m-1);
#    else:
#        result = max(LCS(S,n-1,T,m), LCS(S,n,T,m-1))
#    return result

def LCS(S,n,T,m):
    if n == 0 or m == 0:
        if S[n] == T[m]:
            if n ==0 and m == 0:
                result = 1
            else:
                if n == 0:
                    result = 1+LCS(S,n,T,m-1)
                else:
                    result = 1+LCS(S,n-1,T,m)
        else:
            if n == 0 and m == 0:
                result = 0
            else:
                if n == 0:
                    result = LCS(S,n,T,m-1)
                else:
                    result = LCS(S,n-1,T,m)
    else:  
        if S[n] == T[m]:
            result = 1 + LCS(S,n-1,T,m-1);
        else:
            result = max(LCS(S,n-1,T,m), LCS(S,n,T,m-1))
    return result

print(LCS(S,n,T,m))


### 1.1 end ###

### 1.2 Memoized Version ###
import numpy as np

S = ['A','B','A','Z','D','C']
T = ['B','A','C','B','A','D']
n = 5
m = 5
array1 = np.zeros(shape = (n+1,m+1))
#print(array1[1][1])

def LCS(S,n,T,m):
    if array1[n][m] != 0:
        return array1[n][m];
    if n == 0 or m == 0:
        if S[n] == T[m]:
            if n ==0 and m == 0:
                result = 1
            else:
                if n == 0:
                    result = 1+LCS(S,n,T,m-1)
                else:
                    result = 1+LCS(S,n-1,T,m)
        else:
            if n == 0 and m == 0:
                result = 0
            else:
                if n == 0:
                    result = LCS(S,n,T,m-1)
                else:
                    result = LCS(S,n-1,T,m)
    else:  
        if S[n] == T[m]:
            result = 1 + LCS(S,n-1,T,m-1);
        else:
            result = max(LCS(S,n-1,T,m), LCS(S,n,T,m-1))
    array1[n][m] = result;
    return result

print(LCS(S,n,T,m))

### 1.2 end ###

### 1.3 knapsack problem ###

### exponential-time recursive  ###

value = [7, 9, 5, 12, 14, 6, 12]
size = [3, 4, 2, 6, 7, 3, 5]
S = 15
n = len(value)-1

def Value(n,S):
    if n == 0:
        if size[n] > S:
            return 0
        else:
            result = value[n]
    else:
        if size[n] > S:
            result = Value(n-1,S)
        else:
            result = max(value[n]+Value(n-1,S-size[n]),Value(n-1,S))
    return result

print(Value(n, S))
            
### Memoized Version ###
import numpy as np

value = [7, 9, 5, 12, 14, 6, 12]
size = [3, 4, 2, 6, 7, 3, 5]
S = 15
n = len(value)-1
array2 = np.zeros(shape = (n+1,S+1))

def Value(n,S):
    if array2[n][S] != 0:
        return array2[n][S]
    if n == 0:
        if size[n] > S:
            return 0
        else:
            result = value[n]
    else:
        if size[n] > S:
            result = Value(n-1,S)
        else:
            result = max(value[n]+Value(n-1,S-size[n]),Value(n-1,S))
    array2[n][S] = result
    return result

print(Value(n, S))

### 1.3 Matrix Product Parenthesization ###

### exponetial-time recursive ###
multi = [30,35,15,5,10,20,25]
multi_test = [30,35,15,5]
n_test = len(multi_test)-1
print(multi_test[0:2])
#
#def ministep(multi, n):
#    if n < 3:
#        return 0
#    if n == 3:
#        result = min((multi[n-3]*multi[n-2]*multi[n-1]+multi[n-3]*multi[n-1]*multi[n]),(multi[n]*multi[n-1]*multi[n-2]+multi[n]*multi[n-2]*multi[n-3]))
#    else:
##        result_list = []
##        for i in range(3,n):
##            result_list.append()
#        result = 0
#    return result
#
#print(ministep(multi_test,n_test))

multi = [30,35,15,5,10,20,25]
def ministep(multi):
    n = len(multi)-1
    if n < 2:
        return 0
    else:
        if n == 2:
            result = multi[n-2]*multi[n-1]*multi[n]
        else:
            for i in range(2,n):
                result = float('inf')
                if i == 2:
                    a = multi[i-2]*multi[i-1]*multi[i]+ministep(multi[i-1:n]) ### This formula is wrong
                else:
                    if i == n:
                        a = multi[i-2]*multi[i-1]*multi[i]+ministep(multi[0:i-1])
                    else:
                        a = ministep(multi[0:i-1])+ministep(multi[i-1:n])
                if a < result:
                    result = a
        return result
    
print(ministep(multi))

### Memoizing Version ###
import numpy as np

multi = [30,35,15,5,10,20,25]
n = len(multi)-1
array3 = np.zeros(shape = (n,n))

def ministep(multi):
    n = len(multi)-1
    if n < 2:
        return 0
    else:
        if n == 2:
            result = multi[n-2]*multi[n-1]*multi[n]
        else:
            for i in range(2,n):
                result = float('inf')
                if array3[n-1][i-1] != 0:
                    result = array3[n][i]
                    return result
                if i == 2:
                    a = multi[i-2]*multi[i-1]*multi[i]+ministep(multi[i-1:n+1])
                else:
                    if i == n:
                        a = multi[i-2]*multi[i-1]*multi[i]+ministep(multi[0:i-1])
                    else:
                        a = ministep(multi[0:i-1])+ministep(multi[i-1:n+1])
                if a < result:
                    result = a
                array3[n-1][i-1] = result
        return result
    
print(ministep(multi))

import numpy as np

multi = [30,35,15,5,10,20,25]
n = len(multi)-1
array3 = np.zeros(shape = (n,n))
    
def ministep(multi):
    n = len(multi)-1
    for i in range(2,n):
        for j in range(1,n-i+1):
            if array3[i-1][j-1] != 0:
                result = array3[i-1][j-1]
                return result
            if i == 2:
                result = multi[j-1]*multi[j]*multi[j+1]
                array3[i-1][j-1] = result
            else:
                if i == 3:
                    result_1 = multi[j-1]*multi[j]*multi[j+1]+multi[j-1]*multi[j+1]*multi[j+2]
                    result_2 = multi[j+2]*multi[j+1]*multi[j]+multi[j+2]*multi[j]*multi[j-1]
                    result = min(result_1,result_2)
                    array3[i-1][j-1] = result
                else:
                    result = float('inf')
                    for k in range(1, i-1):
                        if k == 1:
                            tmp_result = multi[j-1]*multi[j]*multi[j+n-1]+ministep(multi[j:j+n-1])
                        else:
                            if k == i-1:
                                tmp_result = multi[j-1]*multi[j+n-2]*multi[j+n-1]+ministep(multi[j-1:j+n-2])
                            else:
                                tmp_result = ministep(multi[j-1:j+k-1])+ministep(multi[j+k-1:j+n-1])
                        if tmp_result < result:
                            result = tmp_result
                        array3[i-1][j-1] = result
            return result
                    
print(ministep(multi))

import numpy as np

multi = [30,35,15,5,10,20,25]
n = len(multi)-1
array3 = np.zeros(shape = (n,n))

multi = [30,35,15,5,10,20,25]    
def ministep(multi):
    n = len(multi)-1
    for i in range(2,n):
        for j in range(1,n-i+1):
#            if array3[i-1][j-1] != 0:
#                result = array3[i-1][j-1]
#                return result
            if i == 2:
                result = multi[j-1]*multi[j]*multi[j+1]
#                array3[i-1][j-1] = result
            else:
                result = float('inf')
                for k in range(1, i-1):
                    if k == 1:
                        tmp_result = multi[j-1]*multi[j]*multi[j+n-1]+ministep(multi[j:j+n-1])
                    else:
                        if k == i-1:
                            tmp_result = multi[j-1]*multi[j+n-2]*multi[j+n-1]+ministep(multi[j-1:j+n-2])
                        else:
                            tmp_result = ministep(multi[j-1:j+k-1])+ministep(multi[j+k-1:j+n-1])+multi[j-1]*multi[j+n-1]*multi[k]
                    if tmp_result < result:
                        result = tmp_result
#                    array3[i-1][j-1] = result
            return result
                    
print(ministep(multi))

import numpy as np

multi = [30,35,15,5,10,20,25]
n = len(multi)-1
array3 = np.zeros(shape = (n,n))

def ministep(multi):
    n = len(multi)-1
    
    for i in range(2,n):
        for j in range(1,n-i+1):
#            if array3[i-1][j-1] != 0:
#                result = array3[i-1][j-1]
#                return result
            if i == 2:
                result = multi[j-1]*multi[j]*multi[j+1]
#                array3[i-1][j-1] = result
            else:
                result = float('inf')
                for k in range(1, i-1):
                    if k == 1:
                        tmp_result = multi[j-1]*multi[j]*multi[j+n-1]+ministep(multi[j:j+n-1])
                    else:
                        if k == i-1:
                            tmp_result = multi[j-1]*multi[j+n-2]*multi[j+n-1]+ministep(multi[j-1:j+n-2])
                        else:
                            tmp_result = ministep(multi[j-1:j+k-1])+ministep(multi[j+k-1:j+n-1])+multi[j-1]*multi[j+n-1]*multi[k]
                    if tmp_result < result:
                        result = tmp_result
#                    array3[i-1][j-1] = result
            return result
                    
print(ministep(multi))



p = [30, 35, 15, 5, 10, 20, 25] 
def matrix_chain_order(p):
    n = len(p) - 1   # 矩阵个数
    m = [[0 for i in range(n)] for j in range(n)] 
    s = [[0 for i in range(n)] for j in range(n)] # 用来记录最优解的括号位置
    for l in range(1, n): # 控制列，从左往右
        for i in range(l-1, -1, -1):  # 控制行,从下往上
            m[i][l] = float('inf') # 保存要填充格子的最优值
            for k in range(i, l):  # 控制分割点
                q = m[i][k] + m[k+1][l] + p[i]*p[k+1]*p[l+1]
                if q < m[i][l]:
                    m[i][l] = q
                    s[i][l] = k
    return m, s

def print_option_parens(s, i, j):
    if i == j:
        print('A'+str(i+1), end='')
    else:
        print('(', end='')
        print_option_parens(s, i, s[i][j])
        print_option_parens(s, s[i][j]+1, j)
        print(')', end='')

r, s = matrix_chain_order(p)
print_option_parens(s, 0, 5)
