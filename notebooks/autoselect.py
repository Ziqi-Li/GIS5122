#!/usr/bin/env python
# coding: utf-8

# ### Linear regression II (W5)
# Automatic model selection
# 1. Backward
# 2. Forward
# 3. Stepwise
# ### Author: Ziqi Li

import numpy as np
import pandas as pd
import statsmodels.api as sm


def backward_model_selection(y_name, X_names, df):
    """
    y_name: the column name of the depdent variable in the dataframe df
    
    X_names: the column names of candidate predictors
    
    df: the dataframe 
    
    """
    if X_names == []:
        print("\nstop")
        return
    
    y = df[y_name]
    X = df[X_names]
    X = sm.add_constant(X)
    current_model = sm.OLS(y,X).fit()
    print("\nCurrent model:",y_name,'~',' + '.join(['intercept'] + X_names))
    print('{:>0}  {:>12}  {:>10}'.format(" ", "current", np.around(current_model.aic,2)))
    
    best_aic = current_model.aic
    to_drop = None
    
    for x in X_names:
        no_x = [a for a in X_names if a != x]
        X = df[no_x]
        X = sm.add_constant(X)
        model = sm.OLS(y,X).fit()
        print('{:>0}  {:>12}  {:>10}'.format("-", x, np.around(model.aic,2)))
        if model.aic <= best_aic:
            to_drop = x
            best_aic = model.aic
          
    if to_drop:
        X_names.remove(to_drop)
        print("dropping ",to_drop)
        backward_model_selection(y_name, X_names, df)
            
    else: 
        print("\nstop")
        print("\nFinal model:",y_name,'~',' + '.join(['intercept'] + X_names))
        return
    


def forward_model_selection(y_name, X_names, df, start=[]):
    """
    y_name: the column name of the depdent variable in the dataframe df
    
    X_names: the column names of candidate predictors
    
    df: the dataframe 
    
    start: the starting list (defualt to an empty list)
    """
    if start == X_names:
        print("\nstop")
        return
    
    y = df[y_name]
    X = df[start]
    X = sm.add_constant(X)
    current_model = sm.OLS(y,X).fit()
    print("\nCurrent model:",y_name,'~',' + '.join(['intercept'] + start))
    print('{:>0}  {:>12}  {:>10}'.format(" ", "current", np.around(current_model.aic,2)))
    
    best_aic = current_model.aic
    
    to_add = None
    for x in [a for a in X_names if a not in start]:
        add_x = start + [x]
    
        X = df[add_x]
        X = sm.add_constant(X)
        model = sm.OLS(y,X).fit()
        print('{:>0}  {:>12}  {:>10}'.format("+", x, np.around(model.aic,2)))
        if model.aic <= best_aic:
            to_add = x
            best_aic = model.aic
          
    if to_add:
        start = start + [to_add]
        print("adding ",to_add)
        forward_model_selection(y_name, X_names, df, start)
            
    else: 
        print("\nstop")
        print("\nFinal model:",y_name,'~',' + '.join(['intercept'] + start))
        return
    


def stepwise_model_selection(y_name, X_names, df, start=[]):
    
    """
    y_name: the column name of the depdent variable in the dataframe df
    
    X_names: the column names of candidate predictors
    
    df: the dataframe 
    
    start: the starting list (defualt to an empty list)
    """
    if start == X_names:
        print("\nstop")
        return
    
    y = df[y_name]
    X = df[start]
    X = sm.add_constant(X)
    current_model = sm.OLS(y,X).fit()
    print("\nCurrent model:",y_name,'~',' + '.join(['intercept'] + start))
    print('{:>0}  {:>12}  {:>10}'.format(" ", "current", np.around(current_model.aic,2)))
    
    best_aic = current_model.aic
    
    to_add = None
    for x in [a for a in X_names if a not in start]:
        add_x = start + [x]
    
        X = df[add_x]
        X = sm.add_constant(X)
        model = sm.OLS(y,X).fit()
        print('{:>0}  {:>12}  {:>10}'.format("+", x, np.around(model.aic,2)))
        if model.aic <= best_aic:
            to_add = x
            best_aic = model.aic
    
    to_drop = None
    for x in start:
        no_x = [a for a in start if a != x]
    
        X = df[no_x]
        X = sm.add_constant(X)
        model = sm.OLS(y,X).fit()
        print('{:>0}  {:>12}  {:>10}'.format("-", x, np.around(model.aic,2)))
        if model.aic <= best_aic:
            to_drop = x
            best_aic = model.aic
    
    if to_drop:
        start.remove(to_drop)
        print("dropping ",to_drop)
        stepwise_model_selection(y_name, X_names, df, start)
        
    elif to_add:
        start = start + [to_add]
        print("adding ",to_add)
        stepwise_model_selection(y_name, X_names, df, start)
        
    else: 
        print("\nstop")
        print("\nFinal model:",y_name,'~',' + '.join(['intercept'] + start))
        return
    

