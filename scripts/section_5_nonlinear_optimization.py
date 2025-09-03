# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Import libraries (do we use all of these?)

import numpy as np
import numpy.random as npr
import pandas as pd
import gurobipy as gp
#  import gurobipy_pandas as gppd
from gurobi_ml import add_predictor_constr
import matplotlib.pyplot as plt
# import seaborn as sns
from matplotlib import cm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sys
sys.path.append('../src')
from gurobi_helpers import *

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

npr.seed(2023)

# Set base dimension

for n1 in [10, 5]:

    # Set uniform lower and upper bounds for the domain X
    
    L = -1
    U =  1
    
    # Set sample size
    
    N = 1000
    
    # Setup data structures for errors
    
    ove_box    = np.array([])
    ove_ch     = np.array([])
    ove_chplus = np.array([])
    ove_chp05  = np.array([])
    ove_chp1   = np.array([])
    ove_if     = np.array([])
    
    ose_box    = np.array([])
    ose_ch     = np.array([])
    ose_chplus = np.array([])
    ose_chp05  = np.array([])
    ose_chp1   = np.array([])
    ose_if     = np.array([])
    
    fve_box    = np.array([])
    fve_ch     = np.array([])
    fve_chplus = np.array([])
    fve_chp05  = np.array([])
    fve_chp1   = np.array([])
    fve_if     = np.array([])
    
    fe_box    = np.array([])
    fe_ch     = np.array([])
    fe_chplus = np.array([])
    fe_chp05  = np.array([])
    fe_chp1   = np.array([])
    fe_if     = np.array([])
    
    # Initialize optimization environment
    
    env = gp.Env()
    
    # Loop over iterations
    
    for iter in range(100):
    
        if iter % 1 == 0:
            print('iter = ' + str(iter))
    
        # Generate random linear objective; quadratic part is currently 0
    
        c = npr.normal(0, 1, [n1, 1])
        c = c / np.linalg.norm(c)
    
        # Determine opt sol and val
    
        x_star = -c
        v_star = -1
    
        # Generate samples D_N inside and outside of unit ball,
        # i.e., not just feasible samples
    
        #D_N = npr.uniform(L, U, (N, n1))
        D_N = npr.normal(0, 1, (N, n1))
        for j in range(np.shape(D_N)[0]):
            vec = D_N[j, ]
            vec = vec.flatten()
            vec = vec / np.linalg.norm(vec)
            vec = (0.5 + npr.uniform(0, 1)) * vec
            D_N[j, ] = vec
        
        # Evaluate function h on D_N (currently with "5%" noise)
        
        h_D_N = np.linalg.norm(D_N, axis = 1)
        h_D_N = h_D_N + 0.05 * npr.normal(0, 1, (N))
        # print("Reset the noise")
        
        # Learn function h_hat
        
        hidden_size = 30
        h_hat = MLPRegressor( \
            hidden_layer_sizes = (hidden_size, hidden_size), max_iter = 1000 \
        )
        h_hat = h_hat.fit(D_N, h_D_N)
    
        # Initialize optimization model
    
        m = gp.Model("trs", env = env)
        m.Params.LogToConsole = 0
        m.Params.NonConvex = 2
        
        # Setup and solve model with Box validity domain
        
        L_tmp = np.min(D_N, axis = 0)
        U_tmp = np.max(D_N, axis = 0)
        x = m.addMVar(shape = (n1), name = 'x', lb = L_tmp, ub = U_tmp)
        y = m.addVar(name = 'y', ub = 1)
        add_predictor_constr(m, h_hat, x, y)
        m.setMObjective(None, c.flatten(), 0, None, None, x, gp.GRB.MINIMIZE)
        m.optimize()
    
        # Calculate and save errors
        
        x_hat_box = x.x
        v_hat_box = m.getObjective().getValue()
        y_hat_box = y.x
        
        opt_val_err_box = np.abs(v_star - v_hat_box)
        opt_sol_err_box = np.linalg.norm(x_star.flatten() - x_hat_box)
        fun_val_err_box = np.abs(y_hat_box - np.linalg.norm(x_hat_box))
        feasibi_err_box = np.max([0, np.linalg.norm(x_hat_box) - 1])
        
        # Add CH validity domain and solve
        
        u = m.addMVar(shape = (N), name = 'u', lb = 0, ub = 1)
        s = m.addMVar(shape=(n1+2), name='s', lb=-gp.GRB.INFINITY)
        m.addConstr(u.sum() == 1)
        m.addConstrs(x[j] == (u @ D_N)[j] + s[j] for j in range(n1))
        epsilon = m.addConstr(s @ s <= (0*np.sqrt(n1))**2)
        m.optimize()
        
        # Calculate and save errors
        
        x_hat_ch = x.x
        v_hat_ch = m.getObjective().getValue()
        y_hat_ch = y.x
        
        opt_val_err_ch = np.abs(v_star - v_hat_ch)
        opt_sol_err_ch = np.linalg.norm(x_star.flatten() - x_hat_ch)
        fun_val_err_ch = np.abs(y_hat_ch - np.linalg.norm(x_hat_ch))
        feasibi_err_ch = np.max([0, np.linalg.norm(x_hat_ch) - 1])
    
        # Add CH+ validity domain. We limit the data set to just those,
        # which are feasible as suggested by theory
    
        indices_of_infeasible = np.where(h_D_N > 1.0)[0]
        if np.shape(indices_of_infeasible)[0] >= N - 4:
            print('Uh oh')
        
        obj = m.getObjective()
        f_D_N = np.dot(D_N, c).flatten()
        m.addConstr(obj == (u @ f_D_N) + s[-2])
        m.addConstr(y == (u @ h_D_N) + s[-1])
        m.addConstrs(u[j] == 0 for j in indices_of_infeasible)
        m.optimize()
        
        # Calculate and save errors
        
        x_hat_chplus = x.x
        v_hat_chplus = m.getObjective().getValue()
        y_hat_chplus = y.x
        
        opt_val_err_chplus = np.abs(v_star - v_hat_chplus)
        opt_sol_err_chplus = np.linalg.norm(x_star.flatten() - x_hat_chplus)
        fun_val_err_chplus = np.abs(y_hat_chplus - np.linalg.norm(x_hat_chplus))
        feasibi_err_chplus = np.max([0, np.linalg.norm(x_hat_chplus) - 1])
        
        # 0.05-CH+
        
        m.remove(epsilon)
        epsilon = m.addConstr(s @ s <= (0.05*np.sqrt(n1))**2)
        m.optimize()
        
        # Calculate and save errors
        
        x_hat_chp05 = x.x
        v_hat_chp05 = m.getObjective().getValue()
        y_hat_chp05 = y.x
        
        opt_val_err_chp05 = np.abs(v_star - v_hat_chp05)
        opt_sol_err_chp05 = np.linalg.norm(x_star.flatten() - x_hat_chp05)
        fun_val_err_chp05 = np.abs(y_hat_chp05 - np.linalg.norm(x_hat_chp05))
        feasibi_err_chp05 = np.max([0, np.linalg.norm(x_hat_chp05) - 1])
        
        # 0.1-CH+
        
        m.remove(epsilon)
        epsilon = m.addConstr(s @ s <= (0.1*np.sqrt(n1))**2)
        m.optimize()
        
        # Calculate and save errors
        
        x_hat_chp1 = x.x
        v_hat_chp1 = m.getObjective().getValue()
        y_hat_chp1 = y.x
        
        opt_val_err_chp1 = np.abs(v_star - v_hat_chp1)
        opt_sol_err_chp1 = np.linalg.norm(x_star.flatten() - x_hat_chp1)
        fun_val_err_chp1 = np.abs(y_hat_chp1 - np.linalg.norm(x_hat_chp1))
        feasibi_err_chp1 = np.max([0, np.linalg.norm(x_hat_chp1) - 1])

        do_isofor = False

        if do_isofor:
        
            # isofor
            # Optimize with isolation tree constraint on x
                
            m = gp.Model(env=env)
            m.Params.LogToConsole = 0
            m.params.NonConvex = 2
        
            x = m.addMVar(shape = (n1), name = 'x', lb = L, ub = U)
            y = m.addVar(name = 'y', ub = 1)
            add_predictor_constr(m, h_hat, x, y)
            m.setMObjective(None, c.flatten(), 0, None, None, x, gp.GRB.MINIMIZE)
        
            scaler = MinMaxScaler()
            D_scaled = scaler.fit_transform(D_N)
            x_scaled_var = m.addMVar(shape = (n1), name = 'x_scaled', ub=1)
            m.addConstr(x_scaled_var == (x-scaler.data_min_)/(scaler.data_max_-scaler.data_min_))
            add_isofor_constr(m, X=D_scaled, xx=x_scaled_var, d=5)
               
            m.optimize()
            
            # Calculate and save errors
            
            x_hat_if = x.x
            v_hat_if = m.getObjective().getValue()
            y_hat_if = y.x
            
            opt_val_err_if = np.abs(v_star - v_hat_if)
            opt_sol_err_if = np.linalg.norm(x_star.flatten() - x_hat_if)
            fun_val_err_if = np.abs(y_hat_if - np.linalg.norm(x_hat_if))
            feasibi_err_if = np.max([0, np.linalg.norm(x_hat_if) - 1])
        
        # Append errors to global data structures
        
        ove_box    = np.append(ove_box   , opt_val_err_box   )
        ove_ch     = np.append(ove_ch    , opt_val_err_ch    )
        ove_chplus = np.append(ove_chplus, opt_val_err_chplus)
        ove_chp05  = np.append(ove_chp05 , opt_val_err_chp05)
        ove_chp1   = np.append(ove_chp1  , opt_val_err_chp1)
        if do_isofor:
            ove_if     = np.append(ove_if    , opt_val_err_if)
    
        ose_box    = np.append(ose_box   , opt_sol_err_box   )
        ose_ch     = np.append(ose_ch    , opt_sol_err_ch    )
        ose_chplus = np.append(ose_chplus, opt_sol_err_chplus)
        ose_chp05  = np.append(ose_chp05 , opt_sol_err_chp05)
        ose_chp1   = np.append(ose_chp1  , opt_sol_err_chp1)
        if do_isofor:
            ose_if     = np.append(ose_if    , opt_sol_err_if    )
    
        fve_box    = np.append(fve_box   , fun_val_err_box   )
        fve_ch     = np.append(fve_ch    , fun_val_err_ch    )
        fve_chplus = np.append(fve_chplus, fun_val_err_chplus)
        fve_chp05  = np.append(fve_chp05 , fun_val_err_chp05)
        fve_chp1   = np.append(fve_chp1  , fun_val_err_chp1)
        if do_isofor:
            fve_if     = np.append(fve_if    , fun_val_err_if    )
    
        fe_box    = np.append(fe_box   ,  feasibi_err_box   )
        fe_ch     = np.append(fe_ch    ,  feasibi_err_ch    )
        fe_chplus = np.append(fe_chplus,  feasibi_err_chplus)
        fe_chp05  = np.append(fe_chp05 ,  feasibi_err_chp05)
        fe_chp1   = np.append(fe_chp1  ,  feasibi_err_chp1)
        if do_isofor:
            fe_if     = np.append(fe_if    ,  feasibi_err_if    )
    
    # Print final results
    
    if do_isofor:
        med_fve = [np.median(fve_box), np.median(fve_ch), np.median(fve_if), np.median(fve_chplus), np.median(fve_chp05), np.median(fve_chp1)]
        med_ose = [np.median(ose_box), np.median(ose_ch), np.median(ose_if), np.median(ose_chplus), np.median(ose_chp05), np.median(ose_chp1)]
        med_ove = [np.median(ove_box), np.median(ove_ch), np.median(ove_if), np.median(ove_chplus), np.median(ove_chp05), np.median(ove_chp1)]
        med_fe =  [np.median(fe_box) , np.median(fe_ch) , np.median(fe_if) , np.median(fe_chplus) , np.median(fe_chp05) , np.median(fe_chp1) ]
    else:
        med_fve = [np.median(fve_box), np.median(fve_ch), np.median(fve_chplus), np.median(fve_chp05), np.median(fve_chp1)]
        med_ose = [np.median(ose_box), np.median(ose_ch), np.median(ose_chplus), np.median(ose_chp05), np.median(ose_chp1)]
        med_ove = [np.median(ove_box), np.median(ove_ch), np.median(ove_chplus), np.median(ove_chp05), np.median(ove_chp1)]
        med_fe =  [np.median(fe_box) , np.median(fe_ch) , np.median(fe_chplus) , np.median(fe_chp05) , np.median(fe_chp1) ]
    
    tmp1 = med_fve / med_fve[0]
    tmp1 = np.array(tmp1)[:, np.newaxis]
    
    tmp2 = med_ove / med_ove[0]
    tmp2 = np.array(tmp2)[:, np.newaxis]
    
    tmp3 = med_ose / med_ose[0]
    tmp3 = np.array(tmp3)[:, np.newaxis]
    
    tmp4 = med_fe / med_fe[0]
    tmp4 = np.array(tmp4)[:, np.newaxis]
    
    tmp = np.hstack((tmp1, tmp2, tmp3, tmp4))

    print("Results for n1 = " + str(n1))
    
    print(pd.DataFrame(tmp).to_latex(index = False))

# %%
