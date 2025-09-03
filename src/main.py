from imports import *
from functions import *
from other import *
from options import *
from gurobi_helpers import *

def numerical_test(func_id, noise_std, distribution, n_sample, my_seed, mod, 
                   v_dom, learned_mod, X, feature_scaler, label_scaler,
                   diam_y, X_scaled, y_scaled, diam_y_scaled, obj_func,
                   global_min):

    # Print the optimal solution and objective value the variables used
    # in this function are global variables from main.py

    def scale_print():

        tmp = xx.x.reshape(1,-1)
        x_inv_sc = feature_scaler.inverse_transform(tmp)[0,:]

        tmp = np.array(model.objVal).reshape(1,-1)
        y_inv_sc = label_scaler.inverse_transform(tmp)[0,0]

        true_val_opt = obj_func(*x_inv_sc)

        error = np.abs(y_inv_sc - true_val_opt)

        dist_global_min = np.linalg.norm(x_inv_sc - global_min)
        
        optval_error = np.abs(y_inv_sc - obj_func(*global_min))
        
        setup_time = setup_end - setup_start
        opt_time = opt_end - setup_end

        return [y_inv_sc, true_val_opt, error, dist_global_min, optval_error,
                setup_time, opt_time]

    # Just return best sample
    
    setup_start = time.time() # time the setup time and optimization time

    if v_dom == 'bestsample':
        model = gp.Model()
        model.Params.LogToConsole = 0
        model.params.NonConvex = 2
        model.params.MIPGap = 1.0e-10
        model.params.FeasibilityTol = 1.0e-9
        model.params.IntFeasTol = 1.0e-9
        model.params.OptimalityTol = 1.0e-9
        xx = model.addMVar(shape=X.shape[1], vtype=GRB.CONTINUOUS, ub=1, \
            name="xx")
        yy = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="yy")
        model.setObjective(yy, GRB.MINIMIZE)
        t = model.addMVar(shape=n_sample, vtype=GRB.CONTINUOUS, ub=1)
        model.addConstr(t.sum() == 1)
        model.addConstr(X_scaled.T @ t == xx)
        model.addConstr(y_scaled.T @ t == yy)
        
        setup_end = time.time()
        model.optimize()
        opt_end = time.time()
        
        
        if model.status == GRB.OPTIMAL:
            obj_to_return = scale_print()
        else:
            print("No solution found (func = " + str(func_id) + ", v = bestsample).")
       
    # Optimize with no validity domain
    
    if v_dom in ['box', 'ch', 'chplus', 'chp.05', 'chp.1']:

        model = gp.Model()
        model.Params.LogToConsole = 0
        model.params.NonConvex = 2
        model.params.MIPGap = 1.0e-10
        model.params.FeasibilityTol = 1.0e-9
        model.params.IntFeasTol = 1.0e-9
        model.params.OptimalityTol = 1.0e-9
        
        # Create the decision variables
        xx = model.addMVar(shape=X.shape[1], vtype=GRB.CONTINUOUS, ub=1, \
            name="xx")
        yy = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="yy")
        
        # Set the objective function
        model.setObjective(yy, GRB.MINIMIZE)
       
        if mod == 'net' or mod == 'net_grid':
            add_mlp_regressor_constr(model, learned_mod, xx, yy)
        elif mod == 'forest':
            add_random_forest_regressor_constr(model, learned_mod, xx, yy)
        elif mod == 'gb':
            add_gradient_boosting_regressor_constr(model, learned_mod, xx, yy)
    
    if v_dom == 'box':

        setup_end = time.time()
        model.optimize()
        opt_end = time.time()
    
        if model.status == GRB.OPTIMAL:
            obj_to_return = scale_print()
        else:
            print("No soln found (func = " + str(func_id) + "), v = box).")
    
    #######################################################################

    if v_dom in ['ch', 'chplus']:

        # Create the decision variables
        t = model.addMVar(shape=n_sample, vtype=GRB.CONTINUOUS, ub=1)

        model.addConstr(t.sum() == 1)
        model.addConstr(X_scaled.T @ t == xx)

    if v_dom == 'ch':
    
        # Optimize with CH constraint on x
        
        setup_end = time.time()
        model.optimize()
        opt_end = time.time()

        # Print the optimal solution and objective value
        if model.status == GRB.OPTIMAL:
            obj_to_return = scale_print()
        else:
            print("No solution found (func = " + str(func_id) + ", v = ch).")
        
        ###################################################################
        
    if v_dom == 'chplus':
        # Optimize with CH constraint on (x,y)
        
        model.addConstr(y_scaled.T @ t == yy)
        
        setup_end = time.time()
        model.optimize()
        opt_end = time.time()

        # Print the optimal solution and objective value
        if model.status == GRB.OPTIMAL:
            obj_to_return = scale_print()
        else:
            print("No solution found (func = " + str(func_id) + ", v = chplus).")

    #######################################################################

    if v_dom == 'chp.05':

        # Create the decision variables
        t = model.addMVar(shape=n_sample, vtype=GRB.CONTINUOUS, ub=1)
        s = model.addMVar(shape=X.shape[1] + 1, vtype=GRB.CONTINUOUS, lb = -GRB.INFINITY)

        model.addConstr(t.sum() == 1)
        model.addConstr(X_scaled.T @ t == xx + s[:-1])
        model.addConstr(y_scaled.T @ t == yy + s[-1])
        model.addConstr(s @ s <= (0.05*np.sqrt(X.shape[1]))**2)

        setup_end = time.time()
        model.optimize()
        opt_end = time.time()

        # Print the optimal solution and objective value
        if model.status == GRB.OPTIMAL:
            obj_to_return = scale_print()
        else:
            print("No solution found (func = " + str(func_id) + ", v = chp.05).")
        
    #######################################################################

    if v_dom == 'chp.1':

        # Create the decision variables
        t = model.addMVar(shape=n_sample, vtype=GRB.CONTINUOUS, ub=1)
        s = model.addMVar(shape=X.shape[1] + 1, vtype=GRB.CONTINUOUS, lb = -GRB.INFINITY)

        model.addConstr(t.sum() == 1)
        model.addConstr(X_scaled.T @ t == xx + s[:-1])
        model.addConstr(y_scaled.T @ t == yy + s[-1])
        model.addConstr(s @ s <= (0.10*np.sqrt(X.shape[1]))**2)

        setup_end = time.time()
        model.optimize()
        opt_end = time.time()

        # Print the optimal solution and objective value
        if model.status == GRB.OPTIMAL:
            obj_to_return = scale_print()
        else:
            print("No solution found (func = " + str(func_id) + ", v = chp.1).")

    #######################################################################

    if v_dom == 'svm' or v_dom == 'svm0':

        # Optimize with PWL OCSVM constraint on x

        model = gp.Model()
        model.Params.LogToConsole = 0
        model.params.NonConvex = 2

        # Create the decision variables
        xx = model.addMVar(shape=X.shape[1], vtype=GRB.CONTINUOUS,
                           ub=1, name="xx")
        yy = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="yy")

        # Set the objective function
        model.setObjective(yy, GRB.MINIMIZE)

        if mod == 'net' or mod == 'net_grid':
            add_mlp_regressor_constr(model, learned_mod, xx, yy)
        elif mod == 'forest':
            add_random_forest_regressor_constr(model, learned_mod, xx, yy)
        elif mod == 'gb':
            add_gradient_boosting_regressor_constr(model, learned_mod, xx, yy)

        if v_dom == 'svm':
            add_svm_constr(model, PWLKernel, X=X_scaled, xx=xx)
        elif v_dom == 'svm0':
            add_svm_constr(model, PWLKernel, X=X_scaled, xx=xx, threshold = 0.0)

        setup_end = time.time()
        model.optimize()
        opt_end = time.time()

        # Print the optimal solution and objective value
        if model.status == GRB.OPTIMAL:
            obj_to_return = scale_print()
        else:
            print("No solution found (func = " + str(func_id) + ", v = " + v_dom + ").")
            
        ###################################################################
        
    if v_dom == 'svmplus' or v_dom == 'svm0plus':
        # Optimize with PWL OCSVM constraint on (x,y)

        model = gp.Model()
        model.Params.LogToConsole = 0
        model.params.NonConvex = 2

        # Create the decision variables
        xx = model.addMVar(shape=X.shape[1], vtype=GRB.CONTINUOUS,
                           ub=1, name="xx")
        yy = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="yy")

        # Set the objective function
        model.setObjective(yy, GRB.MINIMIZE)

        if mod == 'net' or mod == 'net_grid':
            add_mlp_regressor_constr(model, learned_mod, xx, yy)
        elif mod == 'forest':
            add_random_forest_regressor_constr(model, learned_mod, xx, yy)
        elif mod == 'gb':
            add_gradient_boosting_regressor_constr(model, learned_mod, xx, yy)

        if v_dom == 'svmplus':
            add_svm_constr(model, PWLKernel, X=X_scaled, xx=xx,
                           y=y_scaled, yy=yy, diam_y=diam_y_scaled)
        elif v_dom == 'svm0':
            add_svm_constr(model, PWLKernel, X=X_scaled, xx=xx,
                           y=y_scaled, yy=yy, diam_y=diam_y_scaled, threshold = 0.0)

        setup_end = time.time()
        model.optimize()
        opt_end = time.time()

        # Print the optimal solution and objective value
        if model.status == GRB.OPTIMAL:
            obj_to_return = scale_print()
        else:
            print("No solution found (func = " + str(func_id) + ", v = " + v_dom + ").")
    
    #######################################################################
    
    if v_dom == 'AE':

        # Optimize with autoencoder constraint on x

        model = gp.Model()
        model.Params.LogToConsole = 0
        model.params.NonConvex = 2

        # Create the decision variables
        xx = model.addMVar(shape=X.shape[1], vtype=GRB.CONTINUOUS,
                           ub=1, name="xx")
        yy = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="yy")

        # Set the objective function
        model.setObjective(yy, GRB.MINIMIZE)

        if mod == 'net' or mod == 'net_grid':
            add_mlp_regressor_constr(model, learned_mod, xx, yy)
        elif mod == 'forest':
            add_random_forest_regressor_constr(model, learned_mod, xx, yy)
        elif mod == 'gb':
            add_gradient_boosting_regressor_constr(model, learned_mod, xx, yy)

        add_ae_constr(model, X=X_scaled, xx=xx)

        setup_end = time.time()
        model.optimize()
        opt_end = time.time()

        # Print the optimal solution and objective value
        if model.status == GRB.OPTIMAL:
            obj_to_return = scale_print()
        else:
            print("No solution found (func = " + str(func_id) + ", v = AE).")
        
        #######################################################################
        
    if v_dom == 'AE+':
        # Optimize with autoencoder constraint on (x,y)

        model = gp.Model()
        model.Params.LogToConsole = 0
        model.params.NonConvex = 2

        # Create the decision variables
        xx = model.addMVar(shape=X.shape[1], vtype=GRB.CONTINUOUS,
                           ub=1, name="xx")
        yy = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="yy")

        # Set the objective function
        model.setObjective(yy, GRB.MINIMIZE)

        if mod == 'net' or mod == 'net_grid':
            add_mlp_regressor_constr(model, learned_mod, xx, yy)
        elif mod == 'forest':
            add_random_forest_regressor_constr(model, learned_mod, xx, yy)
        elif mod == 'gb':
            add_gradient_boosting_regressor_constr(model, learned_mod, xx, yy)

        add_ae_constr(model, X=X_scaled, xx=xx, y=y_scaled, yy=yy)

        setup_end = time.time()
        model.optimize()
        opt_end = time.time()

        # Print the optimal solution and objective value
        if model.status == GRB.OPTIMAL:
            obj_to_return = scale_print()
        else:
            print("No solution found (func = " + str(func_id) + ", v = AE+).")

        #######################################################################

    if v_dom == 'isofor':

        # Optimize with isolation tree constraint on x
        
        model = gp.Model()
        model.Params.LogToConsole = 0
        model.params.NonConvex = 2
        
        # Create the decision variables
        xx = model.addMVar(shape=X.shape[1], vtype=GRB.CONTINUOUS,
                           ub=1, name="xx")
        yy = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="yy")

        # Set the objective function
        model.setObjective(yy, GRB.MINIMIZE)

        if mod == 'net' or mod == 'net_grid':
            add_mlp_regressor_constr(model, learned_mod, xx, yy)
        elif mod == 'forest':
            add_random_forest_regressor_constr(model, learned_mod, xx, yy)
        elif mod == 'gb':
            add_gradient_boosting_regressor_constr(model, learned_mod, xx, yy)
        
        # depth threshould 5 if beal/peak, 6 otherwise
        d = 5 if func_id in [1, 2] else 6
        add_isofor_constr(model, X=X_scaled, xx=xx, d=d)

        setup_end = time.time()        
        model.optimize()
        opt_end = time.time()

        # Print the optimal solution and objective value
        if model.status == GRB.OPTIMAL:
            obj_to_return = scale_print()
        else:
            print("No solution found (func = " + str(func_id) + ", v = isofor).")

    return obj_to_return


def run_one(func_id, noise_std, distribution, n_sample, my_seed, mod, v_domains):

    # Setup columns

    columns = [ \
        'func', 'noise', 'sampling', 'sample_sz', 'seed', 'learning',
        'v_domain','modval_modsol', 'truval_modsol', 'funval_err',
        'optsol_err','optval_err', 'opt_setup_time', 'opt_opt_time',
        'ML_train_time', 'R2_score']

    # Get objective function details

    obj_func, mean, global_min, domain_l, domain_r, diam_x, global_min_val = \
        generate_function(func_id)

    # Set primary key and a simple variant of the primary key

    pk    = [func_id,           noise_std, distribution, n_sample, my_seed, mod]
    pkalt = [obj_func.__name__, noise_std, distribution, n_sample, my_seed, mod]

    # Create subfolder for csv and joblib files if not already exist

    if not os.path.exists('../results/csv'):
        os.makedirs('../results/csv')

    if not os.path.exists('../results/joblib'):
        os.makedirs('../results/joblib')

    # Create CSV filename

    pkcsv = '_'.join(['../results/csv/output'] + [str(s) for s in pk]) + '.csv'

    # Create joblib filename

    pkjoblib = '_'.join(['../results/joblib/model'] + [str(s) for s in pk]) + '.joblib'

    # Sample the training data

    np.random.seed(my_seed)
    X, y = generate_samples(distribution, obj_func, mean, domain_l, \
        domain_r, noise_std, n_sample)
    diam_y = y.max() - y.min()

    # Scale data

    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X)

    label_scaler = StandardScaler()
    y_scaled = label_scaler.fit_transform(y)
    diam_y_scaled = y_scaled.max() - y_scaled.min()

    # Train the function
    
    train_time = -1 # time of training, set to -1 if it's already trained
    r2 = -1 # r2 score, same

    try:
        learned_mod = joblib.load(pkjoblib)
    except FileNotFoundError:
        train_start = time.time()
        hidden_size = 50 if func_id == 9 else 30
        learned_mod, r2 = train_model(pk[-1], hidden_size, X_scaled, y_scaled)
        train_end = time.time()
        train_time = train_end - train_start
        joblib.dump(learned_mod, pkjoblib)

    # Plot the true and learned models (if do_plots == True)

    if do_plots and func_id in [1,2,3,7]:

        plot_both(obj_func, domain_l, domain_r,
                  learned_mod, feature_scaler, label_scaler, X, y)

    # Setup the CSV file

    try:
        local_df = pd.read_csv(pkcsv)
    except FileNotFoundError:
        local_df = pd.DataFrame(columns = columns)
    loc = len(local_df)

    # Loop over V domains

    for v_dom in sorted(set(v_domains) - set(local_df.v_domain.unique())):

        results = numerical_test(func_id, noise_std, distribution, n_sample,
                                 my_seed, mod, v_dom, learned_mod, X,
                                 feature_scaler, label_scaler, diam_y,
                                 X_scaled, y_scaled, diam_y_scaled,
                                 obj_func, global_min)
        
        local_df.loc[loc] = pkalt + [v_dom] + results + [train_time, r2]
        loc += 1

    # Write data frame to file

    local_df.to_csv(pkcsv, index = False)
