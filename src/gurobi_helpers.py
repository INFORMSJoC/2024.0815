from imports import *

def add_svm_constr(model, svm_kernel, X, xx, y=None, yy=None, diam_y=None, threshold=None):
    if diam_y is not None:
        kernel = lambda X,Y=None,gamma=None: svm_kernel(X,Y,gamma, diam_y=diam_y)
    else:
        kernel = svm_kernel
    ocsvm = OneClassSVM(kernel=kernel, nu=0.1)
    if y is None and yy is None:
        ocsvm.fit(X)
        t = ocsvm.decision_function(X).min() #thresthold
    else:
        ocsvm.fit(np.hstack((X, y)))
        t = ocsvm.decision_function(np.hstack((X, y))).min() #threshold

    if threshold is not None:
        t = threshold
    dual_coef = ocsvm.dual_coef_
    intercept = ocsvm.intercept_
    support_vectors = X[ocsvm.support_, :] if y is None else np.hstack((X, y))[ocsvm.support_, :]
    n_support = np.size(ocsvm.support_)
    
    
    
    if y is None:
        diff = model.addMVar(shape=(n_support, X.shape[1]), vtype=GRB.CONTINUOUS, lb=-1, ub=1)
        distance = model.addMVar(shape=n_support, vtype=GRB.CONTINUOUS)
        kernel = model.addMVar(shape=n_support, vtype=GRB.CONTINUOUS)
        
        model.addConstr(diff == xx-support_vectors)
        model.addConstrs(distance[i] == gp.norm(diff[i,:], 1.0) for i in range(n_support))
        model.addConstr(kernel == X.shape[1] - distance)
        model.addConstr(dual_coef @ kernel + intercept >= t)
    
    else:
        diff = model.addMVar(shape=(n_support, X.shape[1]+1), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
        distance = model.addMVar(shape=n_support, vtype=GRB.CONTINUOUS)
        kernel = model.addMVar(shape=n_support, vtype=GRB.CONTINUOUS)
        
        model.addConstr(diff[:,:X.shape[1]] == xx-support_vectors[:,:X.shape[1]])
        model.addConstrs(diff[i,-1] == yy-support_vectors[i,-1] for i in range(n_support))
        model.addConstrs(distance[i] == gp.norm(diff[i,:], 1.0) for i in range(n_support))
        model.addConstr(kernel == X.shape[1]+diam_y - distance)
        model.addConstr(dual_coef @ kernel + intercept >= t)

def add_ae_constr(model, X, xx, y=None, yy=None, threshold=None):
    if y is None:
        inputs = torch.from_numpy(X).float()
    else:
        inputs = torch.from_numpy(np.hstack((X,y))).float()
        
    autoencoder = nn.Sequential(
        nn.Linear(inputs.shape[1], 2),
        nn.ReLU(True),
        nn.Linear(2, inputs.shape[1]),
        )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        
    t = torch.max(torch.norm(autoencoder(inputs) - inputs, p=2, dim=1))
    if threshold is not None:
        t = threshold
    
    recon = model.addMVar(shape=inputs.shape[1], vtype=GRB.CONTINUOUS,
                          lb=-GRB.INFINITY)
    if y is None:
        add_sequential_constr(model, autoencoder, xx, recon)
    else:
        combined = model.addMVar(inputs.shape[1], vtype=gp.GRB.CONTINUOUS, name="combined_mvar")
        model.addConstr(combined[:X.shape[1]] == xx, "mvar_constraint")
        model.addConstr(combined[X.shape[1]] == yy, "var_constraint")
        add_sequential_constr(model, autoencoder, combined, recon)
    diff = model.addMVar(shape=inputs.shape[1], vtype=GRB.CONTINUOUS,
                         lb=-GRB.INFINITY)
    recon_error = model.addVar(vtype=GRB.CONTINUOUS)
    if y is None:
        model.addConstr(diff == recon - xx)
    else:
        model.addConstr(diff == recon - combined)
    model.addConstr(recon_error == gp.norm(diff, 2.0))
    model.addConstr(recon_error <= t)
        

def get_leaf_bounds(node_id=0, lower_bounds=None, upper_bounds=None, bounds_matrix=None, tree=None):
    n_features = tree.n_features
    if bounds_matrix is None:
        bounds_matrix = np.empty((0, 2 * n_features), dtype=float)
    if lower_bounds is None:
        lower_bounds = [0] * n_features
    if upper_bounds is None:
        upper_bounds = [1] * n_features

    # If it's a leaf node, append the bounds to the matrix
    if tree.children_left[node_id] == tree.children_right[node_id]:
        bounds_row = np.concatenate([lower_bounds, upper_bounds])
        bounds_matrix = np.vstack([bounds_matrix, bounds_row])
    else:
        # Traverse left child
        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]

        left_lower_bounds = lower_bounds.copy()
        left_upper_bounds = upper_bounds.copy()
        left_upper_bounds[feature] = min(left_upper_bounds[feature], threshold)
        bounds_matrix, _ = get_leaf_bounds(tree.children_left[node_id], left_lower_bounds, left_upper_bounds, bounds_matrix, tree)

        # Traverse right child
        right_lower_bounds = lower_bounds.copy()
        right_upper_bounds = upper_bounds.copy()
        right_lower_bounds[feature] = max(right_lower_bounds[feature], threshold)
        bounds_matrix, _ = get_leaf_bounds(tree.children_right[node_id], right_lower_bounds, right_upper_bounds, bounds_matrix, tree)

    return bounds_matrix, tree

def find_leaf_depths(node_id=0, depth=0, leaf_depths=None, tree=None):
    if leaf_depths is None:
        leaf_depths = np.array([], dtype=int)

    # If it's a leaf node, record the depth
    if tree.children_left[node_id] == tree.children_right[node_id]:
        leaf_depths = np.append(leaf_depths, depth)
    else:
        # Recursively traverse left and right children
        leaf_depths = find_leaf_depths(tree.children_left[node_id], depth + 1, leaf_depths, tree)
        leaf_depths = find_leaf_depths(tree.children_right[node_id], depth + 1, leaf_depths, tree)

    return leaf_depths

def add_isofor_constr(model, X, xx, d=5):
    # train IF model
    clf = IsolationForest().fit(X) # Shi et al said, "all the hyperparameters setting as default"
    
    tree_dict = {}
    z_LB = model.addMVar(shape=X.shape[1], ub=1)
    z_UB = model.addMVar(shape=X.shape[1], ub=1)
    
    n_features = X.shape[1]
    
    # depth constraint
    #d = 5
    
    for estimator in clf.estimators_:
        tree = estimator.tree_
        bounds_matrix = get_leaf_bounds(tree=tree)[0]
        leaf_depths = find_leaf_depths(tree=tree)
        n_leaf = len(leaf_depths)
        tree_decision = model.addMVar(shape=n_leaf, vtype=GRB.BINARY)
        
        # 7(a) only one leaf is chosen per tree
        model.addConstr(tree_decision.sum() == 1)
        
        # 7(b) set tree_decision to 0 if the corresponding leaf node has a depth < d
        for i in range(n_leaf):
            if leaf_depths[i] <= d:
                model.addConstr(tree_decision[i] == 0)
        
        #7(c) lower bound & 7(d) upper bound
        for i in range(n_features):
            lb = bounds_matrix[:,i]
            lb_value = gp.quicksum(lb[j] * tree_decision[j] for j in range(n_leaf))
            model.addConstr(lb_value <= z_LB[i])
            ub = bounds_matrix[:,i+n_features]
            ub_value = 1 - gp.quicksum((1-ub[j]) * tree_decision[j] for j in range(n_leaf))
            model.addConstr(ub_value >= z_UB[i])
        
        #save tree decision variables for future reference
        tree_dict[str(estimator)] = tree_decision
    
    #7(e) bounds for xx
    epsilon = 0
    model.addConstr(xx >= z_LB)
    model.addConstr(xx <= z_UB - epsilon)
        

