from imports import *

def PWLKernel(X, Y=None, gamma=None, diam_y=None):
     # If gamma is not provided, use the default value
    if gamma is None:
        gamma = 1.0 / X.shape[1]
        
    if Y is None:
        Y = X
    
    K = X.shape[1] - manhattan_distances(X, Y)
    
    if diam_y is not None:
        K += diam_y-1
        
    return K

def generate_samples(distribution, obj_func, mean, domain_l, domain_r, noise_std, n_sample):
    if distribution == 'uniform':
        #  samples_u = np.random.uniform( \
        #      low = domain_l, high = domain_r, \
        #      size= (np.max(sample_range), len(mean)))
        samples_u = np.random.uniform( \
            low = domain_l, high = domain_r, \
            size= (n_sample, len(mean)))
        X = samples_u[:n_sample, :]
    elif distribution == 'normal_at_min':
        dist_to_U = np.min(domain_r - mean)
        dist_to_L = np.min(mean - domain_l)
        cov = np.eye(len(mean)) * np.min([dist_to_U, dist_to_L]) / 6
        #  np.random.seed(my_seed)
        #  samples_n = np.random.multivariate_normal(mean, cov, np.max(sample_range))
        samples_n = np.random.multivariate_normal(mean, cov, n_sample)
        X = samples_n[:n_sample, :]
    #  np.random.seed(my_seed)
    #  samples_noise = np.random.rand(np.max(sample_range)).reshape(-1, 1)
    samples_noise = np.random.rand(n_sample).reshape(-1, 1)
    X_col = [X[:,i] for i in range(X.shape[1])]
    y = obj_func(*X_col).reshape(-1,1)
    noise = noise_std * samples_noise[:n_sample] * np.std(y)
    y = y + noise
    return X, y

def plot_both(obj_func, domain_l, domain_r, learned_mod, feature_scaler, label_scaler, X, y):

    l0 = np.min(X[:, 0])
    r0 = np.max(X[:, 0])
    l1 = np.min(X[:, 1])
    r1 = np.max(X[:, 1])

    u = torch.arange(l0, r0, (r0 - l0) / 100)
    v = torch.arange(l1, r1, (r1 - l1) / 100)
    #  u = torch.arange(domain_l[0], domain_r[0], 0.1)
    #  v = torch.arange(domain_l[1], domain_r[1], 0.1)

    u1, u2 = torch.meshgrid(u, v, indexing="ij")
    w = obj_func(u1, u2)

    fig, axs = plt.subplots(nrows = 1, ncols = 2, subplot_kw = {"projection": "3d"})

    surf0 = axs[0].plot_surface(u1, u2, w, cmap = cm.coolwarm, \
        linewidth = 0.01, antialiased = False)

    tmp = np.hstack((u.reshape(-1,1), v.reshape(-1,1)))
    uv_scaled = torch.tensor(feature_scaler.transform(tmp))
    u1_scaled, u2_scaled = torch.meshgrid(uv_scaled[:,0], \
        uv_scaled[:,1], indexing="ij")        
    XX = torch.cat([u1_scaled.ravel().reshape(-1, 1), \
        u2_scaled.ravel().reshape(-1, 1)], axis=1)
    tmp = learned_mod.predict(XX).reshape(u1.shape)
    surf1 = axs[1].plot_surface(u1, u2,
        label_scaler.inverse_transform(tmp),
        cmap = cm.coolwarm,
        linewidth = 0.01,
        antialiased = True, # Transparancey
    )

    #  fig.colorbar(surf0, shrink = 0.5, aspect = 5)

    plt.savefig('plot.png', dpi = 300, transparent = True, bbox_inches = 'tight', pad_inches = 0)
    plt.show()

def train_model(mod, hidden_size, X_scaled, y_scaled):
    if mod == 'net':
        learned_mod = MLPRegressor( \
            hidden_layer_sizes=(hidden_size, hidden_size), max_iter=1000)
    elif mod == 'forest':
        learned_mod = RandomForestRegressor(n_estimators=100, max_depth=5)
    elif mod == 'gb':
        learned_mod = GradientBoostingRegressor(n_estimators=100,
                                                learning_rate=0.1,
                                                max_depth=5)
    elif mod == 'net_grid':
        mlp = MLPRegressor(max_iter=1000)
        param_grid = {
            'hidden_layer_sizes': \
                [(i,j) for i in range(10,31,5) for j in range(10,31,5)]
        }
        grid_search = GridSearchCV(mlp, param_grid,
                                   cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_scaled, y_scaled.ravel())
        learned_mod = grid_search.best_estimator_
    else:
        print('Error: Unexpected mod')
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                        y_scaled.ravel(),
                                                        test_size=0.2,
                                                        random_state=42)
    learned_mod.fit(X_train, y_train)
    y_pred = learned_mod.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    learned_mod = learned_mod.fit(X_scaled, y_scaled.ravel())
        
    return learned_mod, r2


