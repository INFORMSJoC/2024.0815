#!/usr/bin/env python
# coding: utf-8

# # Avocado Pricing and Supply Using Mathematical Optimization
# 
# **Goal**: Develop a data-science and decision-making pipeline for pricing and distribution of avocados to maximize revenue.
# 
# To accomplish this, the notebook will walk trough three stages:
# 
# 1. A quick review of [Hass Avocado Board](https://hassavocadoboard.com/) (HAB) data
# 2. A prediction model for avocado demand as a function of price, region, year and seasonality.
# 3. An optimization model that sets the optimal price and supply quantity to maximize the net revenue while incorporating transportation and costs.
# 
# See also: [How Much Is Too Much? Avocado Pricing and Supply Using Mathematical Optimization](https://github.com/Gurobi/modeling-examples/tree/master/price_optimization)

# ## Load the Packages and Prepare the Dataset
# 
# The dataset from HAB contains sales data for the years 2019-2022. This data is augmented by a previous download from HAB available on [Kaggle](https://www.kaggle.com/datasets/timmate/avocado-prices-2020) with sales for the years 2015-2018.
# 
# One of the regions in the above data frame is `Total_US`, so we can create a list of regions, excluding the total, which can be used to subset the data now. It'll be used later in the example as well.

###############################################################################
# Questions to consider
###############################################################################

# If the optimization model has to choose peak=0 or peak=1, should I
# learn just wrt to the peak or off-peak data? Should I plot just the
# peak or off-peak data?

# Gurobi added non-trivial lower and upper bounds on x_r. Should I
# remove these bounds? Yes, I have done so on 2024-04-19. Does not seem
# to make any difference

# We currently set totaly supply B = 30. Is this a reasonable value?

# Should we do everything wrt (p,x,d), not just (p,d)?

# Should we show the x values in the paper? Do they carry much meaning?

# CH and CH^+ are taken wrt to all price data, regardless of peak or
# off-peak, right?

###############################################################################
# STAGE 0 --- License and packages
###############################################################################

# Set the parameters for the Gurobi license

# This is Sam's floating license

params = {
    "WLSACCESSID": '1c31e7ee-6025-4b2e-a5a0-1ac137f87660',
    "WLSSECRET": 'dd7feb58-13dd-4150-995d-4aff2fa9d0b8',
    "LICENSEID": 2439168
}

# Import all the packages that we need

import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# pip install gurobipy_pandas # this also installs gurobipy
# pip install gurobi-machinelearning
import gurobipy as gp
import gurobipy_pandas as gppd
from gurobi_ml import add_predictor_constr
from sklearn.ensemble import GradientBoostingRegressor
import sys
sys.path.append('../src')
from gurobi_helpers import *

###############################################################################
# STAGE 1 --- Data import and preparation
###############################################################################

# Get the data hosted by Gurobi

data_url = "https://raw.githubusercontent.com/Gurobi/modeling-examples/master/price_optimization/"
avocado = pd.read_csv(data_url+"HAB_data_2015to2022.csv")
avocado["date"] = pd.to_datetime(avocado["date"])
avocado = avocado.sort_values(by = "date")

# Specify the list of regions that we wish to keep. This will enable us
# to exclude Total_US

regions = [
    "Great_Lakes",
    "Midsouth",
    "Northeast",
    "Northern_New_England",
    "SouthCentral",
    "Southeast",
    "West",
    "Plains"
]

# Keep only the regions we want to keep

df = avocado[(avocado.region.isin(regions))] # & (avocado.peak==0)

# Drop the date column. We will just need the year column

df.drop(columns=['date']) #,'peak'

###############################################################################
# STAGE 2 --- Basic summary statistics and plot
###############################################################################

# Calculate some statistics for the paper. Would be good to document this

tmp = df
tmp['revenue'] = tmp['units_sold'] * tmp['price']
aggregated_tmp = tmp.groupby('date').sum()
aggregated_tmp['price'] = aggregated_tmp['price'] / 8
#  print(aggregated_tmp.sort_values(by='date'))
print(tmp[['units_sold', 'price', 'revenue']].mean())
print(aggregated_tmp[['units_sold', 'price', 'revenue']].mean())

tmp = df[df['peak'] == 0]
tmp['revenue'] = tmp['units_sold'] * tmp['price']
aggregated_tmp = tmp.groupby('date').sum()
aggregated_tmp['price'] = aggregated_tmp['price'] / 8
#  print(aggregated_tmp.sort_values(by='date'))
print(tmp[['units_sold', 'price', 'revenue']].mean())
print(aggregated_tmp[['units_sold', 'price', 'revenue']].mean())

# Create a scatterplot of all data. Will we include this in the paper?
# Not sure

plt.figure(figsize=(10, 6))
r_plt = sns.scatterplot(data=df, x='price', y='units_sold', hue='region')
r_plt.legend(fontsize = 12)
r_plt.set_xlabel('Price', fontsize = 16)
r_plt.set_ylabel('Units Sold', fontsize = 16)
#  plt.savefig('../results/figures/avocado_data_scatterplot.png', dpi = 300, bbox_inches = 'tight')
# plt.show()

###############################################################################
# STAGE 3 --- Learn demand model
###############################################################################

# ## Predict and Visualize the Sales
# 
# In the first instance of this example, further analysis was done on
# the input data along with a few visualizations. Here, we will go
# directly to the predicive model training, starting with a random split
# of the dataset into $70\%$ training and $30\%$ testing data.
#
# Note that the region is a categorical variable and we will transform
# that variable using Scikit Learn's `OneHotEncoder`. We also use a
# standard scaler for prices and year index, combining all of the ese
# with `Column Transformer` built using `make_column_transformer`.
# 
# The regression model is a pipeline consisting of the `Column Transformer` and the type of model we want to use for the regression. For comparison, we'll stick with a linear regression.
# 
# We can observe a good $R^2$ value in the test set. We will now train the fit to the full dataset.

X = df[["region", "price", "year", "peak"]]

y = df["units_sold"]
d_mean = np.mean(y)
d_std = np.std(y)

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size = 0.7, random_state = 1
)

feat_transform = make_column_transformer(
    (OneHotEncoder(drop="first"), ["region"]),
    (StandardScaler(), ["price", "year"]),
    ("passthrough", ["peak"]),
    verbose_feature_names_out=False,
    remainder='drop'
)

#  reg = make_pipeline(feat_transform, LinearRegression())
#  scores = cross_val_score(reg, X_train, y_train, cv=5)
#  #  print("%0.4f R^2 with a standard deviation of %0.4f" % (scores.mean(), scores.std()))
#  # Find model score on test data
#  reg.fit(X_train, y_train)
#  y_pred = reg.predict(X_test)
#  #  print(f"The R^2 value in the test set is {np.round(r2_score(y_test, y_pred),5)}")
#  reg.fit(X, y)
#  y_pred_full = reg.predict(X)
#  #  print(f"The R^2 value in the full dataset is {np.round(r2_score(y, y_pred_full),5)}")

reg = make_pipeline(feat_transform, GradientBoostingRegressor(n_estimators=100, max_leaf_nodes = 20,
                                              loss = 'absolute_error', random_state = 123))
scores = cross_val_score(reg, X_train, y_train, cv=5)
#  print("%0.4f R^2 with a standard deviation of %0.4f" % (scores.mean(), scores.std()))
# Fit to entire training data
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
#  print(f"The R^2 value in the test set is {np.round(r2_score(y_test, y_pred),5)}")
reg.fit(X, y)
y_pred_full = reg.predict(X)
#  print(f"The R^2 value in the full dataset is {np.round(r2_score(y, y_pred_full),5)}")

###############################################################################
# STAGE 4 --- Define parameters and data structures needed for optimization
###############################################################################

# Because we are optimizing for a specific year and a specific
# seasonality (peak vs off-peak), we need to specify the values of year
# and peak

year = 2023
peak_or_not = 0

# ## Optimize for Price and Supply of Avocados
# 
# Here is a quick review of notation for the formulation of the
# mathematical optimization model. The subscript $r$ will be used to
# denote each region.
#
# ### Input parameters
#
# - $d(p,r)$: predicted demand in region $r$ when the avocado price is $p$
# - $B$: available avocados to be distributed across the regions
# - $c_{waste}$: cost ($\$$) per wasted avocado
# - $c^r_{transport}$: cost ($\$$) of transporting a avocado to region $r$
# - $a_{min},a_{max}$: minimum and maximum price ($\$$) per avocado
# - $b^r_{min},b^r_{max}$: minimum and maximum number of avocados allocated to region $r$
# 
# The following code sets values for these parameters. Feel free to
# adjust these to see how the solution to the optimization model will
# change.
# 
# ### Create dataframe for the fixed features of the regression
# 
# We now start creating the input of the regression in the optimization
# models with the features that are fixed and use `gurobipy-pandas` that
# help to more easily create gurobipy models using pandas data.
#
# First, create a dataframe with the features that are fixed in our
# optimization problem. It is indexed by the regions (we want to use
# one regression to predict demand for all regions) and has the three
# columns corresponding to the fixed features:
# 
# * `year`
# * `peak` with the value of `peak_or_not`
# * `region` that repeats the names of the regions.
# 
# Let's display the dataframe to make sure it is correct.

# Setup the parameters of the model (not including the demand function
# itself)

B = 30.0        # total amount of avocado supply
c_waste = 0.1 # the cost ($) of wasting an avocado

# the cost of transporting an avocado
c_transport = pd.Series(
    {
        "Great_Lakes": 0.3,
        "Midsouth": 0.1,
        "Northeast": 0.4,
        "Northern_New_England": 0.5,
        "SouthCentral": 0.3,
        "Southeast": 0.2,
        "West": 0.2,
        "Plains": 0.2,
    }, name='transport_cost'
)
c_transport = c_transport.loc[regions]

a_min = 0.6  # minimum avocado price # Gurobi originally had 0
a_max = 2.0  # maximum avocado price

# Get the lower and upper bounds from the dataset for the price and the
# number of products to be stocked

data = pd.concat([c_transport,
                  df.groupby("region")["units_sold"].min().rename('min_delivery'),
                  df.groupby("region")["units_sold"].max().rename('max_delivery'),
                  df.groupby("region")["price"].max().rename('max_price'),
                                   df.groupby("region")["price"].min().rename('min_price')], axis=1)

feats = pd.DataFrame(
    data={
        "year": year,
        "peak": peak_or_not,
        "region": regions,
    },
    index=regions
)

# ### Decision Variables
# 
# Let us now define the decision variables. In our model, we want to
# store the price and number of avocados allocated to each region. We
# also want variables that track how many avocados are predicted to be
# sold and how many are predicted to be wasted. The following notation
# is used to model these decision variables.
# 
# - $p$ the price of an avocado ($\$$) in each region
# - $x$ the number of avocados supplied to each region
# - $s$ the predicted number of avocados sold in each region
# - $u$ the predicted number of avocados unsold (wasted) in each region
# - $d$ the predicted demand in each region
# 
# All those variables are created using gurobipy-pandas, with the
# function `gppd.add_vars` they are given the same index as the `data`
# dataframe.
# 
# ### Add the Supply Constraint
# 
# We now introduce the constraints. The first constraint is to make sure
# that the total number of avocados supplied is equal to $B$, which can
# be mathematically expressed as follows.
# 
# \begin{align*} \sum_{r} x_r &= B \end{align*}
# 
# ### Add Constraints That Define Sales Quantity
# 
# As a quick reminder, the sales quantity is the minimum of the
# allocated quantity and the predicted demand, i.e., $s_r = \min
# \{x_r,d_r(p_r)\}$ This relationship can be modeled by the following two
# constraints for each region $r$.
# 
# \begin{align*} s_r &\leq x_r  \\
# s_r &\leq d(p_r,r) \end{align*}
# 
# In this case, we use gurobipy-pandas `add_constrs` function, which is
# intuitive to use given the inequalities above.
# 
# ### Add the Wastage Constraints
# 
# Finally, we should define the predicted unsold number of avocados in
# ach region, given by the supplied quantity that is not predicted to be
# sold. We can express this mathematically for each region $r$.
# 
# \begin{align*} u_r &= x_r - s_r \end{align*}
# 
# ### Add the constraints to predict demand
#
# First, we create our full input for the predictor constraint. We
# concatenate the `p` variables and the fixed features. Remember that
# the predicted price is a function of region, year, and peak/off-peak
# season.
# 
# Now, we just call
# [add_predictor_constr](https://gurobi-machinelearning.readthedocs.io/en/stable/api/AbstractPredictorConstr.html#gurobi_ml.add_predictor_constr)
# to insert the constraints linking the features and the demand into the model `m`.
# 
# It is important that you keep the columns in the order above, otherwise you will see an error. The columns must be in the same order as the training data.
# 
# ### Set the Objective
# 
# The goal is to maximize the **net revenue**, which is the product of price and quantity, minus costs over all regions. This model assumes the purchase costs are fixed (since the amount $B$ is fixed) and are therefore not incorporated.
# 
# Using the defined decision variables, the objective can be written as follows.
# 
# \begin{align} \textrm{maximize} &  \sum_{r}  (p_r * s_r - c_{waste} * u_r -
# c^r_{transport} * x_r)& \end{align}
# 
# ### Fire Up the Solver
# 
# In our model, the objective is **quadratic** since we take the product of price
# and the predicted sales, both of which are variables. Maximizing a quadratic
# term is said to be **non-convex**, and we specify this by setting the value of
# the [Gurobi NonConvex
# parameter](https://www.gurobi.com/documentation/10.0/refman/nonconvex.html) to be
# $2$.
# 
# 
# The solver solved the optimization problem in less than a second. Let us now
# analyze the optimal solution by storing it in a Pandas dataframe.
# 
# We can also check the error in the estimate of the Gurobi solution for the regression model.

###############################################################################
# STAGE 5 --- Build and solve the optimization models
###############################################################################

#------------------------------------------------------------------------------
# Gurobi's default
#------------------------------------------------------------------------------

#env = gp.Env()
env = gp.Env(params = params) # To enable Sam's floating license

m = gp.Model("Avocado_Price_Allocation", env = env)
m.Params.NonConvex = 2
m.Params.LogToConsole = 0
m.Params.TimeLimit = 600

p = gppd.add_vars(m, data, name = "price", lb = a_min, ub = a_max)
#  x = gppd.add_vars(m, data, name = "x", lb = 'min_delivery', ub = 'max_delivery')
x = gppd.add_vars(m, data, name = "x", lb = 0)
s = gppd.add_vars(m, data, name = "s")
u = gppd.add_vars(m, data, name = "w")
d = gppd.add_vars(m, data, lb = -gp.GRB.INFINITY, name="demand")
d_scaled = gppd.add_vars(m, data, lb=-gp.GRB.INFINITY, name="scaled_demand")

m.setObjective((p * s).sum() - c_waste * u.sum() - (c_transport * x).sum(),
               gp.GRB.MAXIMIZE)

m.addConstr(x.sum() == B)
gppd.add_constrs(m, s, gp.GRB.LESS_EQUAL, x)
gppd.add_constrs(m, s, gp.GRB.LESS_EQUAL, d)
gppd.add_constrs(m, u, gp.GRB.EQUAL, x - s)

m_feats = pd.concat([feats, p], axis=1)[["region", "price", "year", "peak"]]

m.update()
#  print(m_feats)
#  print("\n\n\n\n\n\n")

pred_constr = add_predictor_constr(m, reg, m_feats, d)

m.optimize()

solution = pd.DataFrame(index=regions)

solution["Price"] = p.gppd.X
solution["Historical_Max"] = data.max_price
solution["Allocated"] = x.gppd.X
solution["Sold"] = s.gppd.X
solution["Wasted"] = u.gppd.X
solution["Pred_demand"] = d.gppd.X

soln_gur = solution

opt_revenue = m.ObjVal
print("\nThe optimal profit: $%f million\n" % opt_revenue)
print(solution.round(3))

#------------------------------------------------------------------------------
# Box
#------------------------------------------------------------------------------

gppd.add_constrs(m, p, gp.GRB.LESS_EQUAL, data['max_price'].values)
gppd.add_constrs(m, p, gp.GRB.GREATER_EQUAL, data['min_price'].values)

m.optimize()

solution = pd.DataFrame(index=regions)

solution["Price"] = p.gppd.X
solution["Historical_Max"] = data.max_price
solution["Allocated"] = x.gppd.X
solution["Sold"] = s.gppd.X
solution["Wasted"] = u.gppd.X
solution["Pred_demand"] = d.gppd.X

soln_box = solution

opt_revenue = m.ObjVal
print("\nThe optimal profit: $%f million\n" % opt_revenue)
print(solution.round(3))

#  print_graphs_primary()

#------------------------------------------------------------------------------
# CH
#------------------------------------------------------------------------------

# Recode the 378 and 8! They should not be hard-coded

mu = m.addMVar(shape=(378), name='mu', lb = 0, ub = 1)
enlarge_s = m.addMVar(shape=(8+8), lb=-gp.GRB.INFINITY)
enlarged = m.addConstr(enlarge_s @ enlarge_s <= (0*np.sqrt(8+8))**2)
diff_p = gppd.add_vars(m, data, name = "diff_p", lb = -np.inf, ub = np.inf)
m.addConstr(mu.sum() == 1.0)
X_pivot = df.pivot(index='date', columns='region', values='price').loc[:,regions]
m.addConstrs(diff_p.values[i] == p.values[i] - (mu@X_pivot.values)[i] - enlarge_s[i] for i in range(8))
eps = 0.0 # Note that eps is currently zero! 
m.addConstr((diff_p * diff_p).sum() <= eps, name = "diff_p_constraint")
m.optimize()

solution = pd.DataFrame(index=regions)
solution["Price"] = p.gppd.X
solution["Historical_Max"] = data.max_price
solution["Allocated"] = x.gppd.X
solution["Sold"] = s.gppd.X
solution["Wasted"] = u.gppd.X
solution["Pred_demand"] = d.gppd.X

soln_ch = solution

opt_revenue = m.ObjVal
print("\nThe optimal profit: $%f million\n" % opt_revenue)
print(solution.round(3))

#------------------------------------------------------------------------------
# CH^+
#------------------------------------------------------------------------------

diff_d = gppd.add_vars(m, data, name = "diff_d", lb = -np.inf, ub = np.inf)
y_pivot = df.pivot(index='date', columns='region', values='units_sold').loc[:,regions]
m.addConstrs(diff_d.values[i] == d.values[i] - (mu@y_pivot.values)[i] - enlarge_s[i+8] * d_std for i in range(8))
m.addConstr((diff_d * diff_d).sum() <= eps, name = "diff_d_constraint")
m.optimize()

solution = pd.DataFrame(index=regions)
solution["Price"] = p.gppd.X
solution["Historical_Max"] = data.max_price
solution["Allocated"] = x.gppd.X
solution["Sold"] = s.gppd.X
solution["Wasted"] = u.gppd.X
solution["Pred_demand"] = d.gppd.X

soln_chplus = solution

opt_revenue = m.ObjVal
print("\nThe optimal profit: $%f million\n" % opt_revenue)
print(solution.round(3))

#------------------------------------------------------------------------------
# 0.05-CH^+
#------------------------------------------------------------------------------

m.remove(enlarged)
enlarged = m.addConstr(enlarge_s @ enlarge_s <= (0.05*np.sqrt(8+8))**2)
m.optimize()

solution = pd.DataFrame(index=regions)
solution["Price"] = p.gppd.X
solution["Historical_Max"] = data.max_price
solution["Allocated"] = x.gppd.X
solution["Sold"] = s.gppd.X
solution["Wasted"] = u.gppd.X
solution["Pred_demand"] = d.gppd.X

soln_chp05 = solution

opt_revenue = m.ObjVal
print("\nThe optimal profit: $%f million\n" % opt_revenue)
print(solution.round(3))

#------------------------------------------------------------------------------
# 0.1-CH^+
#------------------------------------------------------------------------------

m.remove(enlarged)
enlarged = m.addConstr(enlarge_s @ enlarge_s <= (0.1*np.sqrt(8+8))**2)
m.optimize()

solution = pd.DataFrame(index=regions)
solution["Price"] = p.gppd.X
solution["Historical_Max"] = data.max_price
solution["Allocated"] = x.gppd.X
solution["Sold"] = s.gppd.X
solution["Wasted"] = u.gppd.X
solution["Pred_demand"] = d.gppd.X

soln_chp1 = solution

opt_revenue = m.ObjVal
print("\nThe optimal profit: $%f million\n" % opt_revenue)
print(solution.round(3))

#------------------------------------------------------------------------------
# IsoFor
#------------------------------------------------------------------------------

m = gp.Model("Avocado_Price_Allocation", env = env)
m.Params.NonConvex = 2
m.Params.LogToConsole = 0
m.Params.TimeLimit = 600

p = gppd.add_vars(m, data, name = "price", lb = a_min, ub = a_max)
x = gppd.add_vars(m, data, name = "x", lb = 0)
s = gppd.add_vars(m, data, name = "s")
u = gppd.add_vars(m, data, name = "w")
d = gppd.add_vars(m, data, lb = -gp.GRB.INFINITY, name="demand")

m.setObjective((p * s).sum() - c_waste * u.sum() - (c_transport * x).sum(),
               gp.GRB.MAXIMIZE)

m.addConstr(x.sum() == B)
gppd.add_constrs(m, s, gp.GRB.LESS_EQUAL, x)
gppd.add_constrs(m, s, gp.GRB.LESS_EQUAL, d)
gppd.add_constrs(m, u, gp.GRB.EQUAL, x - s)

m_feats = pd.concat([feats, p], axis=1)[["region", "price", "year", "peak"]]

pred_constr = add_predictor_constr(m, reg, m_feats, d)

p_scaled_var = m.addMVar(shape=(8), name = "price_scaled")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_pivot)
m.addConstrs((p_scaled_var[i] == (p[regions[i]]-scaler.data_min_[i])/(scaler.data_max_[i]-scaler.data_min_[i]) for i in range(8)))
add_isofor_constr(m, X_scaled, p_scaled_var, d=5)

m.optimize()

solution = pd.DataFrame(index=regions)
solution["Price"] = p.gppd.X
solution["Historical_Max"] = data.max_price
solution["Allocated"] = x.gppd.X
solution["Sold"] = s.gppd.X
solution["Wasted"] = u.gppd.X
solution["Pred_demand"] = d.gppd.X

soln_if = solution

opt_revenue = m.ObjVal
print("\nThe optimal profit: $%f million\n" % opt_revenue)
print(solution.round(3))

###############################################################################
# STAGE 6 --- Post-process and draw pictures
###############################################################################

def print_graphs_primary(section):

    #  fig, axs = plt.subplots(4, 2, figsize=(10, 20))
    fig, axs = plt.subplots(4, 2, figsize = (10, 5))

    #  fig.subplots_adjust(wspace=0.4, hspace=0.0)  # You can change these values as needed
    fig.subplots_adjust(hspace=0.0)  # You can change these values as needed

    fig.text(0.05, 0.5, 'Units Sold / Predicted Demand', va='center', ha='center', rotation='vertical', fontsize=16)
    fig.text(0.5, 0.0, 'Price', va='center', ha='center', fontsize=16)

    for k in range(8):

        r = regions[k]
        i = k//2
        j = k%2
        #  X_r = df.loc[(df.region ==r ) & (df.peak == peak_or_not), ["price", "year", "units_sold"]]
        X_r = df.loc[(df.region == r),["price", "year", "units_sold"]]
        x_plt = X_r.price
        #  p_new = np.linspace(.8*min(x_plt),1.2*max(x_plt),50)
        p_new = np.linspace(a_min, a_max, 100)
        x_new = pd.DataFrame(
            data={
                "year": year,
                "peak": peak_or_not,
                "region": r,
                "price": p_new
            },
            index=range(100)
        )
        x_new['units_sold'] = reg.predict(x_new)
        sns.lineplot(data=x_new, x='price', y='units_sold', c='orange', ax=axs[i,j])
        sns.scatterplot(data=X_r, x='price', y='units_sold', legend=0, ax=axs[i,j], c = 'lightsteelblue')
        axs[i, j].legend(title=r, loc='upper right', prop={'size': 3}, handles = [])

        axs[i, j].set_xlabel('')
        axs[i, j].set_ylabel('')
        
        #  plt.tight_layout()

        #  r_plt.legend(fontsize = 12)
        #  plt.savefig('avocado_data_scatterplot.png', dpi = 300, bbox_inches = 'tight')

        #  plt.tight_layout(pad = 1.0)

    #  plt.savefig('../results/figures/avocado_no__opt_soln.png', dpi = 300, bbox_inches = 'tight')

    # main paper section 6
    if section == 'main':
        for k in range(8):
            i = k//2
            j = k%2
            tmp0 = soln_gur.loc[[regions[k]],["Price", "Pred_demand"]]
            sns.scatterplot(data = tmp0, x = 'Price', y = 'Pred_demand', marker='o', legend = 0, ax = axs[i,j], s = 100)
            axs[i, j].set_xlabel('')
            axs[i, j].set_ylabel('')
    
        #  plt.savefig('../results/figures/avocado_gur_opt_soln.png', dpi = 300, bbox_inches = 'tight')
    
        for k in range(8):
            i = k//2
            j = k%2
            tmp1 = soln_box.loc[[regions[k]],["Price", "Pred_demand"]]
            sns.scatterplot(data = tmp1, x = 'Price', y = 'Pred_demand', marker='s', legend = 0, ax = axs[i,j], s = 100)
            axs[i, j].set_xlabel('')
            axs[i, j].set_ylabel('')
    
        #  plt.savefig('../results/figures/avocado_box_opt_soln.png', dpi = 300, bbox_inches = 'tight')
    
        for k in range(8):
            i = k//2
            j = k%2
            tmp2 = soln_ch.loc[[regions[k]],["Price", "Pred_demand"]]
            sns.scatterplot(data = tmp2, x = 'Price', y = 'Pred_demand', marker='^', legend = 0, ax = axs[i,j], s = 100)
            axs[i, j].set_xlabel('')
            axs[i, j].set_ylabel('')
    
        #  plt.savefig('../results/figures/avocado_ch__opt_soln.png', dpi = 300, bbox_inches = 'tight')
    
        for k in range(8):
            i = k//2
            j = k%2
            tmp3 = soln_chplus.loc[[regions[k]],["Price", "Pred_demand"]]
            sns.scatterplot(data = tmp3, x = 'Price', y = 'Pred_demand', marker='D',legend = 0, ax = axs[i,j], s = 100)
            axs[i, j].set_xlabel('')
            axs[i, j].set_ylabel('')
    
        plt.savefig('../results/figures/section_6_avocado_chplus_opt_soln.png', dpi = 300, bbox_inches = 'tight')
        img = Image.open('../results/figures/section_6_avocado_chplus_opt_soln.png').convert("L")  # "L" mode = grayscale
        img.save('../results/figures/section_6_avocado_chplus_opt_soln.png')
    
    
    # EC section s3
    if section == 'EC':
        for k in range(8):
            i = k//2
            j = k%2
            tmp4 = soln_chplus.loc[[regions[k]],["Price", "Pred_demand"]]
            sns.scatterplot(data = tmp4, x = 'Price', y = 'Pred_demand', legend = 0, ax = axs[i,j], s = 100)
            axs[i, j].set_xlabel('')
            axs[i, j].set_ylabel('')
    
        #  plt.savefig('../results/figures/avocado_box_opt_soln.png', dpi = 300, bbox_inches = 'tight')
    
        for k in range(8):
            i = k//2
            j = k%2
            tmp5 = soln_chp05.loc[[regions[k]],["Price", "Pred_demand"]]
            sns.scatterplot(data = tmp5, x = 'Price', y = 'Pred_demand', legend = 0, ax = axs[i,j], s = 100)
            axs[i, j].set_xlabel('')
            axs[i, j].set_ylabel('')
    
        #  plt.savefig('../results/figures/avocado_ch__opt_soln.png', dpi = 300, bbox_inches = 'tight')
    
        for k in range(8):
            i = k//2
            j = k%2
            tmp6 = soln_chp1.loc[[regions[k]],["Price", "Pred_demand"]]
            sns.scatterplot(data = tmp6, x = 'Price', y = 'Pred_demand', legend = 0, ax = axs[i,j], s = 100)
            axs[i, j].set_xlabel('')
            axs[i, j].set_ylabel('')
    
        plt.savefig('../results/figures/section_s4_avocado_enlarged_chplus_opt_soln.png', dpi = 300, bbox_inches = 'tight')

    #  plt.show()
        
print_graphs_primary('main')
print_graphs_primary('EC')

#  solution.round(3)


# ## Changing the Regression Model
# Our regression model has some flaws, so let's try another model type and see how the fit produced, and how that will impact the optimization model.
# 
# Most of the optimization model is unchanged given the new regression model. So to update the optimization we `remove` the previous prediction then add the new one just as we did before.
# 
# With the new model created, we can resolve the optimization and extract the new solution
# 
# This was in introductory look at using the Gurobi Machine Learning package. For more on this example, see the [Price Optimization example of Github](https://github.com/Gurobi/modeling-examples/tree/master/price_optimization)
# as well as how to work with the model interactively.
# 
# Copyright © 2023 Gurobi Optimization, LLC

#  reg = make_pipeline(feat_transform, GradientBoostingRegressor(n_estimators=100, max_leaf_nodes = 20,
#                                                loss = 'absolute_error', random_state = 123))
#  scores = cross_val_score(reg, X_train, y_train, cv=5)
#  print("%0.4f R^2 with a standard deviation of %0.4f" % (scores.mean(), scores.std()))
#  # Fit to entire training data
#  reg.fit(X_train, y_train)
#  y_pred = reg.predict(X_test)
#  print(f"The R^2 value in the test set is {np.round(r2_score(y_test, y_pred),5)}")
#  reg.fit(X, y)
#  y_pred_full = reg.predict(X)
#  print(f"The R^2 value in the full dataset is {np.round(r2_score(y, y_pred_full),5)}")
#
#
#  pred_constr.remove()
#  pred_constr = add_predictor_constr(m, reg, m_feats, d)
#  pred_constr.print_stats()
#  m.update()
#
#  m.optimize()
#
#
#  solution = pd.DataFrame(index=regions)
#
#  solution["Price"] = p.gppd.X
#  solution["Max_Price"] = data.max_price
#  solution["Allocated"] = x.gppd.X
#  solution["Sold"] = s.gppd.X
#  solution["Wasted"] = u.gppd.X
#  solution["Pred_demand"] = d.gppd.X
#
#  opt_revenue = m.ObjVal
#  print("\nThe optimal profit: $%f million" % opt_revenue)
#  solution.round(3)
#
#  print_graphs_primary()
#
