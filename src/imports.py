import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_spd_matrix
from sklearn.metrics import r2_score
import gurobipy as gp
from gurobipy import GRB
from sklearn.svm import OneClassSVM
from sklearn.metrics.pairwise import manhattan_distances
from gurobi_ml.sklearn import add_mlp_regressor_constr
from gurobi_ml.sklearn import add_random_forest_regressor_constr
from gurobi_ml.sklearn import add_gradient_boosting_regressor_constr
from gurobi_ml.torch import add_sequential_constr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.optimize import linprog
from tqdm import tqdm
import joblib
import json
import time