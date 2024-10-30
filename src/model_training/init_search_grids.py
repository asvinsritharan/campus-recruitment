import numpy as np
from hyperopt import hp

class HyperParamGrid():

    def __init__(self):

        self.random_search_grid_dict = self.init_random_search_grids()
        self.bayes_hyper_opt_grid_dict = self.init_bayes_grids()

    def init_random_search_grids(self):
        '''
        Initialize hyper parameter search grid for random search cross validation algorithm. To be run with random forest, l1 logistic regression, 
        l2 logistic regression, decision tree and xgboost algorithms.
        
        Args:
            None

        Returns:
            Dictionary containing hyper parameter grids for L1 Logistic Regression, L2 Logistic Regression, Decision Tree, Random Forest, and XGBoost Models
        '''
        # create number space for random forest n_estimators hyper parameter
        rf_n_estimators = [int(x) for x in np.linspace(start= 5, stop=200, num = 10)]
        # create number space for random forest max_depth hyper parameter
        rf_max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] + [None]
        # create number space for xgboost learning rate hyper parameter
        xgb_learning_rate = [x for x in np.arange(0.05, 0.7, 0.05)]
        # create dictionary to hold search grids for models
        random_search_grid = {}
        # create search grid for l1 logistic regression
        random_search_grid['log_reg_l1_param_set'] = {'C': [0.1, 0.5, 1, 2, 3, 4, 5, 8, 10], 'solver': ['liblinear', 'saga']}
        # create search grid for l2 logistic regression
        random_search_grid['log_reg_l2_param_set'] = {'C': [0.1, 0.5, 1, 2, 3, 4, 5, 8, 10], 'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag' ,'saga']}
        # create search grid for decision tree model
        random_search_grid['dt_param_set'] = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 5, 10, 20], 'min_samples_leaf': [5, 10, 20, 50, 100]}
        # create search grid for random forest model
        random_search_grid['rf_param_set']={'n_estimators': rf_n_estimators, 'max_features': ['log2', 'sqrt'], 'max_depth': rf_max_depth, 'min_samples_split' : [2,5, 10], 'min_samples_leaf' : [1, 2, 4], 'bootstrap' : [True, False]}
        # create search grid for xgboost model
        random_search_grid['xgb_param_set']= {'max_depth': [int(x) for x in np.linspace(start= 3, stop=18, num = 1)], 'learning_rate': xgb_learning_rate, 'gamma': [int(x) for x in np.linspace(start= 1, stop=9, num = 1)], 'reg_alpha': [int(x) for x in np.linspace(start= 5, stop=200, num = 10)], 'reg_lambda': [0.1, 0.3, 0.5, 0.7, 0.9, 1], 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1], 'min_child_weight': [int(x) for x in np.linspace(start= 0, stop=10, num = 1)]}
        # return full search grid
        return random_search_grid
    
    def init_bayes_grids(self):
        '''
        Initialize hyper parameter search grid for random search cross validation algorithm. To be run with random forest, l1 logistic regression, 
        l2 logistic regression, decision tree and xgboost algorithms.
        
        Args:
            None

        Returns:
            Dictionary containing hyper parameter grids for L1 Logistic Regression, L2 Logistic Regression, Decision Tree, Random Forest, and XGBoost Models
        '''
        # initialize search grid for bayes hyper parameter optimization
        bayes_hyper_opt_grid_dict = {}
        # create search grid for l1 logistic regression
        bayes_hyper_opt_grid_dict['log_reg_l1_param_set'] = {'C': hp.uniform('l1_C', 0.1, 10), 'solver': hp.choice('l1_solver', ['liblinear', 'saga'])}
        # create search grid for l2 logistic regression
        bayes_hyper_opt_grid_dict['log_reg_l2_param_set'] = {'C': hp.uniform('l2_C', 0.1, 10),
                                'solver': hp.choice('l2_solver', ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag' ,'saga'])}
        # create search grid for decision tree
        bayes_hyper_opt_grid_dict['dt_param_set'] = {'criterion': hp.choice('dt_criterion', ['gini', 'entropy']),
                        'max_depth': hp.quniform('dt_max_depth', 2, 20, 1),
                        'min_samples_leaf': hp.quniform('dt_min_samples_leaf', 5, 100, 1)}
        # create search grid for random forest
        bayes_hyper_opt_grid_dict['rf_param_set'] = {'n_estimators': hp.quniform("rf_n_estimators", 5, 200, 1), 
                        'max_depth': hp.quniform("rf_max_depth", 3, 15, 1), 
                        'max_features': hp.choice('rf_max_features', ['log2', 'sqrt']),
                        'min_samples_split': hp.quniform("rf_min_samples_split", 2, 10, 1),
                        'min_samples_leaf' : hp.quniform("rf_min_samples_leaf", 1, 10, 1),
                        'bootstrap': hp.choice('rf_model_max_features', [True, False])}
        # create search grid for bayes hyper parameter optimization
        bayes_hyper_opt_grid_dict['xgb_param_set'] = {'max_depth': hp.quniform("xgb_max_depth", 3, 18, 1),
                'learning_rate': hp.uniform("xgb_learning_rate", 0.05, 0.7),
                'gamma': hp.uniform ('xgb_gamma', 1,9),
                'reg_alpha' : hp.quniform('xgb_reg_alpha', 40,180,1),
                'reg_lambda' : hp.uniform('xgb_reg_lambda', 0,1),
                'colsample_bytree' : hp.uniform('xgb_colsample_bytree', 0.5,1),
                'min_child_weight' : hp.quniform('xgb_min_child_weight', 0, 10, 1)}
        # return full search grid
        return bayes_hyper_opt_grid_dict