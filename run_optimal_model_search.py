import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score


from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import *

class HyperParamGrid():

    def __init__(self):

        self.random_search_grid_dict = self.init_random_search_grids()
        self.bayes_hyper_opt_grid_dict = self.init_bayes_grids()

    def init_random_search_grids(self):
        rf_n_estimators = [int(x) for x in np.linspace(start= 5, stop=200, num = 10)]
        rf_max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] + [None]
        xgb_learning_rate = [x for x in np.arange(0.01, 0.3, 0.1)]
        random_search_grid = {}
        random_search_grid['log_reg_l1_param_set'] = {'C': [0.1, 0.5, 1, 2, 3, 4, 5, 8, 10], 'solver': ['liblinear', 'saga']}
        random_search_grid['log_reg_l2_param_set'] = {'C': [0.1, 0.5, 1, 2, 3, 4, 5, 8, 10], 'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag' ,'saga']}
        random_search_grid['dt_param_set'] = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 5, 10, 20], 'min_samples_leaf': [5, 10, 20, 50, 100]}
        random_search_grid['rf_param_set']={'n_estimators': rf_n_estimators, 'max_features': ['log2', 'sqrt'], 'max_depth': rf_max_depth, 'min_samples_split' : [2,5, 10], 'min_samples_leaf' : [1, 2, 4], 'bootstrap' : [True, False]}
        random_search_grid['xgb_param_set']= {'max_depth': [int(x) for x in np.linspace(start= 3, stop=18, num = 1)], 'learning_rate': xgb_learning_rate, 'gamma': [int(x) for x in np.linspace(start= 1, stop=9, num = 1)], 'reg_alpha': [int(x) for x in np.linspace(start= 5, stop=200, num = 10)], 'reg_lambda': [0.1, 0.3, 0.5, 0.7, 0.9, 1], 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1], 'min_child_weight': [int(x) for x in np.linspace(start= 0, stop=10, num = 1)]}
        return random_search_grid
    
    def init_bayes_grids(self):
        bayes_hyper_opt_grid_dict = {}
        bayes_hyper_opt_grid_dict['log_reg_l1_param_set'] = {'C': hp.uniform('l1_C', 0.1, 10), 'solver': hp.choice('l1_solver', ['liblinear', 'saga'])}

        bayes_hyper_opt_grid_dict['log_reg_l2_param_set'] = {'C': hp.uniform('l2_C', 0.1, 10),
                                'solver': hp.choice('l2_solver', ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag' ,'saga'])}

        bayes_hyper_opt_grid_dict['dt_param_set'] = {'criterion': hp.choice('dt_criterion', ['gini', 'entropy']),
                        'max_depth': hp.quniform('dt_max_depth', 2, 20, 1),
                        'min_samples_leaf': hp.quniform('dt_min_samples_leaf', 5, 100, 1)}

        bayes_hyper_opt_grid_dict['rf_param_set'] = {'n_estimators': hp.quniform("rf_n_estimators", 5, 200, 1), 
                        'max_depth': hp.quniform("rf_max_depth", 3, 15, 1), 
                        'max_features': hp.choice('rf_max_features', ['log2', 'sqrt']),
                        'min_samples_split': hp.quniform("rf_min_samples_split", 2, 10, 1),
                        'min_samples_leaf' : hp.quniform("rf_min_samples_leaf", 1, 10, 1),
                        'bootstrap': hp.choice('rf_model_max_features', [True, False])}


        bayes_hyper_opt_grid_dict['xgb_param_set'] = {'max_depth': hp.quniform("xgb_max_depth", 3, 18, 1),
                'learning_rate': hp.uniform("xgb_learning_rate", 0.01, 0.3),
                'gamma': hp.uniform ('xgb_gamma', 1,9),
                'reg_alpha' : hp.quniform('xgb_reg_alpha', 40,180,1),
                'reg_lambda' : hp.uniform('xgb_reg_lambda', 0,1),
                'colsample_bytree' : hp.uniform('xgb_colsample_bytree', 0.5,1),
                'min_child_weight' : hp.quniform('xgb_min_child_weight', 0, 10, 1)}
        return bayes_hyper_opt_grid_dict

class RunRandomizedSearchCV():

    def __init__(self, x: np.array, y: np.array, search_grid):
        self.x = x
        self.y = y
        self.search_grid = search_grid
        self._models = [LogisticRegression(penalty='l1', max_iter=10000), LogisticRegression(penalty='l2', max_iter=500), DecisionTreeClassifier(), RandomForestClassifier(), XGBClassifier()]
        self._model_names = ['L1 Logistic Regression', 'L2 Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
        self.results = self.run_randomized_search_cv(self.x, self.y, self._models, self.search_grid, self._model_names)

    def run_randomized_search_cv(self, x, y, models, param_grids, model_names):
        model_name = []
        results = []
        for param_grid_name, model, model_name in zip(list(param_grids.keys()), models, model_names):
            param_grid = param_grids[param_grid_name]
            loocv = LeaveOneOut()
            clf = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv = loocv, error_score='raise', n_jobs=-1)
            nested_score = cross_val_score(clf, X=x, y=y, cv=5, scoring='precision', n_jobs=-1)
            model_name = model_name
            print("Precision scores for %s is %s" % (model_name, nested_score))
            print("Mean precision score for %s is %s" % (model_name, nested_score.mean()))
            results.append(nested_score.mean())
        return results


class RunBayesHyperOpt():

    def __init__(self, x: np.array, y: np.array, search_grid):
        self.x = x
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, train_size=0.8, random_state=23)
        self._search_grid = search_grid
        self._objective_functions = [self._lr_l1_objective, self._lr_l2_objective, self._dt_objective, self._rf_objective, self._xgb_objective]
        self._model_names = ['L1 Logistic Regression', 'L2 Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
        self.results = self.run_bayes_alg_model(self._model_names, self._objective_functions, self._search_grid)

    def _xgb_objective(self, param_set):
        classifier =XGBClassifier(
            learning_rate = param_set['learning_rate'], max_depth = int(param_set['max_depth']), gamma = param_set['gamma'],
            reg_alpha = int(param_set['reg_alpha']),min_child_weight=int(param_set['min_child_weight']),
            colsample_bytree=int(param_set['colsample_bytree']), early_stopping_rounds=5)
        evaluation = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        classifier.fit(self.X_train, self.y_train, eval_set=evaluation, verbose=False)
        pred = classifier.predict(self.X_test)
        prec = precision_score(self.y_test, pred)
        print (prec)
        return {'loss': -prec, 'status': STATUS_OK }

    def _lr_l1_objective(self, param_set):
        classifier = LogisticRegression(penalty='l1', C = param_set['C'], solver= param_set['solver'], max_iter=10000)
        classifier.fit(self.X_train, self.y_train)
        pred = classifier.predict(self.X_test)
        prec = precision_score(self.y_test, pred)
        print (prec)
        return {'loss': -prec, 'status': STATUS_OK }

    def _lr_l2_objective(self, param_set):
        classifier = LogisticRegression(penalty='l2', C = param_set['C'], solver= param_set['solver'], max_iter=500)
        classifier.fit(self.X_train, self.y_train)
        pred = classifier.predict(self.X_test)
        prec = precision_score(self.y_test, pred)
        print (prec)
        return {'loss': -prec, 'status': STATUS_OK }

    def _dt_objective(self, param_set):
        classifier = DecisionTreeClassifier(criterion= param_set['criterion'],max_depth = int(param_set['max_depth']), min_samples_leaf=int(param_set['min_samples_leaf']))
        classifier.fit(self.X_train, self.y_train)
        pred = classifier.predict(self.X_test)
        prec = precision_score(self.y_test, pred)
        print (prec)
        return {'loss': -prec, 'status': STATUS_OK }

    def _rf_objective(self, param_set):
        classifier = RandomForestClassifier(n_estimators = int(param_set['n_estimators']), max_depth = int(param_set['max_depth']), max_features = param_set['max_features'], min_samples_split = int(param_set['min_samples_split']), min_samples_leaf = int(param_set['min_samples_leaf']), bootstrap = bool(param_set['bootstrap']))
        classifier.fit(self.X_train, self.y_train)
        pred = classifier.predict(self.X_test)
        prec = precision_score(self.y_test, pred)
        print (prec)
        return {'loss': -prec, 'status': STATUS_OK}
    
    def run_bayes_alg_model(self, model_names, models, param_sets):
        best_hyperparams_per_model = []
        for name, model, param_set_name in zip(model_names, models, list(param_sets.keys())):
            param_set = param_sets[param_set_name]
            best_hyperparams = self.run_bayes_alg_single_model(param_set, model, name)
            best_hyperparams_per_model.append(best_hyperparams)
        return best_hyperparams_per_model

    def run_bayes_alg_single_model(self, param_set, model, name):
        trials = Trials()
        print("The precision score for model " +  name + " is: ", )
        best_hyperparams = fmin(fn = model,
                                space = param_set,
                                algo = tpe.suggest,
                                max_evals = 100,
                                trials = trials)
        return best_hyperparams
            
class CleanData():

    def __init__(self, input_data: pd.DataFrame):
        self.x, self.y = self._process_input_data(input_data)
        
    def _process_input_data(self, input_data: pd.DataFrame):
        y = input_data.status
        y = y.replace({"Placed": 1, "Not Placed": 0})
        x = input_data.drop(['sl_no', 'status'], axis=1)
        x = self.run_data_processing_pipeline(x)
        return x, y
    
    def run_data_processing_pipeline(self, x):    
        numeric_preprocessor = Pipeline(
            steps=[
                ("iterative_impute", IterativeImputer()),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_preprocessor = Pipeline(
            steps=[
                ("ohe",OneHotEncoder(sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("categorical", categorical_preprocessor, x.select_dtypes(include='object').columns),
                ("numerical", numeric_preprocessor, x.select_dtypes(exclude='object').columns),
            ]
        )
        preprocessor_pipeline = Pipeline([('preprocessor', preprocessor)])
        return preprocessor_pipeline.fit_transform(X=x)

class PerformModelSelection():
    
    def __init__(self, input_data: pd.DataFrame):

        self._data_cleaner = CleanData(input_data)

        self.x = self._data_cleaner.x
        self.y = self._data_cleaner.y

        search_grids = HyperParamGrid()

        RunRandomizedSearchCV(self.x, self.y, search_grids.random_search_grid_dict)
        RunBayesHyperOpt(self.x, self.y, search_grids.bayes_hyper_opt_grid_dict)