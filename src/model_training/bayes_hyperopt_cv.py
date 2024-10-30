from hyperopt import STATUS_OK, Trials, fmin, tpe

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

from sklearn.model_selection import train_test_split

class RunBayesHyperOpt():

    def __init__(self, x, y, search_grid):
        '''
        Run Bayes Hyper Parameter Optimization on L1 Logistic Regression, L2 Logistic Regression, Decision Tree, Random Forest, XGBoost models

        Args:
            x: the x matrix to be used to train models
            y: the y array which contains all labels for each corresponding row in x
            search grid: a dictionary which contains all search grids for all models that are to be trained
        
        Returns:
            None
        '''
        self.x = x
        self.y = y
        # get 80-20 train test split on input x and y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, train_size=0.8, random_state=23)
        self._search_grid = search_grid
        # create list of objective functions which are to be minimized
        self._objective_functions = [self._lr_l1_objective, self._lr_l2_objective, self._dt_objective, self._rf_objective, self._xgb_objective]
        # create list of model names
        self._model_names = ['L1 Logistic Regression', 'L2 Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
        # get optimal hyperparameters for each model
        self.optimal_hyper_params = self.run_bayes_alg_model(self._model_names, self._objective_functions, self._search_grid)

    def _xgb_objective(self, param_set):
        '''
        Initailize XGBoost Objective function to run Bayes Hyper Parameter Optimization on to obtain tuned model
        
        Args:
            param_set: a dictionary which is the parameter set to be used to tune model

        Returns:
            a dictionary containing loss metric and status of best tuned model
        '''
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
        '''
        Initailize L1 Logistic Regression Objective function to run Bayes Hyper Parameter Optimization on to obtain tuned model
        
        Args:
            param_set: a dictionary which is the parameter set to be used to tune model

        Returns:
            a dictionary containing loss metric and status of best tuned model
        '''
        classifier = LogisticRegression(penalty='l1', C = param_set['C'], solver= param_set['solver'], max_iter=10000)
        classifier.fit(self.X_train, self.y_train)
        pred = classifier.predict(self.X_test)
        prec = precision_score(self.y_test, pred)
        print (prec)
        return {'loss': -prec, 'status': STATUS_OK }

    def _lr_l2_objective(self, param_set):
        '''
        Initailize L2 Logistic Regression Objective function to run Bayes Hyper Parameter Optimization on to obtain tuned model
        
        Args:
            param_set: a dictionary which is the parameter set to be used to tune model

        Returns:
            a dictionary containing loss metric and status of best tuned model
        '''
        classifier = LogisticRegression(penalty='l2', C = param_set['C'], solver= param_set['solver'], max_iter=500)
        classifier.fit(self.X_train, self.y_train)
        pred = classifier.predict(self.X_test)
        prec = precision_score(self.y_test, pred)
        print (prec)
        return {'loss': -prec, 'status': STATUS_OK }

    def _dt_objective(self, param_set):
        '''
        Initailize Decision Tree Objective function to run Bayes Hyper Parameter Optimization on to obtain tuned model
        
        Args:
            param_set: a dictionary which is the parameter set to be used to tune model

        Returns:
            a dictionary containing loss metric and status of best tuned model
        '''
        classifier = DecisionTreeClassifier(criterion= param_set['criterion'],max_depth = int(param_set['max_depth']), min_samples_leaf=int(param_set['min_samples_leaf']))
        classifier.fit(self.X_train, self.y_train)
        pred = classifier.predict(self.X_test)
        prec = precision_score(self.y_test, pred)
        print (prec)
        return {'loss': -prec, 'status': STATUS_OK }

    def _rf_objective(self, param_set):
        '''
        Initailize Random Forest Objective function to run Bayes Hyper Parameter Optimization on to obtain tuned model
        
        Args:
            param_set: a dictionary which is the parameter set to be used to tune model

        Returns:
            a dictionary containing loss metric and status of best tuned model
        '''
        classifier = RandomForestClassifier(n_estimators = int(param_set['n_estimators']), max_depth = int(param_set['max_depth']), max_features = param_set['max_features'], min_samples_split = int(param_set['min_samples_split']), min_samples_leaf = int(param_set['min_samples_leaf']), bootstrap = bool(param_set['bootstrap']))
        classifier.fit(self.X_train, self.y_train)
        pred = classifier.predict(self.X_test)
        prec = precision_score(self.y_test, pred)
        print (prec)
        return {'loss': -prec, 'status': STATUS_OK}
    
    def run_bayes_alg_model(self, model_names, models, param_sets):
        '''
        Run Bayes Hyper Parameter Optimization algorithm on all models in models and get the best hyper parameters for each model
        
        Args:
            model_names: a list of str where each string is a model in the same order as models
            models: a list of objective functions where objective function at models[i] corresponds to model at model_names[i]
            param_sets: a dictionary containing dictionaries of parameter sets to be used to tune models

        Returns:
            a list containing best hyper parameters for each model in models
        '''
        best_hyperparams_per_model = []
        for name, model, param_set_name in zip(model_names, models, list(param_sets.keys())):
            param_set = param_sets[param_set_name]
            best_hyperparams = self.run_bayes_alg_single_model(param_set, model, name)
            best_hyperparams_per_model.append(best_hyperparams)
        return best_hyperparams_per_model

    def run_bayes_alg_single_model(self, param_set, model, name):
        '''
        Run Bayes Hyper Parameter Optimization algorithm for model and get the best hyper parameter set
        
        Args:
            model_name: str which is the name of the model that is being trained
            model: objective functions to minimize
            param_set: a dictionary of parameter set to be used to tune model

        Returns:
            a dict of best hyper parameters for model
        '''
        trials = Trials()
        print("The precision score for model " +  name + " is: ", )
        best_hyperparams = fmin(fn = model,
                                space = param_set,
                                algo = tpe.suggest,
                                max_evals = 100,
                                trials = trials)
        return best_hyperparams