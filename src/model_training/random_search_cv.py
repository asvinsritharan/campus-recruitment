from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

class RunRandomizedSearchCV():

    def __init__(self, x, y, search_grid):
        '''
        Run Random Search Cross Validation on L1 Logistic Regression, L2 Logistic Regression, Decision Tree, Random Forest, XGBoost algorithms

        Args:
            x: the x matrix to be used to train models
            y: the y array which contains all labels for each corresponding row in x
            search grid: a dictionary which contains all search grids for all models that are to be trained
        
        Returns:
            None
        '''
        self.x = x
        self.y = y
        self.search_grid = search_grid
        # create list containing all models to train
        self._models = [LogisticRegression(penalty='l1', max_iter=10000), LogisticRegression(penalty='l2', max_iter=500), DecisionTreeClassifier(), RandomForestClassifier(), XGBClassifier()]
        # list of string names for each model in the same order as self._models
        self._model_names = ['L1 Logistic Regression', 'L2 Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
        # get results from running randomsearchcv algorithm for all models
        self.results = self.run_randomized_search_cv(self.x, self.y, self._models, self.search_grid, self._model_names)

    def run_randomized_search_cv(self, x, y, models, param_grids, model_names):
        '''
        Run Randomized Search CV algorithm and get mean precision scores for best model of each of the model types

        Args:
            x: the x matrix used to train models
            y: the y array which contains all labels for each corresponding row in x
            models: a list containing models to be tuned with random search cv
            param_grids: a dictionary containing search grids for L1 LR, L2 LR, Decision Tree, Random Forest, XGBoost

        Returns:
            a list containing mean precision score for each best performing model in models after hyper parameter tuning
        '''
        # list to hold mean precision score for each model in models
        results = []
        # get parameter grid key, model and model name to run random search cv on
        for param_grid_name, model, model_name in zip(list(param_grids.keys()), models, model_names):
            # get param grid for current model
            param_grid = param_grids[param_grid_name]
            # initialize LOOCV
            loocv = LeaveOneOut()
            # run random search cv to get optimal model using param grid we specified
            clf = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv = loocv, error_score='raise', n_jobs=-1)
            # run 5 fold cv on optimized model to get precision scores for all 5 folds
            nested_score = cross_val_score(clf, X=x, y=y, cv=5, scoring='precision', n_jobs=-1)
            # print precision scores for model
            print("Precision scores for %s is %s" % (model_name, nested_score))
            # print mean precision score of all 5 folds for current model
            print("Mean precision score for %s is %s" % (model_name, nested_score.mean()))
            # save mean precision score in results list
            results.append(nested_score.mean())
        # return mean precision score for all models
        return results
