from src.model_training.bayes_hyperopt_cv import RunBayesHyperOpt
from src.model_training.init_search_grids import HyperParamGrid
from src.model_training.random_search_cv import RunRandomizedSearchCV
from src.preprocessing.data_cleaning import CleanData

import run_optimal_model_search as run_optimal_model_search
import pandas as pd

class PerformModelSelection():
    
    def __init__(self, input_data):

        self._data_cleaner = CleanData(input_data)

        self.x = self._data_cleaner.x
        self.y = self._data_cleaner.y

        search_grids = HyperParamGrid()

        RunRandomizedSearchCV(self.x, self.y, search_grids.random_search_grid_dict)
        RunBayesHyperOpt(self.x, self.y, search_grids.bayes_hyper_opt_grid_dict)

if __name__ == '__main__':
    data = pd.read_csv('Placement_Data_Full_Class.csv')
    print(run_optimal_model_search.PerformModelSelection(data))