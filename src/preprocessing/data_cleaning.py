from sklearn.experimental import enable_iterative_imputer
from sklearn.pipeline import Pipeline
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class CleanData():

    def __init__(self, input_data):
        self.x, self.y = self._process_input_data(input_data)
        
    def _process_input_data(self, input_data):
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