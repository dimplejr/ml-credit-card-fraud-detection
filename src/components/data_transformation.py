import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    resampler_obj_file_path = os.path.join('artifacts', "resampler.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self,X):
        try:
            num_features = X.select_dtypes(exclude="object").columns
            data_pipeline = Pipeline(
                steps=[
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("data_pipeline", data_pipeline, num_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def get_resampling_pipeline(self, X, y):
        try:
            count_0 = y.value_counts()[0]
            count_1 = y.value_counts()[1]
            average_count = int((count_0 + count_1) / 2)

            smote = SMOTE(sampling_strategy={1: average_count}, random_state=42)
            undersampler = RandomUnderSampler(sampling_strategy={0: average_count}, random_state=42)
            
            X_resampled, y_resampled = smote.fit_resample(X, y)
            X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)

            return X_resampled, y_resampled

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Obtain resampling pipeline
            logging.info("Obtaining resampling pipeline")
            target_column_name = "Class"
            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]

            X_train_resampled, y_train_resampled = self.get_resampling_pipeline(X_train, y_train)

            # Apply preprocessing after resampling
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(X_train_resampled)
            X_train_preprocessed = preprocessing_obj.fit_transform(X_train_resampled)
            X_test_preprocessed = preprocessing_obj.transform(test_df.drop(columns=[target_column_name], axis=1))

            # Combine features and target
            train_arr = np.c_[X_train_preprocessed, np.array(y_train_resampled)]
            test_arr = np.c_[X_test_preprocessed, np.array(test_df[target_column_name])]

            # Save preprocessing and resampling objects
            logging.info("Saving preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Saving resampling pipeline object.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                    )

        except Exception as e:
            raise CustomException(e, sys)