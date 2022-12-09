import yaml
from src.logger import lg
import os
import pandas as pd
import numpy as np
from typing import List, Tuple
import dill


class BasicUtils:

    @classmethod
    def configure_float_columns(cls, df: pd.DataFrame, exclude_columns: List, desc: str) -> pd.DataFrame:
        """Typecasts columns other than the ones in list `exclude_columns` as float dtype.

        Args:
            df (pd.DataFrame): Dataframe whose columns gotta be configured.
            exclude_columns (List): List of columns which are not to be typecasted into float dtype.
            desc (str): Description of the said dataframe.

        Returns:
            pd.DataFrame: Dataframe after its desired columns has been typecasted.
        """
        try:
            lg.info(
                f'Typecasting the desired columns of the "{desc}" dataframe into float dtype..')
            df_numerical = df.drop(columns=exclude_columns)
            df_numerical = df_numerical.astype('float')

            for col in df.columns:
                if col not in exclude_columns:
                    df[col] = df_numerical[col]

            lg.info("Typecasting done successfully!")
            return df
            ...
        except Exception as e:
            lg.exception(e)

    @classmethod
    def get_numerical_and_categorical_attributes(cls, df: pd.DataFrame, desc: str):
        """This method returns the Numerical and Categorical attribtes names based on the each column datatype
        and return them as a tuple of respective attributes.

        Args:
            df (pd.DataFrame): Dataframe whose attributes gotta be fetched.
            desc (str): Description of the said dataframe.

        Returns:
            tuple(List, List): First list contains numerical columns names and as such the second contains
            names of categorical ones.
        """
        try:
            lg.info(
                f'fetching the Numerical and Categorical attributes names from the "{desc}" dataframe..')
            num_cols = [col for col in df.columns if df[col].dtypes != 'O']
            cat_cols = [col for col in df.columns if df[col].dtypes == 'O']

            lg.info("attributes fetched successfully!")
            return num_cols, cat_cols
            ...
        except Exception as e:
            lg.exception(e)

    @classmethod
    def get_features_and_labels(cls, df: pd.DataFrame, target: List, desc: str) -> Tuple:
        """Returns the desired features and labels as pandas Dataframe in regard to the said target
        column name.

        Args:
            df (pd.DataFrame): Dataframe whose features and labels are to be returned.
            target (List): List of target column names to be included in the labels dataframe.
            desc (str): Description of the said dataframe.

        Returns:
            Tuple (pd.DataFrame, pd.DataFrame): Tuple of features pandas dataframe and labels pandas 
            dataframe respectively.
        """
        try:
            lg.info(
                f'fetching the input features and target labels out from the "{desc}" dataframe..')
            features = df.drop(columns=target)
            labels = df[target]

            lg.info("returning the said input features and dependent labels..")
            return features, labels
            ...
        except Exception as e:
            lg.exception(e)

    @classmethod
    def save_numpy_array(cls, file_path: str, arr: np.array, desc: str):
        """Saves the numpy array ath the desired `file_path` location.

        Args:
            file_path (str): Location where the numpy array is to be stored.
            arr (np.array): Numpy array which is to be stored.
            desc (str): Description of the numpy array.
        """
        try:
            lg.info(f'Saving the "{desc} Array" at "{file_path}"..')
            # Making sure the dir do exist
            dir = os.path.dirname(file_path)
            os.makedirs(dir, exist_ok=True)
            with open(file_path, "wb") as f:
                np.save(f, arr)
            lg.info(f'"{desc} array" saved successfully!')
            ...
        except Exception as e:
            lg.exception(e)

    @classmethod
    def save_object(cls, file_path: str, obj: object, obj_desc: str) -> None:
        """Saves the desired object at the said desired location.

        Args:
            file_path (str): Location where the object is to be stored.
            obj (object): Object that is to be stored.
            obj_desc (str): Object's description.
        """
        try:
            lg.info(f'Saving the "{obj_desc}" at "{file_path}"..')
            obj_dir = os.path.dirname(file_path)
            os.makedirs(obj_dir, exist_ok=True)
            dill.dump(obj, open(file_path, 'wb'))
            lg.info(f'"{obj_desc}" saved successfully!')
            ...
        except Exception as e:
            lg.exception(e)

    @classmethod
    def load_object(cls, file_path: str, obj_desc: str) -> object:
        try:
            lg.info(f'Loading the {obj_desc}..')
            if not os.path.exists(file_path):
                lg.error('Uh Oh! Looks like the said file path or the object doesn\'t even exist!')
            else:
                dill.load(open(file_path, 'rb'))
                lg.info(f'{obj_desc} loaded successfully!')
            ...
        except Exception as e:
            lg.exception(e)

    @classmethod
    def write_yaml_file(cls, file_path: str, data: dict, desc: str):
        """Dumps the desired data into `yaml` file at the said location.

        Args:
            file_path (str): Location where yaml file is to be created.
            data (dict): Data that is to be dumped into yaml file.
            desc (str): Description of the file.
        """
        try:
            lg.info(f"readying the `{desc}` yaml file..")
            file_dir = os.path.dirname(file_path)
            os.makedirs(file_dir, exist_ok=True)
            with open(file_path, "w") as f:
                yaml.dump(data, f)
            ...
        except Exception as e:
            lg.exception(e)
