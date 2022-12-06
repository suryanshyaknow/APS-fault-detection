import yaml
from src.logger import lg
import os
import pandas as pd
from typing import List

class BasicUtils:

    def configure_float_columns(self, df: pd.DataFrame, exclude_columns: List) -> pd.DataFrame:
        """Typecasts columns other than the ones in list `exclude_columns` as float dtype.

        Args:
            df (pd.DataFrame): Dataframe whose columns gotta be configured.
            exclude_columns (List): List of columns which are not to be typecasted into float dtype.

        Returns:
            pd.DataFrame: Dataframe after its desired columns has been typecasted.
        """
        try:
            lg.info("Typecasting the desired columns of the dataframe into float dtype..")
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

    def separate_numerical_and_categorical_columns(self, df: pd.DataFrame):
        """This method separates the Numerical and Categorical columns based on the each column datatype
        and return them as a tuple of respective columns.

        Args:
            df (pd.DataFrame): Dataframe of which columns have to be separated out.

        Returns:
            tuple(List, List): First list contains numerical columns names and as such the second contains
            names of categorical ones.
        """
        try:
            lg.info("separating out the Numerical and Categorical columns..")
            num_cols = [col for col in df.columns if df[col].dtypes != 'O']
            cat_cols = [col for col in df.columns if df[col].dtypes == 'O']

            lg.info("columns have been separated out successfully!")
            return num_cols, cat_cols
            ...
        except Exception as e:
            lg.exception(e)


    def write_yaml_file(self, file_path: str, data: dict, desc: str):
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
