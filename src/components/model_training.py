from src.logger import lg
import os
import numpy as np
from src.entities.config import ModelTrainingConfig
from src.entities.artifact import ModelTrainingArtifact, DataTransformationArtifact
from src.utils.file_operations import BasicUtils
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from typing import Dict
from dataclasses import dataclass


@dataclass
class ModelTraining:
    """Shall be used for training the shortlisted model, finetuning it and apparently returning configurations of the built
    (and finetuned) model and its peformance measures.

    Args:
        data_transformation_artifact (DataTransforamtion): Takes in a `DataTransformationArtifact` object to have access to 
        all relevant configs of Data Transforamtion stage.
    """
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.ModelTraining" class')

    data_transformation_artifact: DataTransformationArtifact
    model_training_config = ModelTrainingConfig()

    def finetune_model(self, X: np.array, y: np.array, base_model=XGBClassifier()) -> Dict:
        """Finetunes the base XGBClassifier and returns the `best params` for the base classifier 
        to be trained on the given features and labels, via GridSearchCV.

        Args:
            X (np.array): Dataset's features on which the best model has to be trained.
            y (np.array): Respective target labels.
            base_model (XGBClassifier, optional): Base XGBClassifier. Defaults to XGBClassifier().

        Raises:
            e: Raises relevant exception should any sort of error pops up while finetuning the said model.

        Returns:
            Dict: Best params found (via GridSearchCV) for the base XGBClassifier to be trained on.
        """
        try:
            grid_params = {
                "eta": [.3, .5, 1],
                "colsample_bytree": [.2, .5, 1],
                "colsample_bylevel": [.5, 1],
                "colsample_bynode": [.5, 1],
                "max_depth": [3, 6],
                "random_state": [42],
                "n_estimators": [100, 300, 500]
            }
            lg.info(
                f"Range of params to choose the best ones from: {grid_params}")
            grid_search = GridSearchCV(
                param_grid=grid_params, estimator=base_model, cv=5, verbose=3, scoring='f1_micro')
            lg.info("Grid Search cross-validation begins..")
            grid_search.fit(X, y)
            lg.info(
                f"Cross-Validation concluded with the best params as {grid_search.best_params_}\nand the best estimator as {grid_search.best_estimator_}")

            return grid_search.best_params_
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def train_model(self, X: np.array, y: np.array) -> XGBClassifier:
        """Trains the XGBClassifier on the provided features and target.

        Args:
            X (np.array): Features on which the XGBClassifer has to be trained.
            y (np.array): Target for the given features.

        Raises:
            e: Raises relevant exception should any sort of error pops up while training the said model.

        Returns:
            XGBClassifier: Fitted XGBClassifier on the given features and label.
        """
        try:
            xgb_clf = XGBClassifier()
            lg.info(f"Our base Model: {xgb_clf}")
            # lg.info(
            #     'Since, `refit` defaults to True, we don\'t need to firstly fetch the best params and then train our base model using them.\nWe could straight away grab the best estimator that\'s been already trained on the entire training set.')
            lg.info(
                "fetching the best params found to train base XGBClassifier, as the result of the GridSearchCV..")
            # best_params = self.finetune_model(X, y, base_model=xgb_clf)
            best_params = {
                'colsample_bylevel': 0.5,
                'colsample_bynode': 1,
                'colsample_bytree': 1,
                'eta': 0.3,
                'max_depth': 6,
                'n_estimators': 100,
                'random_state': 42
            }
            lg.info(f"best_params we got via GridSearchCV: {best_params}")
            lg.info("building the best XGBClassifier using the fetched best params..")
            best_mod = XGBClassifier(**best_params)
            lg.info(
                f"fitting the XGBClassifier on training features and labels..")
            best_mod.fit(X, y)
            lg.info("XGBClassifier trained well!")

            return best_mod
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def initiate(self) -> ModelTrainingArtifact:
        """Triggers the Model Building stage of the training pipeline and returns the configurations of the model built and its 
        performance measures, as in contained by the `ModelTrainingArtifact`.

        Raises:
            e: Raises relevant exception should any sort of error pops in the Model Building stage.

        Returns:
            ModelTrainingArtifact: Contains the built model's config and its performance measures.
        """
        try:
            lg.info(f"\n{'='*27} MODEL TRAINING {'='*40}")

            ############################# Fetch the Training and Test arrays ##################################
            lg.info("fetching the transformed training and test arrays..")
            training_arr = BasicUtils.load_numpy_array(
                file_path=self.data_transformation_artifact.transformed_training_file_path,
                desc="Training")
            test_arr = BasicUtils.load_numpy_array(
                file_path=self.data_transformation_artifact.transformed_test_file_path,
                desc="Test")
            lg.info("transformed training and test arrays fetched successfully..")

            ############################## Separate Features and Label out ####################################
            lg.info(
                "separating the independent features and dependent target out from training and test arrays..")
            X_train, y_train = training_arr[:, :-1], training_arr[:, -1]
            lg.info("training features and target label fetched!")
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            lg.info("test features and target label fetched!")

            ######################################## Train the Model ##########################################
            mod = self.train_model(X_train, y_train)

            ################################# Compute Performance Metric ######################################
            lg.info(
                "Computing Performance Metric..\nChose `F1 score` as the performance metric for this project")
            # Performance on Training set
            yhat_train = mod.predict(X_train)
            f1_training_score = round(
                f1_score(y_true=y_train, y_pred=yhat_train), 4)
            lg.info(f'Training `F1 score`: {f1_training_score}')
            # Performance on Test set
            yhat_test = mod.predict(X_test)
            f1_test_score = round(f1_score(y_true=y_test, y_pred=yhat_test), 4)
            lg.info(f'Test `F1 score`: {f1_test_score}')

            ########################## Check if Model's performance is good enough ############################
            lg.info("checking whether the model's performance is good enough..")
            lg.info(
                f'Expected Score: {self.model_training_config.expected_score}')
            if f1_test_score >= self.model_training_config.expected_score:
                lg.info("Yep, Model's performance is good enough!")
            else:
                lg.warning(
                    "Model's performance ain't that good. Gotta retrain with better params!!")

            ####################################### Overfitting Check #########################################
            lg.info("Performing check for Overfitting..")
            diff = abs(f1_test_score - f1_training_score)
            lg.info(
                f"Overfitting Threshold: {self.model_training_config.overfit_thresh}")
            lg.info(f"the difference we got : {diff}")
            if diff > self.model_training_config.overfit_thresh:
                lg.warning(
                    f"Since the difference between `f1_test_score` and `f1_train_score` is greater than the overfitting thresh i.e {self.model_training_config.overfit_thresh}, the model definitely Overfits! ")
            else:
                lg.info("Model ain't Overfitting. We're good to go!")

            #################################### Save the Trained Model #######################################
            lg.info("Saving the trained model i.e. XGBClassifier..")
            BasicUtils.save_object(
                file_path=self.model_training_config.model_path, obj=mod,
                obj_desc="Trained Model (XGBClassifier)")

            #################################### Save Artifacts Config ########################################
            model_training_artifact = ModelTrainingArtifact(
                model_path=self.model_training_config.model_path,
                f1_training_score=f1_training_score,
                f1_test_score=f1_test_score
            )
            lg.info(f"Model Training Artifact: {model_training_artifact}")
            lg.info(f"Model Training completed!")

            return model_training_artifact
            ...
        except Exception as e:
            lg.exception(e)
            raise e
