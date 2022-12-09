from src.logger import lg
import os
from src.entities.config import ModelTrainingConfig
from src.entities.artifact import ModelTrainingArtifact, DataTransformationArtifact
from src.utils.file_operations import BasicUtils
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from dataclasses import dataclass


@dataclass
class ModelTraining:
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.ModelTraining" class')

    data_transformation_artifact: DataTransformationArtifact
    model_training_config = ModelTrainingConfig()

    def finetune_model(self):
        try:
            ...
        except Exception as e:
            lg.exception(e)

    def train_model(self, X, y) -> XGBClassifier:
        try:
            xgb_clf = XGBClassifier()
            lg.info("training the XGBClassifier..")
            xgb_clf.fit(X, y)
            lg.info("XGBClassifier trained well!")
            return xgb_clf
            ...
        except Exception as e:
            lg.exception(e)

    def initiate(self) -> ModelTrainingArtifact:
        try:
            lg.info(f"\n{'='*22} MODEL TRAINING {'='*35}")

            ######################## Fetch the Training and Test arrays #######################################
            lg.info("fetching the transformed training and test arrays..")
            training_arr = BasicUtils.load_numpy_array(
                file_path=self.data_transformation_artifact.transformed_training_file_path,
                desc="Training")
            test_arr = BasicUtils.load_numpy_array(
                file_path=self.data_transformation_artifact.transformed_test_file_path,
                desc="Test")
            lg.info("transformed training and test arrays fetched successfully..")

            ######################## Separate Features and Label out ##########################################
            lg.info(
                "separating the independent features and dependent target out from training and test arrays..")
            X_train, y_train = training_arr[:, :-1], training_arr[:, -1]
            lg.info("training features and target label fetched!")
            X_test, y_test = training_arr[:, :-1], training_arr[:, -1]
            lg.info("test features and target label fetched!")

            ################################ Train the Model ##################################################
            mod = self.train_model(X_train, y_train)

            ############################## Compute Performance Metric #########################################
            lg.info(
                "Computing Performance Metric..\nChose `F1 score` as the performance metric for this project")
            # Performance on Training set
            yhat_train = mod.predict(X_train)
            f1_training_score = f1_score(y_true=y_train, y_pred=yhat_train)
            lg.info(f'Training `F1 score`: {f1_training_score}')
            # Performance on Test set
            yhat_test = mod.predict(X_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=yhat_test)
            lg.info(f'Test `F1 score`: {f1_training_score}')

            ######################### Check if Model's performance is good enough #############################
            lg.info("checking whether the model's performance is good enough..")
            lg.info(
                f'Expected Score: {self.model_training_config.expected_score}')
            if f1_test_score >= self.model_training_config.expected_score:
                lg.info("Yep, Model's performance is good enough!")
            else:
                lg.warning(
                    "Model's performance ain't that good. Gotta retrain with better params!!")

            ############################## Overfitting Check ##################################################
            lg.info("Performing check for Overfitting..")
            diff = abs(f1_test_score - f1_training_score)
            if diff > self.model_training_config.overfit_thresh:
                lg.info(
                    f"Overfitting Threshold: {self.model_training_config.overfit_thresh}")
                lg.info(f"the difference we got : {diff}")
                lg.warning(
                    f"Since the difference between `f1_test_score` and `f1_train_score` is greater than the overfitting thresh i.e {self.model_training_config.overfit_thresh}, the model definitely Overfits! ")
            else:
                lg.info("Model ain't Overfitting. We're good to go!")

            ############################## Save the Trained Model #############################################
            lg.info("Saving the trained model i.e. XGBClassifier..")
            BasicUtils.save_object(
                file_path=self.model_training_config.model_path, obj=mod, 
                obj_desc="Trained Model (XGBClassifier)")

            ############################## Save Artifacts Config ##############################################
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
