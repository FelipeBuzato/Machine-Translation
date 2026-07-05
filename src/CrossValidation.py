from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
import time

class CrossValidation:
    def __init__(self, cv_folds=5, pipeline=None):
        self.cv_folds = cv_folds
        self.pipeline = pipeline
        self.kfold = KFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=42  # 'None' to shuffle the splits across every call
        )
        self.rng = np.random.default_rng(42)
    

    def evaluate_one_fold(self, X_train, y_train, X_test, y_test, reduced=False, train_size=10000):
        if(reduced):
            # For faster training time
            X_train, y_train = self.reduce(X_train, y_train, train_size)

        start_fit = time.time()
        self.pipeline.fit(X_train, y_train)
        time_fit = time.time() - start_fit
        print(f"Fit time: {time_fit}")

        start_pred = time.time()
        y_pred = self.pipeline.predict(X_test)
        time_pred = time.time() - start_pred
        print(f"Prediction time: {time_pred}\n")

        score = self.pipeline.accuracy(y_pred, y_test)
        return score
    

    def hyper_param_tune_one_fold(self, X_train, y_train, X_test, y_test, param_grid, 
                                  reduced=False, train_size=10000, return_loss_curves=False):  
        if(reduced):
            # For faster training time
            X_train, y_train = self.reduce(X_train, y_train, train_size)
        
        start_cv = time.time()
        best_score, best_params = -1, None
        results, loss_curves, validation_score_curves = [], {}, {}

        for params in ParameterGrid(param_grid):
            self.pipeline.set_params(**params)

            start_fit = time.time()
            self.pipeline.fit(X_train, y_train)
            time_fit = time.time() - start_fit
            print("Params: ", params)
            print(f"Fit time: {time_fit}")

            # Neural Network Stats
            epochs, loss_curve, final_loss, validation_curve, final_validation_score = self.get_NN_stats(self.pipeline.named_steps["model"])
            loss_curves[str(params)], validation_score_curves[str(params)] = loss_curve, validation_curve

            start_pred = time.time()
            y_pred = self.pipeline.predict(X_test)
            time_pred = time.time() - start_pred
            print(f"Prediction time: {time_pred}\n")

            score = self.pipeline.accuracy(y_pred, y_test)

            results.append({**params, "fit_time": time_fit, "pred_time": time_pred, "epochs": epochs, "final_loss": final_loss, 
                            "final_validation_score": final_validation_score, "score": score})
            
            if(score > best_score):
                best_score = score
                best_params = params.copy()
        
        self.pipeline.set_params(**best_params)
        results = pd.DataFrame(results).sort_values("score", ascending=False)

        cv_time = time.time() - start_cv
        print(f"Total CV time: {cv_time}\n")

        if(return_loss_curves):
            return results, loss_curves, validation_score_curves
        return results
    

    def get_NN_stats(self, model):
        loss_curve = getattr(model, "loss_curve_", None)

        if loss_curve is not None and len(loss_curve) > 0:
            final_loss = loss_curve[-1]
            epochs = len(loss_curve)
        else:
            loss_curve = np.nan
            final_loss = np.nan
            epochs = np.nan

        validation_curve = getattr(model, "validation_scores_", None)

        if validation_curve is not None and len(validation_curve) > 0:
            final_validation_score = validation_curve[-1]
        else:
            validation_curve = np.nan
            final_validation_score = np.nan

        return epochs, loss_curve, final_loss, validation_curve, final_validation_score
    

    def reduce(self, X_train, y_train, size):
        indices = self.rng.permutation(X_train.shape[0])
        X_reduced = X_train.iloc[indices[:size]]
        y_reduced = y_train.iloc[indices[:size]]
        return X_reduced, y_reduced