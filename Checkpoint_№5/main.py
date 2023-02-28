# -*- coding: utf-8 -*-

from helpers import reduce_memory_usage, plot_roc_curve, catboost_with_params_for_train

import pickle
import optuna
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier, Pool
from catboost.utils import get_roc_curve

from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, auc
from sklearn.model_selection import train_test_split

PATH_TO_DF_ = "./merge_filled_without_drop.parquet" # объединенные transaction и identity


def main(train=False):
    df=pd.read_parquet(PATH_TO_DF_)
    merged = reduce_memory_usage(df)
    X = merged.drop('isFraud', axis=1)
    y = merged['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    und = RandomUnderSampler(random_state=42)

    X_und, y_und = und.fit_resample(X_train, y_train)

    categorical_features_indices = np.where(X.dtypes == 'category')[0]

    if train == True:
        print("train_mode = True")
        study = optuna.create_study(direction="maximize")

        study.optimize(catboost_with_params_for_train, n_trials=50, timeout=600)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        model = CatBoostClassifier(verbose=True, random_state=42,
                                   objective=trial.params["objective"],
                                   depth=trial.params["depth"],
                                   boosting_type=trial.params["boosting_type"], task_type="GPU", eval_metric="AUC",
                                   bootstrap_type=trial.params["bootstrap_type"])

        model.fit(X_train, y_train, cat_features=categorical_features_indices, eval_set=(X_test, y_test), plot=True)
        return model
    else:
        print("train_mode = False")

        model = pickle.load(open("catboost_classifier.pkl", 'rb'))
        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))

        eval_pool = Pool(X_test, y_test, cat_features=categorical_features_indices)
        curve = get_roc_curve(model, eval_pool)
        (fpr, tpr, thresholds) = curve
        roc_auc = auc(fpr, tpr)

        plot_roc_curve(fpr, tpr, roc_auc=roc_auc)



if __name__ == "__main__":
    main(train=False)

