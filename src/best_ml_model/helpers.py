import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from pandas.api.types import is_integer_dtype, is_numeric_dtype

def reduce_memory_usage(df):
    """Сокращает объем, занимаемый датасетом в памяти за счет изменения типов столбцов;

    df -- pandas DataFrame"""
    for col in df.columns:
        col_type = df[col].dtype

        if is_numeric_dtype(col_type):
            min_ = df[col].min()
            max_ = df[col].max()

            if is_integer_dtype(col_type):
                if min_ > np.iinfo(np.int8).min and max_ < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif min_ > np.iinfo(np.int16).min and max_ < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif min_ > np.iinfo(np.int32).min and max_ < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif min_ > np.iinfo(np.int64).min and max_ < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if min_ > np.finfo(np.float16).min and max_ < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif min_ > np.finfo(np.float32).min and max_ < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    return df


def plot_roc_curve(x, y, roc_auc):
    plt.figure(figsize=(16, 8))
    lw = 2

    plt.plot(x, y, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc, alpha=0.5)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', alpha=0.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver operating characteristic', fontsize=20)
    plt.legend(loc="lower right", fontsize=16)
    return plt.show()


def catboost_with_params_for_train(trial, X_train, X_test, y_train, y_test, df):
    X = df.drop(target_name, axis=1)
    categorical_features_indices = np.where(X.dtypes == 'category')[0]

    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "task_type": "GPU",
        "eval_metric": "AUC",
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    cat_cls = CatBoostClassifier(**param)

    cat_cls.fit(X_train, y_train, eval_set=[(X_test, y_test)], cat_features=categorical_features_indices, verbose=0, early_stopping_rounds=100)

    preds = cat_cls.predict(X_test)
    pred_labels = np.rint(preds)
    roc_auc_ = roc_auc_score(y_test, pred_labels)
    return roc_auc_
