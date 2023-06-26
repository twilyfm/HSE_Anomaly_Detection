class FillNan():
    """
    This class fills in missing values in dataframe.

    Replacing missing values for numeric columns can be done
    using methods "mean" or "median" ("mean" by default).

    Replacing missing values for categorical columns can be done
    using most frequent value or constant value ("constant" by default).

    The class also supports dropping columns with a lot of gaps (False by default).

    By default, methods mutate the original dataframe (inplace=True)
    this can be changed by setting the 'inplace' parameter to False.
    """
    
    _parameter_constraints: dict = {
        "num_filler": {"mean", "median"},
        "cat_filler": {"most_frequent", "constant"},
        'drop_highly_missed': ["boolean"],
        'inplace': ["boolean"]
    }

    def __init__(self, num_filler="mean", cat_filler='constant',
                 drop_highly_missed=False, inplace=True):
        self.num_filler = num_filler
        self.cat_filler = cat_filler
        self.drop_highly_missed = drop_highly_missed
        self.inplace = inplace

    def fit(self, X, y=None):
        cols_filler = {}
        for col in X.columns:
            if X[col].dtypes != "O":
                if self.num_filler == "mean":
                    cols_filler[col] = X[col].mean()
                elif self.num_filler == "median":
                    cols_filler[col] = X[col].median()
            elif X[col].dtypes == "O":
                if self.cat_filler == "most_frequent":
                    cols_filler[col] = X[col].mode()[0]
                elif self.cat_filler == "constant":
                    cols_filler[col] = 'missing'
        if self.drop_highly_missed == True:
            col_missings = X.isna().sum()
            cols_to_drop = col_missings[(col_missings / X.shape[0]) > 0.9].index
            self.cols_to_drop = cols_to_drop

        self.cols_filler = cols_filler

        return self

    def transform(self, X):
        if self.inplace == False:
            if self.drop_highly_missed == True:
                X_new = X.drop(self.cols_to_drop, axis=1).copy()
            else:
                X_new = X.copy()
            X_new = X_new.fillna(value=self.cols_filler)
            return X_new
        else:
            if self.drop_highly_missed == True:
                X = X.drop(self.cols_to_drop, axis=1)
            X = X.fillna(value=self.cols_filler)
            return X
