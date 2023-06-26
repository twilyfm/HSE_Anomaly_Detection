import pandas as pd

def merging_left(transaction_df, identity_df):
    """
    This function merging two dataframe using left method
    """

    return pd.merge(transaction_df, identity_df,
                    on='TransactionID', how = 'left')