# Import Libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer

def preprocessData(file='assets/updated_house_df.csv', data_from = 2000):
    """
    take dataframe from manipulation step 
    to complete preprocessing

    returns tuple of 3 items (df, column_trans, idx):
        1. Preprocessed DF
        2. Preprecess Pipeline Image
        3. Index on where to find the columns for each transformer applied to the orig df
    
    """

    #read in file
    df = pd.read_csv(file)
    df = df[df['year'] >= data_from]

    # a few manipulations
    df['physical_zip_code'] = df['physical_zip_code'].astype('str')
    if data_from < 2017: # don't include wake supply and demand date if year is before 2017
        df = df.drop(columns=['wake_supply_index', 'wake_demand_index'])  # incomplete data

    # define which handling to which columns

    col_drop = ['deed_date', 'land_sale_price'] # columns to drop
    col_passthru = ['electric', 'gas', 'water', 'sewer', 'all', 'is_covid', 'covid_year_timeline'] #columns to keep
    col_minmax_scale = ['bath_fixtures', 'bath'] #cols to apply min max scale

    # if cols are not from the above, we take that categorical (non number) cols for OneHotEncoding
    df_to_process1 = df.drop(columns = col_drop + col_passthru + col_minmax_scale)
    col_onehot_cat = df_to_process1.select_dtypes(exclude=np.number).columns.tolist()

    # if cols are not from the above, they are the remaining numeric cols, we now check for skewness
    df_to_process2 = df_to_process1.drop(columns = col_onehot_cat).columns.tolist()

    skew_check = df[df_to_process2].skew()

    col_log_scale = skew_check.loc[skew_check >= 1].index.to_list() # if skewness > 1 (right skewwed, then we first log then std scale)
    col_scale = skew_check.loc[skew_check < 1].index.to_list() # if not skew to the right, then we only apply std scale

    # define the function for log and scale the column - for next step    
    log_scale_transformer = make_pipeline(
        FunctionTransformer(func=np.log1p, feature_names_out='one-to-one'), 
        StandardScaler()
        )

    # transformer object - transform the columns based on above define strategies
    column_trans = ColumnTransformer(
    [
        ("onehot", OneHotEncoder(), col_onehot_cat),
        ("log_scaled", log_scale_transformer, col_log_scale),
        ('mm_scaled', MinMaxScaler(), col_minmax_scale),
        ('std_scale', StandardScaler(), col_scale),
        ("passthru", "passthrough", col_passthru),
    ],
    remainder="drop",
    verbose_feature_names_out=True
    )

    # preprocess all features from original df 
    X = column_trans.fit_transform(df)

    # new featurenames 
    colnames = column_trans.get_feature_names_out().tolist()

    # preprocessed dataframe
    df = pd.DataFrame(X, columns = colnames)
    
    # index to know how to slice and differentiate the columns
    idx = column_trans.output_indices_

    return df, column_trans, idx

# https://www.analyticsvidhya.com/blog/2021/05/understanding-column-transformer-and-machine-learning-pipelines/

