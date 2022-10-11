# Import Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def preprocessData(file='assets/updated_house_df.csv'):
    """
    take dataframe from manipulation step 
    to complete preprocessing

    returns 3 dataframes:
        1. Scaled df will all data preprocessed
        2. Scaled df will post-COVID period data preprocessed
        3. Scaled df will pre-COVID period data preprocessed
    
    """

    #read in file
    df = pd.read_csv(file)

    #preprocessing features
    df['isCovid'] = df['covid_cases'].apply(lambda x: 1 if x > 0 else 0)

    #todo: should use one hot encoder
    #todo: log transform

    le = LabelEncoder() 
    df['physical_city_codes'] = le.fit_transform(df['physical_city'])
    df['planning_jurisdiction_codes'] = le.fit_transform(df['planning_jurisdiction'])
    df['physical_zip_code_codes'] = le.fit_transform(df['physical_zip_code'])

    # columns to drop
    drop_cols = [
        'street_name', 'address', 'street_type', 'deed_date', 'street_number',
        'physical_city', 'planning_jurisdiction', 'physical_zip_code', # already comverted into numeric with LabelEncoder
        'wake_supply_index', 'wake_demand_index', # incomplete data
        'deed_date', # date feature
        'land_sale_price', 'total_sale_price', # y features
        ]

    df_drop_cols_full = df.drop(columns=drop_cols)
    df_drop_cols_isCOVID = df_drop_cols_full[df_drop_cols_full['isCovid'] == 1]
    df_drop_cols_notCOVID = df_drop_cols_full[df_drop_cols_full['isCovid'] == 0]
    col_names = df_drop_cols_full.columns.to_list()

    # scale data to unit variance
    scaler = StandardScaler()

    df_scaled_full = pd.DataFrame(scaler.fit_transform(df_drop_cols_full), columns=col_names)
    df_scaled_isCOVID = pd.DataFrame(scaler.fit_transform(df_drop_cols_isCOVID), columns=col_names)
    df_scaled_notCOVID = pd.DataFrame(scaler.fit_transform(df_drop_cols_notCOVID), columns=col_names)

    print(df_drop_cols_full.columns.to_list())

    return df_scaled_full, df_scaled_isCOVID, df_scaled_notCOVID

