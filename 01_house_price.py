import pandas as pd
from geopy.geocoders import Nominatim

house_file_path = "assets\house_price.csv"
used_columns_list = ['Land_Sale_Price', 'Total_sale_Price', 'Deed_Date', 'Assessed_Building_Value', 'Story_Height', 'HEATED_AREA',
       'UTILITIES', 'Remodeled_Year', 'BATH', 'BATH_FIXTURES', 'TYPE_AND_USE', 'PHYSICAL_ZIP_CODE', 'PHYSICAL_CITY', 'Street_Number', 
       'Street_Name', 'Street_Type', 'Planning_Jurisdiction']
wake_cities = ['APEX', 'CARY', 'FUQUAY VARINA', 'GARNER', 'HOLLY SPRINGS', 'KNIGHTDALE', 'MORRISVILLE', 'RALEIGH', 'ROLESVILLE', 'WAKE FOREST', 'WENDELL', 'ZEBULON']

column_dropna = ['UTILITIES', 'BATH', 'Story_Height']

update_location = {'5500 BATOUL LN, raleigh': '27606',
                   '1436 INDIGO CREEK DR, zebulon': '27597',
                   '1432 INDIGO CREEK DR, zebulon': '27597',
                   '1428 INDIGO CREEK DR, zebulon': '27597',
                   '1424 INDIGO CREEK DR, zebulon': '27597',
                   '1420 INDIGO CREEK DR, zebulon': '27597',
                   '1416 INDIGO CREEK DR, zebulon': '27597',
                   '1412 INDIGO CREEK DR, zebulon': '27597',
                   '1413 INDIGO CREEK DR, zebulon': '27597',
                   '1429 INDIGO CREEK DR, zebulon': '27597',
                   '1433 INDIGO CREEK DR, zebulon': '27597',
                   '1517 INDIGO CREEK DR, zebulon': '27597',
                   '1519 INDIGO CREEK DR, zebulon': '27597',
                   '1521 INDIGO CREEK DR, zebulon': '27597',
                   '1523 INDIGO CREEK DR, zebulon': '27597',
                   '1520 INDIGO CREEK DR, zebulon': '27597',
                   '1518 INDIGO CREEK DR, zebulon': '27597',
                   '1516 INDIGO CREEK DR, zebulon': '27597',
                   '529 BASIN HILL DR, wake forest': '27587',
                   '505 BASIN HILL DR, wake forest': '27587',
                   '501 BASIN HILL DR, wake forest': '27587'}

class get_house_price_df:
    def __init__(self):
        pass
    
    def read_csv(file_path, print_columns=False):
        """Read csv files and return a dataframe
        
        Args:
            file_path (str): path of the csv file
            print_columns (bool): print columns of the dataframe (default: False)
        """
        df = pd.read_csv(file_path)
        if print_columns:
            print(df.columns)
        print(df.shape)
        print("The data is loaded successfully!")
        return df
    
class keep_used_columns:
    def __init__(self, df, columns):
        self.df = df
        self.columns = columns
    def used_columns_df(self):
        global df
        """Keep only the used columns
        
        Args:
            df (dataframe): dataframe
            columns (list): list of columns
        """
        new_df = df[self.columns].copy()
        return new_df

class data_format:
    def __init__(self, df) -> None:
        self.df = df
    def format_price_value(self,column, convert_type, replace=False):
        global df
        """Convert price value to float
        
        Args:
            df (dataframe): dataframe
            column (str): column name
            convert_type (str): convert type
            replace (bool): replace the original column or not (default: False)
        """
        if replace:
            df[column] = df[column].str.replace(',', '').astype(convert_type)
        df[column] = df[column].astype(convert_type)
        print(f"{column} is converted successfully")
        return df

    def format_date(self, column, errors='coerce'):
        global df
        """Convert format of date
        
        Args:
            df (dataframe): dataframe
            column (str): column name
            errors (str): errors (default: 'coerce')
        """
        df[column] = pd.to_datetime(df[column], errors=errors)
        print(f"{column} is converted successfully")
        return df

class find_zipcode:
    def __init__(self, df) -> None:
        self.df = df
    
    def find_zipcode(self, update_location):
        global df
        df.loc[(df['PHYSICAL_ZIP_CODE'] == 0) & (df['Planning_Jurisdiction'] == 'CA'), 'PHYSICAL_CITY'] = 'cary'
        df.loc[(df['PHYSICAL_ZIP_CODE'] == 0) & (df['Planning_Jurisdiction'] == 'RA'), 'PHYSICAL_CITY'] = 'raleigh'
        df.loc[(df['PHYSICAL_ZIP_CODE'] == 0) & (df['Planning_Jurisdiction'] == 'WE'), 'PHYSICAL_CITY'] = 'wendell'
        df.loc[(df['PHYSICAL_ZIP_CODE'] == 0) & (df['Planning_Jurisdiction'] == 'ZB'), 'PHYSICAL_CITY'] = 'zebulon'
        df.loc[(df['PHYSICAL_ZIP_CODE'] == 0) & (df['Planning_Jurisdiction'] == 'WF'), 'PHYSICAL_CITY'] = 'wake forest'
        df['address'] = df['Street_Number'].astype(str) + " " + df['Street_Name'] + " " + \
            df['Street_Type'] + ", " + df['PHYSICAL_CITY']

        geolocator = Nominatim(user_agent="myGeocoder")

        addresses = df[df['PHYSICAL_ZIP_CODE'] == 0]['address'].tolist()
        location_dict = {}
        location_lst = []
        for address in addresses:
            location_dict[address] = None
            location = geolocator.geocode(address, timeout=None)
            if location != None:
                location_dict[address] = location.address.split(',')[-2]
            else: 
                print(f"{address} is not found")
        non_location = {}
        for address, zip in location_dict.items():
            if location_dict[address] is None:
                # print(address)
                non_location[address] = None
        location_dict.update(update_location)
        print("Updating missing zipcodes data...")

        for key, value in location_dict.items():
            df.loc[df['address'] == key, 'PHYSICAL_ZIP_CODE'] = value
        print("ALL missing zipcodes are found successfully")
        return df

class drop_missing_value:
    def __init__(self, df) -> None:
        self.df = df
    def fill_drop_na(self, column_dropna, column, fill_zero=True):
        global df
        """Fill or drop na values
        
        Args:
            df (dataframe): dataframe
            column (str): column name
            fill_zero (bool): fill na as zero or not (default: True)
        """
        if column == 'Deed_Date':
            df.loc[df[column].isnull(), column] = df['Remodeled_Year']
        elif column in column_dropna:
            df.dropna(subset=[column], inplace=True)
            df.reset_index(drop=True, inplace=True)
            print(f"None value in {column} is dropped successfully")
            print("Index of the dataframe is reset successfully")
            return df
        else:
            if fill_zero:
                df[column] = df[column].fillna(0).astype(int)
        print(f"Number of nan values of {column} is {df[column].isnull().sum()}")
        return df

    def remove_zero(self, column):
        global df
        """Remove zero values
        
        Args:
            df (dataframe): dataframe
            column (str): column name
        """
        df = df[df[column] != 0]
        print(f"Number of zero values of {column} is {df[column].eq(0).sum()}")
        return df

class convert_cat_to_num:
    def __init__(self, df) -> None:
        self.df = df
    def convert_categorical_to_numeric_variables(self, variable):
        global df
        """Convert categorical variables to numeric variables
        Bath variable:
            A: 1
            B: 1.5
            C: 2
            D: 2.5
            E: 3
            F: 3.5
            G: limited plbg
            H: no plumbing
            I: adequate
            J: no of fixtures
        
        Args:
            df (dataframe): dataframe
            variable (str): variable name
        """
        if variable == 'BATH':
            df.loc[df['BATH'] == 'A', 'BATH'] = 1
            df.loc[df['BATH'] == 'B', 'BATH'] = 1.5
            df.loc[df['BATH'] == 'C', 'BATH'] = 2
            df.loc[df['BATH'] == 'D', 'BATH'] = 2.5
            df.loc[df['BATH'] == 'E', 'BATH'] = 3
            df.loc[df['BATH'] == 'F', 'BATH'] = 3.5
            df.loc[df['BATH'] == 'G', 'BATH'] = 0
            df.loc[df['BATH'] == 'H', 'BATH'] = 0
            df.loc[(df['BATH'] == 'I') & (df['BATH_FIXTURES'] <= 3), 'BATH'] = 0
            df.loc[(df['BATH'] == 'I') & (df['BATH_FIXTURES'] <= 6), 'BATH'] = 1
            df.loc[(df['BATH'] == 'I') & (df['BATH_FIXTURES'] == 7), 'BATH'] = 1.5
            df.loc[(df['BATH'] == 'I') & (df['BATH_FIXTURES'] == 8), 'BATH'] = 2
            df.loc[(df['BATH'] == 'I') & (df['BATH_FIXTURES'] == 9), 'BATH'] = 2.5
            df.loc[(df['BATH'] == 'I') & (df['BATH_FIXTURES'] == 10), 'BATH'] = 3
            df.loc[(df['BATH'] == 'I') & (df['BATH_FIXTURES'] == 11), 'BATH'] = 3.5
            df.loc[(df['BATH'] == 'I') & (df['BATH_FIXTURES'] > 11), 'BATH'] = 4

            df.loc[(df['BATH'] == 'J') & (df['BATH_FIXTURES'] <= 3), 'BATH'] = 0
            df.loc[(df['BATH'] == 'J') & (df['BATH_FIXTURES'] <= 6), 'BATH'] = 1
            df.loc[(df['BATH'] == 'J') & (df['BATH_FIXTURES'] == 7), 'BATH'] = 1.5
            df.loc[(df['BATH'] == 'J') & (df['BATH_FIXTURES'] == 8), 'BATH'] = 2
            df.loc[(df['BATH'] == 'J') & (df['BATH_FIXTURES'] == 9), 'BATH'] = 2.5
            df.loc[(df['BATH'] == 'J') & (df['BATH_FIXTURES'] == 10), 'BATH'] = 3
            df.loc[(df['BATH'] == 'J') & (df['BATH_FIXTURES'] == 11), 'BATH'] = 3.5
            df.loc[(df['BATH'] == 'J') & (df['BATH_FIXTURES'] > 11), 'BATH'] = 4
            print("Bathroom number is converted successfully")
            return df

        elif variable == 'Story_Height':
            df.loc[df['Story_Height'] == 'A', 'Story_Height'] = 1
            df.loc[df['Story_Height'] == 'B', 'Story_Height'] = 1.5
            df.loc[df['Story_Height'] == 'C', 'Story_Height'] = 2
            df.loc[df['Story_Height'] == 'D', 'Story_Height'] = 2.5
            df.loc[df['Story_Height'] == 'E', 'Story_Height'] = 3
            df.loc[df['Story_Height'] == 'F', 'Story_Height'] = 3.5
            df.loc[df['Story_Height'] == 'G', 'Story_Height'] = 4
            df.loc[df['Story_Height'] == 'H', 'Story_Height'] = 5
            df.loc[df['Story_Height'] == 'I', 'Story_Height'] = 1.75
            df.loc[df['Story_Height'] == 'J', 'Story_Height'] = 1.4
            df.loc[df['Story_Height'] == 'K', 'Story_Height'] = 1.63
            df.loc[df['Story_Height'] == 'L', 'Story_Height'] = 1.88
            df.loc[df['Story_Height'] == 'M', 'Story_Height'] = 2.4
            df.loc[df['Story_Height'] == 'N', 'Story_Height'] = 2.63
            df.loc[df['Story_Height'] == 'O', 'Story_Height'] = 2.75
            print("Story height is converted successfully")
            return df
    
class filter_column:
    def __init__(self, df, filter_column) -> None:
        self.df = df
        self.filter_column = filter_column
    def filter_column(self, city_list=None, filter_date=None):
        global df
        """Filter dataframe by city and date
        
        Args:
            df (dataframe): dataframe
            filter_column (str): filter column
            city_list (list): list of cities (default: None)
            filter_date (str): filter date (default: None)
        """
        if filter_column == 'TYPE_AND_USE':
            # According to the U.S. Census Bureau, a single-family house is one that may be fully detached, semi-detached, a row house or a townhome. df.loc[df['column_name'].isin(some_values)]
            df = df.loc[df[filter_column].isin([1, 8])]
            print(f"{filter_column} is filtered successfully")
        elif filter_column == 'PHYSICAL_CITY':
            if city_list is None:
                print("Please provide city list")
                print("Stop filtering")
            else:
                df = df.drop(df[~df[filter_column].isin(city_list)].index)
                df[filter_column] = df[filter_column].str.lower()
                print(f"{filter_column} is filtered successfully")
        elif filter_column == 'Deed_Date':
            if filter_date is None:
                print("Please provide date")
                print("Stop filtering")
            else:
                df = df.loc[df[filter_column] > filter_date]
                df.drop(columns=['Remodeled_Year', ], inplace=True)
                df.reset_index(drop=True, inplace=True)
                print(f"{filter_column} is filtered successfully")
                print("Remodeled_Year is removed")
                print("Index of dataframe is reset")
        return df

def main(df):
    df_start = keep_used_columns.used_columns_df(df, used_columns_list)
    df_1 = data_format.format_date(df_start, 'Remodeled_Year')
    df_2 = data_format.format_date(df_1, 'Deed_Date')
    df_3 = drop_missing_value.fill_drop_na(df_2, column_dropna, 'PHYSICAL_ZIP_CODE')
    df_4 = drop_missing_value.fill_drop_na(df_3, column_dropna, 'HEATED_AREA')
    df_5 = data_format.format_price_value(df_4, 'Land_Sale_Price', 
                                        'float', replace=True)
    df_6 = data_format.format_price_value(df_5, 'Total_sale_Price', 
                                        'float', replace=True)
    df_7 = data_format.format_price_value(df_6, 'Assessed_Building_Value', 
                                        'float', replace=True)
    df_8 = convert_cat_to_num.convert_categorical_to_numeric_variables(df_7, 'BATH')
    df_9 = convert_cat_to_num.convert_categorical_to_numeric_variables(df_8, 'Story_Height')
    df_10 = filter_column.filter_column(df_9, 'TYPE_AND_USE')
    df_11 = filter_column.filter_column(df_10, 'PHYSICAL_CITY', wake_cities)
    df_12 = drop_missing_value.remove_zero(df_11, 'Total_sale_Price')
    df_13 = filter_column.filter_column(df_12,  'Deed_Date', filter_date="2000-01-01")
    df_14 = drop_missing_value.fill_drop_na(df_13, column_dropna, 'UTILITIES')
    df_15 = drop_missing_value.fill_drop_na(df_14, column_dropna, 'BATH')
    df_16 = drop_missing_value.fill_drop_na(df_15, column_dropna, 'Story_Height')
    df_17 = find_zipcode.find_zipcode(df_16, update_location)
    print("Pre-processing data is done.")
    df_17.to_csv('assets/updated_house_price.csv')
    print("The data is saved as a CSV file, called updated_house_price, please check assets folder.")
    pass

if __name__ == '__main__':
    print("Pre-processing data...")
    df = get_house_price_df.read_csv(file_path=house_file_path)
    main(df)
