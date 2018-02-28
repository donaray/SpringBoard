#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from IPython.display import display

#Read the dataset
df_acc_all = pd.read_csv('/Users/dona/Documents/DonaRay/DataIncubator/project/Acc.csv', index_col='Accident_Index')
df_veh_all = pd.read_csv('/Users/dona/Documents/DonaRay/DataIncubator/project/Veh.csv', index_col='Accident_Index')
df_cas_all = pd.read_csv('/Users/dona/Documents/DonaRay/DataIncubator/project/Cas.csv', index_col='Accident_Index')


#Drop rows with missing/nan values
df_acc_sub = df_acc_all.drop(['LSOA_of_Accident_Location', 'Location_Northing_OSGR', 'Location_Easting_OSGR', 'Longitude', 'Latitude'], axis=1)
df_acc = df_acc_sub.dropna()

df_merge_acc_veh = pd.merge(df_acc, df_veh_all, left_index=True, right_index=True)

#Convert each date to a datetime object

for date, time in zip(df_acc.Date, df_acc.Time):
  date_obj = datetime.strptime(date, '%d/%m/%Y')
  date = date_obj
  time_obj = datetime.strptime(time, '%H:%M')
  time = time_obj


df_cas_freq = df_cas_all.drop(['Age_of_Casualty'], axis=1)
df_cas_classFreq = df_cas_all[['Age_of_Casualty']]

def Print_Stats(df, num):
    #Prints basic summary statistics of a pandas dataframe    

    print(df.head(num))
    print(df.shape)
    print(df.info())
    print(df.columns)
    print(df.describe())
    
    return None


def Print_Frequency(df, bins = []):
    #Prints frequency distribution
    
    col_list = list(df.columns.values)
    
    if bins == []:
        for col in col_list:
            print(df[col].value_counts())
    else:
        for col in col_list:
            print(pd.cut(df[col],bins).value_counts())


    #print(df.apply(pd.Series.value_counts))

    return None


def Plot_features(df):
    #Plots scatter plots of colmns

    df.plot()
    plt.show()

    return None

#df_cas_plot = df_cas_all[['Age_of_Casualty']]

#Plot_features(df_cas_plot)

df_acc_freq = df_acc[['Accident_Severity', 'Number_of_Vehicles', 'Number_of_Casualties', 'Day_of_Week', 'Road_Type', 'Speed_limit', 'Junction_Detail',
                       'Junction_Control', 'Pedestrian_Crossing-Human_Control', 'Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions',
                        'Weather_Conditions', 'Road_Surface_Conditions', 'Special_Conditions_at_Site', 'Carriageway_Hazards', 'Urban_or_Rural_Area',
                        'Did_Police_Officer_Attend_Scene_of_Accident']] 

df_veh_freq = df_veh_all[['Vehicle_Reference', 'Towing_and_Articulation', 'Vehicle_Manoeuvre', 'Vehicle_Location-Restricted_Lane', 'Junction_Location',
                          'Skidding_and_Overturning', 'Hit_Object_in_Carriageway', 'Vehicle_Leaving_Carriageway', 'Hit_Object_off_Carriageway',
                          'Journey_Purpose_of_Driver', 'Sex_of_Driver', 'Age_Band_of_Driver', 'Driver_Home_Area_Type']]


df_cas_freq = df_cas_all[['Casualty_Class', 'Sex_of_Casualty', 'Age_Band_of_Casualty', 'Casualty_Severity', 'Pedestrian_Location',
                          'Pedestrian_Movement', 'Car_Passenger', 'Bus_or_Coach_Passenger', 'Pedestrian_Road_Maintenance_Worker',
                          'Casualty_Type', 'Casualty_Home_Area_Type']]
    
bins = [-10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]    

#Print_Frequency(df_cas_freq)
#Print_Frequency(df_cas_classFreq, bins)
#Print_Frequency(df_acc_freq)
#Print_Frequency(df_veh_freq)
Print_Frequency(df_cas_freq)


#Print_Stats(df_acc, 0)
#Print_Stats(df_acc_sub, 0)
#Print_Stats(df_acc_all, 0)
#Print_Stats(df_veh_all, 10)
#Print_Stats(df_cas_all, 0)

#Print_Frequency(df_cas_freq)


#df_numeric_acc = df_acc._get_numeric_data()



#df_categorical = df_acc.select_dtypes(['category'])
#df_categorical.drop(['Accident_Index'])
#print(df_categorical.apply(pd.value_counts))

