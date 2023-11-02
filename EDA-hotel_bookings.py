# -*- coding: utf-8 -*-
"""
Hotel Booking project — Exploratory Data Analysis
https://medium.com/@ethan.duong1120/hotel-booking-project-exploratory-data-analysis-48bcfb7ae7cd
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
"""

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

#read data using read_csv
df = pd.read_csv('E:\ARTI_INT\Code\EDA-HotelBooking\hotel_bookings.csv')
df.head()

#handle missing data
for i in df.columns:
  if df[i].isna().sum() != 0:
    print('null values in', i, 'column :', df[i].isna().sum() ) 
    
# run len(df.index): we can see that our dataset contains 119390 rows
# Most rows in company columns are missing -> we will drop the whole column
df = df.drop(columns = 'company')

# We can drop 4 rows containing null values from chidren column
# It will not affect our result too much 
df = df.dropna(subset = ['children'])

# For country we will fill missing values with the mode frequent value.
df['country'].fillna(df['country'].mode()[0], inplace = True)

#lastly, for agent column will will fill 9 for every null values. 
# this will represent agent ID 
df['agent'].fillna(0, inplace = True)

# Convert columns values
#inplace = true -> directly modify the dataframe.
#inplace = false -> Creating a new df (default)
#replace TA/TO with Undefined in distribution_channel column
df['distribution_channel'].replace("TA/TO", "Undefined", inplace = True)

#replace Undefined, BB, FB, HB, SC to its meaning. 
df['meal'].replace(['Undefined', 'BB', 'FB', 'HB', 'SC'], 
                   [ 'No Meal', 'Breakfast', 'Full Board', 'Hald Board', 'No Meal'],
                   inplace = True)


#Changing datatypes
#turn column into int data type
df['children'].astype(int)
df['agent'].astype(int)

#turn column into datetime data type
pd.to_datetime(df['reservation_status_date'])

#Handle duplicates
df.duplicated().sum() # -> 32020 duplicated rows
df.drop_duplicates(inplace = True)

#Create new columns by combining other columns
#create total night column
df['total_night'] = df['stays_in_weekend_nights'] + df['stays_in_weekend_nights']

#convert month name to number then create new arrival date column by combining year month date
df['arrival_date_month'] = pd.to_datetime(df['arrival_date_month']).dt.month
df['arrival_date'] = pd.to_datetime(dict(year=df.arrival_date_year, month=df.arrival_date_month, day=df.arrival_date_day_of_month))

#Drop unnecessary columns
columns_to_drop = ['stays_in_weekend_nights','stays_in_week_nights', 
'booking_changes','deposit_type','adr','arrival_date_year',
'arrival_date_month','arrival_date_day_of_month','arrival_date_week_number']
#drop
df.drop(columns = columns_to_drop, inplace = True)


#Descriptive Analysis and Correlations
df.describe()

#Correlation heatmap - Type of plot that visualize the strength of relationships between numerical variables. 
plt.rcParams['figure.figsize'] =(12, 6)
sns.heatmap(df.corr(), annot=True, cmap='Reds', linewidths=5)
plt.suptitle('Correlation Between Variables', fontweight='heavy', 
             x=0.03, y=0.98, ha = "left", fontsize='18', 
             fontfamily='sans-serif', color= "black")


# Exploratory Data Analysis
df.to_csv('result.csv', index=False)
