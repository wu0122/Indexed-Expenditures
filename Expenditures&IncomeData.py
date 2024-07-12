#%%
#CPI DATA
print("hello")

import pandas as pd 
import pip 
import openpyxl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Use Raw String
excel_file_path2 = r'C:\Users\Owner\Downloads\CPIAUCSL.csv'

df = pd.read_csv(excel_file_path2)
df['Year'] = pd.to_datetime(df['DATE']).dt.year
df.drop(columns=['DATE'], inplace=True)
# Renaming the 'CPIAUCSL' column to 'CPI'
#df.rename(columns={'CPIAUCSL': 'Index'}, inplace=True)

# Reordering the columns to have 'Year' first
df = df[['Year', 'CPIAUCSL']]

# Displaying the DataFrame
print(df)

x_CPI = df['Year'].values.reshape(-1,1)
y_CPI = df['CPIAUCSL'].values

model = LinearRegression()
model.fit(x_CPI, y_CPI)

future_x_CPI = np.arange(1980,2036).reshape(-1,1)
future_y_CPI = model.predict(future_x_CPI)

poly_features = PolynomialFeatures(degree=3)
CPIpoly = poly_features.fit_transform(x_CPI)
model.fit(CPIpoly, y_CPI)

futureCPIPoly = poly_features.transform(future_x_CPI)
futureCPIPred = model.predict(futureCPIPoly)

plt.plot(future_x_CPI, futureCPIPred)
plt.plot(x_CPI,y_CPI, color='red')
plt.show()

# %%
#ELECTRICITY DATA
print("hello")

import pandas as pd 
import pip 
import openpyxl
import matplotlib.pyplot as plt

# Use Raw String
excel_file_path = r'C:\Users\Owner\Downloads\SeriesReport-20240129204703_16d30d - 2.xlsx'

df1 = pd.read_excel(excel_file_path, skiprows=9)

excel_file_path2 = r'C:\Users\Owner\Downloads\CPIAUCSL.csv'

base_year = 1980
df

df1 = pd.merge(df1, df, on='Year', how='left')

base_cpi = df1[df1['Year'] == base_year]['CPIAUCSL'].values[0]
base_cpi = 307.917



df1['Electricity'] = df1['Jan'] * (base_cpi / df1['CPIAUCSL'])

df1 = df1.drop(columns= ['CPIAUCSL', 'Jan'])


x_Electricity = df1['Year'].values.reshape(-1,1)
y_Electricity = df1['Electricity'].values

model = LinearRegression()
model.fit(x_Electricity, y_Electricity)
y_pred = model.predict(x_Electricity)

future_x_Electricity = np.arange(1980,2035).reshape(-1,1)
future_y_Electricity = model.predict(future_x_Electricity)


# %%
#GROCERIES DATA

# Use Raw String
excel_file_path = r'C:\Users\Owner\Downloads\CUSR0000SAF11.xlsx'

df2 = pd.read_excel(excel_file_path, skiprows=10)

df2['Year'] = pd.to_datetime(df2['observation_date']).dt.year
df2['Month'] = pd.to_datetime(df2['observation_date']).dt.month

df2 = (
    df2[df2['Month'] == 1] # Just filter to January observations for each year
    .drop(columns='Month') # Drop unnecessary month column
    .reset_index(drop=True) # Reset index to only have one observation per year
)
df2.drop(columns=['observation_date'], inplace=True)

df2.rename(columns={'CUSR0000SAF11': 'Groceries'}, inplace=True)

# Reordering the columns to have 'Year' first
df2 = df2[['Year', 'Groceries']]

x_Groceries = df2['Year'].values.reshape(-1,1)
y_Groceries = df2['Groceries'].values

model = LinearRegression()
model.fit(x_Groceries, y_Groceries)
y_pred = model.predict(x_Groceries)

future_x_Groceries = np.arange(1980,2035).reshape(-1,1)
future_y_Groceries = model.predict(future_x_Groceries)

# %%

#HEALTHCARE COSTS

excel_file_path = r'C:\Users\Owner\Downloads\data-INvHR.xlsx'

df3 = pd.read_excel(excel_file_path, usecols=lambda x: x not in ['column2'])

df3 = (
    df3[df3['Year'] >= 1980] # Just filter to January observations for each year
)
df3

base_year = 1980  # Change this to the desired base year
base_costs = df3[df3['Year'] == base_year]['Constant 2022 dollars'].values[0]

df3['Healthcare'] = (df3['Constant 2022 dollars'] / base_costs) * 100

df3.drop(columns=['Constant 2022 dollars'], inplace=True)
df3.drop(columns= ['Total national health expenditures'], inplace=True)

#df3.rename(columns={'Index': 'Healthcare Expenditure'}, inplace=True)

print(df3)

x_Healthcare = df3['Year'].values.reshape(-1,1)
y_Healthcare = df3['Healthcare'].values

model = LinearRegression()
model.fit(x_Healthcare, y_Healthcare)
y_pred = model.predict(x_Healthcare)

future_x_Healthcare = np.arange(1980,2035).reshape(-1,1)
future_y_Healthcare = model.predict(future_x_Healthcare)



# %%
# RENT COSTS

excel_file_path = r'C:\Users\Owner\Downloads\CUSR0000SEHA.xlsx'

df4 = pd.read_excel(excel_file_path,skiprows=10)
df4['Year'] = pd.to_datetime(df4['observation_date']).dt.year
df4


base_year = 1981

base_costs = df4[df4['Year'] == base_year]['CUSR0000SEHA'].values[0]

df4['Rent'] = (df4['CUSR0000SEHA'] / base_costs) * 100

#df4.rename(columns={'Index': 'Rent'}, inplace=True)
df4.drop(columns=['observation_date'], inplace=True)
df4 = df4[['Year', 'Rent']]

print(df4)

x_Rent = df4['Year'].values.reshape(-1,1)
y_Rent = df4['Rent'].values

model = LinearRegression()
model.fit(x_Rent, y_Rent)
y_pred = model.predict(x_Rent)

future_x_Rent = np.arange(1980,2035).reshape(-1,1)
future_y_Rent = model.predict(future_x_Rent)

# %%
# HOUSING COSTS
excel_file_path = r'C:\Users\Owner\Downloads\USSTHPI.xlsx'

df5 = pd.read_excel(excel_file_path,skiprows=10)
df5['Year'] = pd.to_datetime(df5['observation_date']).dt.year
df5


base_year = 1975
df5.drop(columns=['observation_date'], inplace=True)
df5.rename(columns={'USSTHPI': 'Housing'}, inplace=True)
df5 = df5[['Year', 'Housing']]
print(df5)

x_Housing = df5['Year'].values.reshape(-1,1)
y_Housing = df5['Housing'].values

model = LinearRegression()
model.fit(x_Housing, y_Housing)
y_pred = model.predict(x_Housing)

future_x_Housing = np.arange(1980,2035).reshape(-1,1)
future_y_Housing = model.predict(future_x_Housing)

recent_data = df5[(df5['Year'] >= 2015) & (df['Year'] <= 2024)]

future_x_Housing2 = recent_data[['Year']]
future_y_Housing2 = recent_data['Housing']

model_recent = LinearRegression()
model_recent.fit(future_x_Housing2, future_y_Housing2)

future_x_Housing2 = np.arange(2015,2035).reshape(-1,1)
future_y_Housing2 = model_recent.predict(future_x_Housing2)






# %%
# MEDIAN INCOME
excel_file_path = r'C:\Users\Owner\Downloads\MEPAINUSA672N.xlsx'

df6 = pd.read_excel(excel_file_path,skiprows=10)
df6['Year'] = pd.to_datetime(df6['observation_date']).dt.year
base_year = 1980

base_costs = df6[df6['Year'] == base_year]['MEPAINUSA672N'].values[0]

df6['Median Income'] = (df6['MEPAINUSA672N'] / base_costs) * 100




df6.drop(columns=['observation_date'], inplace=True)
df6.drop(columns=['MEPAINUSA672N'], inplace=True)
#df6.rename(columns={'Index': 'Median Income'}, inplace=True)
df6 = df6[['Year', 'Median Income']]

print(df6)

x_MedianIncome = df6['Year'].values.reshape(-1,1)
y_MedianIncome = df6['Median Income'].values

model = LinearRegression()
model.fit(x_MedianIncome, y_MedianIncome)
y_pred = model.predict(x_MedianIncome)

future_x_MedianIncome = np.arange(1980,2035).reshape(-1,1)
future_y_MedianIncome = model.predict(future_x_MedianIncome)



# %%
# Natural gas

excel_file_path = r'C:\Users\Owner\Downloads\DGHERG3A086NBEA.xlsx'

df8 = pd.read_excel(excel_file_path, skiprows=10)

df8['Year'] = pd.to_datetime(df8['observation_date']).dt.year

base_year = 1980  # Change this to the desired base year
base_costs = df8[df8['Year'] == base_year]['DGHERG3A086NBEA'].values[0]

df8['Natural Gas'] = (df8['DGHERG3A086NBEA'] / base_costs) * 100
df8.drop(columns=['observation_date'], inplace=True)
df8.drop(columns=['DGHERG3A086NBEA'], inplace=True)
print(df8)

x_NaturalGas = df8['Year'].values.reshape(-1,1)
y_NaturalGas = df8['Natural Gas'].values

model = LinearRegression()
model.fit(x_NaturalGas, y_NaturalGas)
y_pred = model.predict(x_NaturalGas)

future_x_NaturalGas = np.arange(1980,2035).reshape(-1,1)
future_y_NaturalGas = model.predict(future_x_NaturalGas)

recent_data = df5[(df5['Year'] >= 2015) & (df['Year'] <= 2024)]

future_x_Housing2 = recent_data[['Year']]
future_y_Housing2 = recent_data['Housing']

model_recent = LinearRegression()
model_recent.fit(future_x_Housing2, future_y_Housing2)

future_x_Housing2 = np.arange(2015,2035).reshape(-1,1)
future_y_Housing2 = model_recent.predict(future_x_Housing2)

# %%

#TUITION

excel_file_path = r'C:\Users\Owner\Downloads\tabn330.10.xlsx'

df13 = pd.read_excel(excel_file_path, skiprows=18, nrows=41, usecols=[0,1])
df13.columns = ['Year', 'Tuition']
df13['Year'] = df13['Year'].str.extract(r'(\d{4})')


base_year = 1980  # Change this to the desired base year

#base_costs = df13[df13['Year'] == base_year]['Tuition'].values[0]
base_costs = 9421.054975
df13['Index'] = (df13['Tuition'] / base_costs) * 100
df13.drop(columns=['Tuition'], inplace=True)
df13['Year'] = df13['Year'].astype('int32')
print(df13[['Year', 'Index']])


x_Tuition = df13['Year'].values.reshape(-1,1)
y_Tuition = df13['Index'].values

model = LinearRegression()
model.fit(x_Tuition, y_Tuition)
y_pred = model.predict(x_Tuition)

future_x_Tuition = np.arange(1980,2035).reshape(-1,1)
future_y_Tuition = model.predict(future_x_Tuition)
df13


# %%

#TUITION COSTS

excel_file_path = r'C:\Users\Owner\Downloads\SeriesReport-20231220204930_d8c92e.xlsx'

df9 = pd.read_excel(excel_file_path,skiprows=11)

PEGGING_DATE = '1980-01'


# Reformat into one giant series
# Dictionary to map month names to numbers  
month_to_num = {  
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',  
    'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12' , 'HALF1':'HALF1','HALF2':'HALF2' 
}  


df9 = df9.melt(id_vars=['Year'], var_name='Month', value_name='index_value')
df9['Month'] = df9['Month'].map(month_to_num)
df9['yr_mo'] = df9['Year'].astype(str) + '-' + df9['Month']
df9 = df9[['yr_mo', 'index_value']].sort_values('yr_mo').reset_index(drop=True)

# Fix new index to 1980
df9['index_1980'] = df9['index_value'] / df9[df9['yr_mo'] == PEGGING_DATE]['index_value'].values[0] * 100

# If you want to drop the old variable
df9= df9[['yr_mo', 'index_1980']]
df9['Year'] = df9['yr_mo'].str.split("-").apply(lambda x: x[0])
df9 = (
    df9
    .rename(columns={'index_1980':'Tuition'})
    [(df9['yr_mo'].str.contains('-01')) &
     (df9['Year'] >= '1980')]
    [['Year', 'Tuition']]
    .reset_index(drop=True)

)

df9['Year'] = df9['Year'].astype(int)
#df9.rename(columns={'Index': 'Tuition Costs'}, inplace=True)

print(df9)

x_Tuition2 = df9['Year'].values.reshape(-1,1)
y_Tuition2 = df9['Tuition'].values

model = LinearRegression()
model.fit(x_Tuition2, y_Tuition2)
y_pred = model.predict(x_Tuition2)

future_x_Tuition2 = np.arange(1980,2035).reshape(-1,1)
future_y_Tuition2 = model.predict(future_x_Tuition2)



# %%

excel_file_path = r'C:\Users\Owner\Downloads\DVEDRC1A027NBEA.xlsx'

df12 = pd.read_excel(excel_file_path,skiprows=10)
df12['Year'] = pd.to_datetime(df12['observation_date']).dt.year


base_year = 1984

base_costs = df12[df12['Year'] == base_year]['DVEDRC1A027NBEA'].values[0]

df12['Vocational School Tuition Cost'] = (df12['DVEDRC1A027NBEA'] / base_costs) * 100
df12.drop(columns=['observation_date'], inplace=True)
df12.drop(columns=['DVEDRC1A027NBEA'], inplace=True)
print(df12)


# %%

# Merge the two DataFrames on the 'Year' column
combined_df = pd.merge(df, df1, on='Year')
combined_df = pd.merge(combined_df, df2, on='Year')
combined_df = pd.merge(combined_df, df3, on= 'Year')
combined_df = pd.merge(combined_df, df4, on= 'Year')
combined_df = pd.merge(combined_df, df5, on= 'Year')
combined_df = pd.merge(combined_df, df6, on= 'Year')

combined_df = pd.merge(combined_df, df8, on= 'Year')
combined_df = pd.merge(combined_df, df12, on= 'Year')
combined_df = pd.merge(combined_df, df13, on= 'Year')


# Displaying the merged DataFrame
print(combined_df)
# %%

#TOTAL EVERYTHING


plt.figure(figsize=(10, 6))

# Loop through each column and plot a line chart
for column in ["CPI"]:
    plt.plot(combined_df['Year'], combined_df['Healthcare'], label='Healthcare Expenditure', color= 'magenta')  # Plot Healthcare Expenditure
    plt.plot(combined_df['Year'], combined_df['Housing'], label='House Price', color= 'goldenrod')  # Plot Housing
    plt.plot(combined_df['Year'], combined_df['Natural Gas'], label='Natural Gas', color= 'deepskyblue')  # Natural Gas
    plt.plot(combined_df['Year'], combined_df['Index'], label='Undergraduate Tuition Fees', color= 'firebrick')  # Plot Housing
    plt.plot(combined_df['Year'], combined_df['Rent'], label='Rent', color= 'forestgreen')  # Plot Rent
    plt.plot(combined_df['Year'], combined_df['CPIAUCSL'], label='CPI', color= 'royalblue')  # Plot CPI
  
    plt.plot(combined_df['Year'], combined_df['Electricity'], label='Electricity', color= 'violet')  # Plot Electricity

    plt.plot(combined_df['Year'], combined_df['Groceries'], label='Groceries', color= 'purple')  # Plot Groceries
    plt.plot(combined_df['Year'], combined_df['Median Income'], label='Median Income', color= 'black')


plt.title('Expenditures per Capita Indexed to 1980')
plt.xlabel('Month \n source: FRED, BLS, NCES')
plt.ylabel('Values (Indexed = 100 at 1980)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)

plt.show()

# %%

#HOUSING V INCOME

plt.figure(figsize=(10, 6))

for column in ["CPI"]:
    
    plt.plot(combined_df['Year'], combined_df['Housing'], label='House Price', color= 'goldenrod')
    plt.plot(future_x_Housing, future_y_Housing, color= 'goldenrod', linestyle = '--', label = 'Best Fit Housing')
    plt.plot(future_x_Housing2, future_y_Housing2, color= 'goldenrod', linestyle = ':', label = 'Best Fit Housing after 2008 Recession')
    plt.plot(combined_df['Year'], combined_df['Rent'], label='Rent', color= 'forestgreen')  # Plot Rent
    plt.plot(future_x_Rent, future_y_Rent, color= 'forestgreen', linestyle = '--', label = 'Best Fit Rent')
    plt.plot(combined_df['Year'], combined_df['CPIAUCSL'], label='CPI', color= 'royalblue')  # Plot CPI
    plt.plot(future_x_CPI, futureCPIPred, color= 'royalblue', linestyle = '--', label = 'Best Fit CPI')
    plt.plot(combined_df['Year'], combined_df['Median Income'], label='Median Income', color= 'black')
    plt.plot(future_x_MedianIncome, future_y_MedianIncome, color= 'black', linestyle = '--', label = 'Best Fit Median Income')


plt.title('Expenditures per Capita Indexed to 1980, Housing Vs. Income')
plt.xlabel('Year \n source: FRED, BLS, NCES')
plt.ylabel('Values (Indexed = 100 at 1980)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)

plt.show()
# %%

#UTILIES V INCOME

plt.figure(figsize=(10, 6))

for column in ["CPI"]:

    plt.plot(combined_df['Year'], combined_df['Median Income'], label='Median Income', color= 'black')
    plt.plot(future_x_MedianIncome, future_y_MedianIncome, color= 'black', linestyle = '--', label = 'Best Fit Median Income')
    
    plt.plot(combined_df['Year'], combined_df['Natural Gas'], label='Natural Gas', color= 'deepskyblue')  # Natural Gas
    plt.plot(future_x_NaturalGas, future_y_NaturalGas, color= 'deepskyblue', linestyle = '--', label = 'Best Fit Natural Gas')
    
    plt.plot(combined_df['Year'], combined_df['CPIAUCSL'], label='CPI', color= 'royalblue')  # Plot CPI
    plt.plot(future_x_CPI, future_y_CPI, color= 'royalblue', linestyle = '--', label = 'Best Fit CPI')
    
    plt.plot(combined_df['Year'], combined_df['Electricity'], label='Electricity', color= 'violet')  # Plot Electricity
    plt.plot(future_x_Electricity, future_y_Electricity, color= 'violet', linestyle = '--', label = 'Best Fit Electricity')
    plt.plot(combined_df['Year'], combined_df['Groceries'], label='Groceries', color= 'purple')  # Plot Groceries
    plt.plot(future_x_Groceries, future_y_Groceries, color= 'purple', linestyle = '--', label = 'Best Fit Groceries')


plt.title('Expenditures per Capita Indexed to 1980-1984, Utilities Vs. Income')
plt.xlabel('Month \n source: FRED, BLS')
plt.ylabel('Values (Indexed = 100 at 1980-1984)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)

plt.show()

# %%
plt.figure(figsize=(10, 6))
# TUITION & HEALTHCARE V INCOME



for column in ["CPI"]:
    plt.plot(combined_df['Year'], combined_df['Index'], label='Undergraduate Tuition Fees', color= 'firebrick')  
    plt.plot(future_x_Tuition, future_y_Tuition, color= 'firebrick', linestyle = '--', label = 'Best Fit Tuition')
    plt.plot(combined_df['Year'], combined_df['Healthcare'], label='Healthcare Expenditure', color= 'magenta')  # Plot Healthcare Expenditure
    plt.plot(future_x_Healthcare, future_y_Healthcare, color= 'magenta', linestyle = '--', label = 'Best Fit Healthcare')
    plt.plot(combined_df['Year'], combined_df['CPIAUCSL'], label='CPI', color= 'royalblue')  # Plot CPI
    plt.plot(future_x_CPI, future_y_CPI, color= 'royalblue', linestyle = '--', label = 'Best Fit CPI')
    plt.plot(combined_df['Year'], combined_df['Median Income'], label='Median Income', color= 'black')
    plt.plot(future_x_MedianIncome, future_y_MedianIncome, color= 'black', linestyle = '--', label = 'Best Fit Median Income')
    

plt.title('Expenditures per Capita Indexed to 1980, Tuition & Healthcare Vs. Income')
plt.xlabel('Month \n source: FRED, BLS')
plt.ylabel('Values (Indexed = 100 at 1980)')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)

plt.show()
# %%
