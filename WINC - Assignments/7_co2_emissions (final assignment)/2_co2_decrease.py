### Please formulate an answer to the following question in your report:
###     2. Which countries are making the biggest strides in (relative) decreasing CO2 output?

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Using datatables in Google Colab (sorting columns: looking at .head and .tail values)
# %load_ext google.colab.data_table
# %matplotlib inline

# pd.options.display.max_columns = 50   # Allows to view more columns (default max = 20)
# pd.options.display.max_rows = 10000   # Allows to view more rows (default max = 20000)

#------------------------------------------------------------------------------------------------------------------------------------#

### CO2 CHANGE ANALYSIS
# Read and filter data
df_co2_per_capita = pd.read_csv("data/co2-per-capita.csv")  # Most recent year = 2021
df_co2_per_capita.dropna(subset=['Code'], inplace=True)
df_co2_per_capita = df_co2_per_capita.drop(df_co2_per_capita[df_co2_per_capita['Entity'].isin(['World'])].index)
df_co2_per_capita.drop(['Code'], axis=1, inplace=True)

# Calculate the annual percentage change in CO2 emissions per capita
df_co2_per_capita['Percentage change - CO₂ emissions (per capita)'] = df_co2_per_capita.groupby('Entity')['Annual CO₂ emissions (per capita)'].pct_change(periods=1) * 100

# Calculate the absolute change in CO2 emissions per capita
df_co2_per_capita['Absolute change - CO₂ emissions (per capita)'] = df_co2_per_capita.groupby('Entity')['Annual CO₂ emissions (per capita)'].diff()
df_co2_per_capita.dropna(inplace=True)

#------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------#

### VISUALS

## BAR CHART (RELATIVE CHANGE)
# Select a certain year
year = 2021

# Filter the data for the selected year
df_year = df_co2_per_capita[df_co2_per_capita['Year'] == year]

# Sort the data by 'Change (%) CO₂ emissions (per capita)'
df_sorted = df_year.sort_values(by='Percentage change - CO₂ emissions (per capita)', ascending=True)

# Select the top 10 countries
top_10_countries = df_sorted.head(10)

# Create a colormap
cmap = plt.get_cmap('RdBu_r')
norm = plt.Normalize(vmin=-25, vmax=25)

# Create a horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 10))
sns.barplot(x='Percentage change - CO₂ emissions (per capita)', y='Entity', data=top_10_countries, ax=ax, palette=cmap(norm(top_10_countries['Percentage change - CO₂ emissions (per capita)'])))

ax.set_xlabel('Relative annual change (%) in CO₂ emissions per capita (tonnes)')
ax.set_title(f'Figure 8. Top 10: Countries - "Relative change" in CO₂ emissions per capita for {year}')
ax.invert_xaxis()

plt.tight_layout()
plt.savefig('figures/figure8.png')
#plt.show()

#------------------------------------------------------------------------------------------------------------------------------------#

## CHOROPLETH MAP (RELATIVE CHANGE)
world_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world_map = world_map.rename(columns={'name': 'Entity'})

# Select a specific year
year = 2021

# Filter your data for the selected year
data_year = df_co2_per_capita[df_co2_per_capita['Year'] == year]

# Merge data with world map data
merged_data = world_map.merge(data_year, left_on='Entity', right_on='Entity')

# Create the Choropleth Map
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_facecolor('lightgray')

norm = plt.Normalize(vmin=-25, vmax=25)
merged_data.plot(column='Percentage change - CO₂ emissions (per capita)', cmap='RdBu_r', linewidth=0.8, ax=ax, edgecolor='0.8', norm = norm, legend=True)
ax.set_title(f'Figure 9. Relative change (%) in CO₂ emissions per capita (tonnes) for {year}')

plt.savefig('figures/figure9.png')
#plt.show()

#------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------#

## BAR CHART (ABSOLUTE CHANGE)
# Select a certain year
year = 2021

# Filter the data for the selected year
df_year = df_co2_per_capita[df_co2_per_capita['Year'] == year]

# Sort the data by 'Change (%) CO₂ emissions (per capita)' 
df_sorted = df_year.sort_values(by='Absolute change - CO₂ emissions (per capita)', ascending=True)

# Select the top 10 countries
top_10_countries = df_sorted.head(10)

# Create a colormap
cmap = plt.get_cmap('RdBu_r')
norm = plt.Normalize(vmin=-5, vmax=5)

# Create a horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 10))
sns.barplot(x='Absolute change - CO₂ emissions (per capita)', y='Entity', data=top_10_countries, ax=ax, palette=cmap(norm(top_10_countries['Absolute change - CO₂ emissions (per capita)'])))

ax.set_xlabel('Absolute annual change in CO₂ emissions per capita (tonnes)')
ax.set_title(f'Figure 10. Top 10: Countries - "Absolute change" in CO₂ emissions per capita for {year}')
ax.invert_xaxis()

plt.tight_layout()
plt.savefig('figures/figure9.png')
#plt.show()

#------------------------------------------------------------------------------------------------------------------------------------#

## CHOROPLETH MAP (ABSOLUTE CHANGE)
world_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world_map = world_map.rename(columns={'name': 'Entity'})

# Select a specific year
year = 2021

# Filter your data for the selected year
data_year = df_co2_per_capita[df_co2_per_capita['Year'] == year]

# Merge data with world map data
merged_data = world_map.merge(data_year, left_on='Entity', right_on='Entity')

# Create the Choropleth Map
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_facecolor('lightgray')

norm = plt.Normalize(vmin=-5, vmax=5)
merged_data.plot(column='Absolute change - CO₂ emissions (per capita)', cmap='RdBu_r', linewidth=0.8, ax=ax, edgecolor='0.8', norm = norm, legend=True)
ax.set_title(f'Figure 11. Absolute change in CO₂ emissions per capita (tonnes) for {year}')

plt.savefig('figures/figure11.png')
#plt.show()