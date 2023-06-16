### Please formulate an answer to the following question in your report:
###     1. What is the biggest predictor of a large CO2 output per capita of a country?

from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplcursors
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

# Using datatables in Google Colab (sorting columns: looking at .head and .tail values)
# %load_ext google.colab.data_table

# pd.options.display.max_columns = 50   # Allows to view more columns (default max = 20)
# pd.options.display.max_rows = 10000   # Allows to view more rows (default max = 20000)

#------------------------------------------------------------------------------------------------------------------------------------#

### BIGGEST CO2 ANALYSIS
## Find the biggest CO2 emitting countries per capita for 2021
df_co2_per_capita = pd.read_csv("data/co2-per-capita.csv")
df_co2_per_capita.dropna(subset=['Code'], inplace=True)
df_co2_per_capita = df_co2_per_capita.drop(df_co2_per_capita[df_co2_per_capita['Entity'] == 'World'].index)

df_co2_per_capita_2021 = df_co2_per_capita[df_co2_per_capita['Year'] == 2021]
df_co2_per_capita_2021_sorted = df_co2_per_capita_2021.sort_values('Annual CO₂ emissions (per capita)', ascending=False)

top_50 = list(df_co2_per_capita_2021_sorted['Entity'].head(50))

#------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------#

### SECTOR ANALYSIS
## Find the biggest CO2 contributing sector per country per year
df_sector = pd.read_csv("data/co2-by-sector.csv")
df_sector.dropna(subset=['Code'], inplace=True)
df_sector = df_sector.drop(df_sector[df_sector['Entity'] == 'World'].index)
df_sector.fillna(0, inplace=True)

columns_to_check = ['Buildings', 'Industry', 'Land-use change and forestry', 'Other fuel combustion', 'Transport',
                   'Manufacturing and construction', 'Fugitive emissions', 'Electricity and heat']

df_sector['Biggest sector'] = df_sector[columns_to_check].abs().idxmax(axis=1)  # .idxmax() returns the index label of the max value

# Calculate the relative size of the biggest sector vs total
absolute_total = df_sector[columns_to_check].abs().sum(axis=1)
df_sector['Relative size'] = df_sector[columns_to_check].abs().div(absolute_total, axis=0).max(axis=1)

# Select the latest year (2019) for every country
year_2019 = df_sector['Year'] == 2019
df_sector_2019 = df_sector.loc[year_2019, ['Entity', 'Year', 'Biggest sector', 'Relative size']]

# Total emissions by sector
sum_columns = df_sector[columns_to_check].sum()
sum_columns = sum_columns.sort_values(ascending=False)

# Convert total to relative percentages
total_sum = sum_columns.sum()
relative_percentages = (sum_columns / total_sum) * 100

#------------------------------------------------------------------------------------------------------------------------------------#

# Bar chart - Total CO2 emissions by sector (2019)
# Assign colors to each sector
sector_colors = {
    'Buildings': 'blue',
    'Industry': 'green',
    'Land-use change and forestry': 'orange',
    'Other fuel combustion': 'red',
    'Transport': 'purple',
    'Manufacturing and construction': 'brown',
    'Fugitive emissions': 'pink',
    'Electricity and heat': 'gray'
}

plt.figure(figsize=(12, 6))
plt.bar(relative_percentages.index, relative_percentages.values, color=['gray', 'purple', 'brown', 'blue', 'orange', 'green', 'red', 'pink'])

plt.xticks(rotation=90)
plt.ylabel("CO2 Emission percentage (%)")
plt.title("Figure 1. Relative CO2 Emissions by Sector (2019)")
plt.grid(axis='y')

plt.tight_layout()
plt.savefig('figures/figure1.png')
#plt.show()

#------------------------------------------------------------------------------------------------------------------------------------#

# Bar chart - Relative size of biggest sector per country (2019)
df_top_50 = df_sector_2019[df_sector_2019['Entity'].isin(top_50)]
df_top_50 = df_top_50.sort_values('Relative size', ascending=True)
unique_sectors = ['Land-use change and forestry', 'Transport', 'Electricity and heat']

fig, ax = plt.subplots(figsize=(14, 6))

# Iterate over each entity and plot the horizontal bar chart
for index, row in df_top_50.iterrows():
    entity = row['Entity']
    sector = row['Biggest sector']
    relative_size = row['Relative size']
    color = sector_colors[sector]
    
    ax.barh(entity, relative_size, color=color)

ax.yaxis.set_tick_params(labelsize=8)
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.05))
plt.xticks(rotation=90)

ax.set_xlabel('Relative size of sector')
ax.set_title('Figure 2. Top 50 - CO2 emitting countries per capita (2021)\nSorted by the relative size of their biggest CO2 emitting sector (2019)')

legend_handles = [plt.Rectangle((0, 0), 1, 1, color=sector_colors[sector]) for sector in unique_sectors]
ax.legend(legend_handles, unique_sectors, loc='lower right')
ax.grid(axis='x')

fig.tight_layout()
plt.savefig('figures/figure2.png')
#plt.show()

#------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------#

### CORRILATION ANALYSIS
## Read in the data
df0 = pd.read_csv("data/co2-per-capita.csv")
df1 = pd.read_csv("data/co2-vs-gdp.csv")
df2 = pd.read_csv("data/energy-per-capita-source-stacked.csv")
df3 = pd.read_csv("data/registered-vehicles-per-1000-people.csv")
df4 = pd.read_csv("data/population-density.csv")
df5 = pd.read_csv("data/co2-per-capita-consumption-vs-production.csv")
df6 = pd.read_csv("data/co2-by-sector.csv")

# Filter data per dataset
df0.dropna(subset=['Code'], inplace=True)
df0 = df0.drop(df0[df0['Entity'] == 'World'].index)
df0.drop(['Code'], axis=1, inplace=True)
df0 = df0[df0['Year'] == 2021]          # Selecting '2021' = most recent year available
df0 = df0.rename(columns={'Annual CO₂ emissions (per capita)': 'CO₂ emissions per capita'})

df1.dropna(subset=['Code'], inplace=True)
df1 = df1.drop(df1[df1['Entity'] == 'World'].index)
df1.drop(['Code', '417485-annotations', 'Annual CO₂ emissions (per capita)', 'Population (historical estimates)', 'Continent'], axis=1, inplace=True)
df1.dropna(inplace=True)
df1 = df1[df1['Year'] == 2018]          # Selecting '2018' due lots of missing data in later years

df2.dropna(subset=['Code'], inplace=True)
df2 = df2.drop(df2[df2['Entity'] == 'World'].index)
df2.drop(['Code'], axis=1, inplace=True)
df2 = df2[df2['Year'] == 2021]          # Selecting '2021' = most recent year available
df2 = df2.rename(columns={'Fossil fuels per capita (kWh)': 'Fossil per capita (kWh)', 'Nuclear per capita (kWh - equivalent)': 'Nuclear per capita (kWh)', 'Renewables per capita (kWh - equivalent)': 'Renewables per capita (kWh)'})
df2.fillna(0, inplace=True)
df2['Total energy per capita (kWh)'] = df2['Fossil per capita (kWh)'] + df2['Nuclear per capita (kWh)'] + df2['Renewables per capita (kWh)']

df3.dropna(subset=['Code'], inplace=True)
df3 = df3.drop(df3[df3['Entity'] == 'World'].index)
df3.drop(['Code'], axis=1, inplace=True)
df3 = df3[df3['Year'] == 2016]          # Selecting '2016' = most recent year available
df3 = df3.rename(columns={'Registered vehicles per 1,000 people': 'Vehicles per 1,000 people'})

df4.dropna(subset=['Code'], inplace=True)
df4 = df4.drop(df4[df4['Entity'] == 'World'].index)
df4.drop(['Code'], axis=1, inplace=True)
df4 = df4[df4['Year'] == 2021]          # Selecting '2021' = aligning with other datasets

df5.dropna(subset=['Code'], inplace=True)
df5 = df5.drop(df5[df5['Entity'] == 'World'].index)
df5.drop(['Code', 'Share of world population', 'Income classifications (World Bank (2017))'], axis=1, inplace=True)
df5.dropna(inplace=True)
df5['CO₂ consumption vs production (%)'] = round(df5['Annual consumption-based CO₂ emissions (per capita)'] / df5[ 'Annual CO₂ emissions (per capita)'] * 100, 2)
df5.drop(['Annual consumption-based CO₂ emissions (per capita)', 'Annual CO₂ emissions (per capita)'], axis=1, inplace=True)
df5 = df5[df5['Year'] == 2020]          # Selecting '2020' = most recent year, 2021 has a lot of missing data

df6.dropna(subset=['Code'], inplace=True)
df6 = df6.drop(df6[df6['Entity'] == 'World'].index)
df6.drop(['Code', 'Buildings', 'Industry', 'Other fuel combustion', 'Transport', 'Manufacturing and construction', 'Fugitive emissions', 'Electricity and heat'], axis=1, inplace=True)
df6.dropna(inplace=True)
df6 = df6[df6['Year'] == 2019]          # Selecting '2019' = most recent year

# Merge datasets together on 'Entity' (Note: data from different kind of year (latest year possible))
merged_df = df0.merge(df1, on='Entity', how='left', suffixes=('_df0', '_df1'))
merged_df = merged_df.merge(df2, on='Entity', how='left', suffixes=('_df1', '_df2'))
merged_df = merged_df.merge(df3, on='Entity', how='left', suffixes=('_df2', '_df3'))
merged_df = merged_df.merge(df4, on='Entity', how='left', suffixes=('_df3', '_df4'))
merged_df = merged_df.merge(df5, on='Entity', how='left', suffixes=('_df4', '_df5'))
merged_df = merged_df.merge(df6, on='Entity', how='left', suffixes=('_df5', '_df6'))

# Dropping Year-columns and NaN values of merged DataFrame (total: 219 countries)
merged_df.set_index('Entity', inplace=True)
merged_df.drop(columns=[f'Year_df{i}' for i in range(6)], inplace=True)
merged_df.drop(columns='Year', inplace=True)
merged_df.dropna(inplace=True)         # Leaving only 49 countries

#------------------------------------------------------------------------------------------------------------------------------------#

# Corrilations & P-values heatmap
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))
fig.suptitle("Figure 3. CO₂ Emissions per capita - Relationships ")

correlation_matrix = merged_df.corr()
p_values = np.zeros_like(correlation_matrix.values)
for i in range(len(merged_df.columns)):
    for j in range(len(merged_df.columns)):
        _, p_val = stats.pearsonr(merged_df[merged_df.columns[i]], merged_df[merged_df.columns[j]])
        p_values[i, j] = p_val

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax1, fmt='.3f', annot_kws={'size': 8})
sns.heatmap(p_values, annot=True, cmap='YlGnBu', ax=ax2, cbar=False, fmt='.3f', annot_kws={'size': 8})

ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=8)
ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=8)
ax1.set_title('Correlation matrix')

ax2.set_xticklabels(merged_df.columns, rotation=90, fontsize=8)
ax2.set_yticklabels(merged_df.columns, rotation=0, fontsize=8)
ax2.set_title('P-value matrix')

fig.tight_layout()
plt.savefig('figures/figure3.png')
#plt.show()

#------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------#

### GDP ANALYSIS
## Read in the data and select 2018 (latest year)
df_gdp = pd.read_csv("data/co2-vs-gdp.csv")
df_gdp.dropna(subset=['Code'], inplace=True)
df_gdp = df_gdp.drop(df_gdp[df_gdp['Entity'] == 'World'].index)
df_gdp.drop(['Code', '417485-annotations', 'Population (historical estimates)', 'Continent'], axis=1, inplace=True)
df_gdp = df_gdp[df_gdp['Year'] == 2018]
df_gdp.dropna(inplace=True)

#------------------------------------------------------------------------------------------------------------------------------------#

## Scatter plot - CO2 per capita vs GDP per capita
x = df_gdp['Annual CO₂ emissions (per capita)']
y = df_gdp['GDP per capita']

# Logarithmic transformation
x_log = np.log10(x)
y_log = np.log10(y)

# Linear regression (logarithmic scale)
slope, intercept = np.polyfit(x_log, y_log, 1)
regression_line = slope * x_log + intercept

plt.figure(figsize=(10, 6))

plt.scatter(x, y, edgecolor='black', alpha=0.75)
plt.plot(x, 10**(regression_line), color='red', label='Linear Regression')

plt.xscale('log')       # Set plot to logarithmic scale
plt.yscale('log')

plt.xlabel('GDP per capita (thousand $)')
plt.ylabel('CO₂ Emissions per capita (tonnes)')
plt.title('Figure 4. CO₂ Emissions per capita vs GDP per capita (2018)')
plt.legend()

fig.tight_layout()
plt.savefig('figures/figure4.png')
#plt.show()


#------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------#

### ENERGY ANALYSIS
## Read in the data and select 2021
df_energy_sources = pd.read_csv("data/energy-per-capita-source-stacked.csv")
df_energy_sources = df_energy_sources.drop(df_energy_sources[df_energy_sources['Entity'] == 'World'].index)
df_energy_sources.dropna(subset=['Code'], inplace=True)
df_energy_sources.fillna(0, inplace=True)
df_energy_sources = df_energy_sources[df_energy_sources['Year'] == 2021]

# Merge with dataframe 'CO2 per capita'
df_energy_sources = df_energy_sources.merge(df_co2_per_capita_2021[['Entity', 'Annual CO₂ emissions (per capita)']], on='Entity', how='left')

# Calculate energy usage per capita
df_energy_sources['Total'] = df_energy_sources['Fossil fuels per capita (kWh)'] + df_energy_sources['Nuclear per capita (kWh - equivalent)'] + df_energy_sources['Renewables per capita (kWh - equivalent)']

# Sort dataframe
df_energy_sources_sorted = df_energy_sources.sort_values('Annual CO₂ emissions (per capita)', ascending=False)

#------------------------------------------------------------------------------------------------------------------------------------#

## Scatter plot - Energy per capita vs CO2 per capita
# Data
x_total = df_energy_sources['Total']
x_fossil = df_energy_sources['Fossil fuels per capita (kWh)']
x_nuclear = df_energy_sources['Nuclear per capita (kWh - equivalent)']
x_renewables = df_energy_sources['Renewables per capita (kWh - equivalent)']
y = df_energy_sources['Annual CO₂ emissions (per capita)']

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Figure 5. Energy Consumption vs. CO₂ Emissions per capita (2021)')

# Subplot - Total Energy 
axs[0, 0].scatter(x_total, y, color='purple', label='Total', edgecolor='black', alpha=0.75)
axs[0, 0].set_xscale('log')
axs[0, 0].set_yscale('log')
axs[0, 0].set_xlabel('Total Energy Consumption per capita (kWh)')
axs[0, 0].set_ylabel('CO₂ Emissions per capita')
axs[0, 0].legend()

# Subplot - Fossil 
axs[0, 1].scatter(x_fossil, y, color='brown', label='Fossil', edgecolor='black', alpha=0.75)
axs[0, 1].set_xscale('log')
axs[0, 1].set_yscale('log')
axs[0, 1].set_xlabel('Fossil Fuel Consumption per capita (kWh)')
axs[0, 1].set_ylabel('CO₂ Emissions per capita')
axs[0, 1].legend()

# Subplot - Nuclear
axs[1, 0].scatter(x_nuclear, y, color='yellow', label='Nuclear', edgecolor='black', alpha=0.75)
axs[1, 0].set_xscale('log')
axs[1, 0].set_yscale('log')
axs[1, 0].set_xlabel('Nuclear Energy Consumption per capita (kWh)')
axs[1, 0].set_ylabel('CO₂ Emissions per capita')
axs[1, 0].legend()

# Subplot - Renewable
axs[1, 1].scatter(x_renewables, y, color='green', label='Renewables', edgecolor='black', alpha=0.75)
axs[1, 1].set_xscale('log')
axs[1, 1].set_yscale('log')
axs[1, 1].set_xlabel('Renewable Energy Consumption per capita (kWh)')
axs[1, 1].set_ylabel('CO₂ Emissions per capita')
axs[1, 1].legend()

fig.tight_layout()
plt.savefig('figures/figure5.png')
#plt.show()


#------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------#

### CO2 CONSUMPTION VS PRODUCTION ANALYSIS
## Read in the data and select 2020
df_co2_cons_vs_prod = pd.read_csv("data/co2-per-capita-consumption-vs-production.csv")   
df_co2_cons_vs_prod.dropna(subset=['Code'], inplace=True)
df_co2_cons_vs_prod = df_co2_cons_vs_prod.drop(df_co2_cons_vs_prod[df_co2_cons_vs_prod['Entity'] == 'World'].index)
df_co2_cons_vs_prod.drop(['Code', 'Share of world population', 'Income classifications (World Bank (2017))'], axis=1, inplace=True)
df_co2_cons_vs_prod.dropna(inplace=True)
df_co2_cons_vs_prod['CO₂ consumption vs production (%)'] = round(df_co2_cons_vs_prod['Annual consumption-based CO₂ emissions (per capita)'] / df_co2_cons_vs_prod[ 'Annual CO₂ emissions (per capita)'] * 100, 2)
df_co2_cons_vs_prod = df_co2_cons_vs_prod[df_co2_cons_vs_prod['Year'] == 2020]          # Selecting '2020' = most recent year, 2021 has a lot of missing data
df_co2_cons_vs_prod.set_index('Entity', inplace=True)

#------------------------------------------------------------------------------------------------------------------------------------#

## Scatter plot - Consumption- vs Production-based CO2 emissions (per capita)
countries = df_co2_cons_vs_prod.index
consumption_emissions = df_co2_cons_vs_prod['Annual consumption-based CO₂ emissions (per capita)']
production_emissions = df_co2_cons_vs_prod['Annual CO₂ emissions (per capita)']
consumption_vs_production = df_co2_cons_vs_prod['CO₂ consumption vs production (%)']

fig, ax = plt.subplots(figsize=(10, 6))

norm = colors.Normalize(vmin=50, vmax=200)
cmap = colors.LinearSegmentedColormap.from_list('TwoColorGradient', ['green', 'red'])

scatter = ax.scatter(production_emissions, consumption_emissions, c=consumption_vs_production, alpha=0.75, cmap=cmap, norm=norm, edgecolors='black')

plt.xlabel('CO₂ emissions per capita (tonnes)')
plt.ylabel('Consumption-Based CO2 Emissions per capita (tonnes)')
plt.title('Figure 6. Consumption-based vs. production-based CO₂ emissions per capita')

# Adding labels
tooltip = mplcursors.cursor(scatter, hover=True)

@tooltip.connect("add")
def on_add(sel):
    index = sel.index
    sel.annotation.set_text(f"{countries[index]}\n{consumption_vs_production[sel.index]:.2f}%")

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('CO₂ consumption vs production (%)')

plt.xscale('log')       
plt.yscale('log')

plt.tight_layout()
plt.savefig('figures/figure6.png')
#plt.show()


#------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------#

### LAND-USE CHANGE & FORESTRY ANALYSIS
## Read in the data and select 2019
df_sector = df_sector[df_sector['Year'] == 2019]          # Selecting '2019' = most recent year 
df_sector = df_sector.drop(['Code', 'Year', 'Buildings', 'Industry', 'Other fuel combustion', 'Transport', 'Manufacturing and construction', 'Fugitive emissions', 'Electricity and heat', 'Biggest sector', 'Relative size'], axis=1)
df_sector = df_sector[df_sector['Land-use change and forestry'] != 0]   # All 0 values are excluded, because data is missing or no correction is needed

# Add CO2 emissions per capita to dataset
df_co2_per_capita_2019 = df_co2_per_capita[df_co2_per_capita['Year'] == 2019]
df_co2_per_capita_2019 = df_co2_per_capita_2019.drop(['Code', 'Year'], axis=1)

df_sector = df_sector.merge(df_co2_per_capita_2019, on='Entity', how='left')
df_sector.set_index('Entity', inplace=True)

#------------------------------------------------------------------------------------------------------------------------------------#

## Scatter plot - Consumption- vs Production-based CO2 emissions (per capita)
fig, ax = plt.subplots(figsize=(10, 6))
x = df_sector['Annual CO₂ emissions (per capita)']
y = round(df_sector['Land-use change and forestry'] / 1000, 0)

norm = colors.Normalize(vmin=-100000, vmax=100000)
cmap = colors.LinearSegmentedColormap.from_list('TwoColorGradient', ['green', 'red'])

scatter = ax.scatter(x, y, c=y, alpha=0.75, cmap=cmap, norm=norm, edgecolors='black')

plt.xlabel('CO₂ emissions per capita (tonnes)')
plt.ylabel('Land-use change and forestry CO₂ emissions (1000 tonnes)')
plt.title('Figure 7. Land-use change and forestry CO₂ emissions vs CO₂ emissions per capita')

# Adding labels
tooltip = mplcursors.cursor(scatter, hover=True)

@tooltip.connect("add")
def on_add(sel):
    index = sel.index
    land_use_emissions = y[sel.index]
    co2_per_capita = x[sel.index]
    sel.annotation.set_text(f"Country: {df_sector.index[index]}\nLand-use change and forestry CO₂ emissions: {land_use_emissions}\nCO₂ emissions per capita: {co2_per_capita}")

# Customizing x-axis tick labels for log scale
def format_ticks(value, pos):
    return "{:.1f}".format(value)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

plt.xscale('log')

plt.tight_layout()
plt.savefig('figures/figure7.png')
#plt.show()


#------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------#