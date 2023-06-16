import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sklearn.model_selection import LeaveOneGroupOut

### Question: Is there a relation between a country's Gross Domestrict Product (GDP) and its income inequality?

''' 
Some tips:
Be aware of the difference between correlation and causation here. A might cause B. B might cause A. 
But both A and B could be caused by an unknown C as well.

One way to express income inequality is to look at a country's "Gini coefficient" (also known as "Gini index"). 
You can find a dataset of Gini Coefficients here: https://ourworldindata.org/income-inequality#high-income-countries-tend-to-have-lower-inequality

You can find a dataset with historical GDP data here: https://ourworldindata.org/economic-growth#gdp-per-capita-over-the-long-run

To be able to answer this question you would want to calculate the "correlation coefficient" of the GDP and the Gini coefficient. 
But before you can do that you may need to resample the data so a correlation coefficient can be calculated.

More info about calculating correlations using pandas and other Python libraries: https://www.youtube.com/watch?v=TRNaMGkdn-A

'''

## Using datatables in Google Colab 
# %load_ext google.colab.data_table
# pd.options.display.max_columns = 75   # Allows to view more columns (default max = 20)

gini_df = pd.read_csv("economic-inequality-gini-index.csv") 
gdp_df = pd.read_csv("gdp-per-capita-maddison-2020.csv")

## Filter & Clean data:

## Dropping columns and rows not needed for this assignment
gini_df.dropna(subset=["Code"], inplace=True)                   # Dropping non country rows (like: World, Europe)
gini_df.drop(["Code"], axis=1, inplace=True)                    # Dropping the "Code" column

gdp_df.dropna(subset=["Code"], inplace=True)                        # Dropping non country rows (like: World, Europe)
gdp_df = gdp_df[gdp_df["Entity"] != "World"]                        # Dropping the "World" rows
gdp_df.drop(["Code", "417485-annotations"], axis=1, inplace=True)   # Dropping "Code" and "417485-annotations" column

## Checking for a null values
gini_df.isnull().sum()          # No null values present
gdp_df.isnull().sum()           # No null values present

## Merge the two dataframes on "Entity" and "Year"

df = pd.merge(gdp_df, gini_df, on=["Entity", "Year"], how='outer')

## Droppings rows (countries) where there isn't a Gini-coefficient present
df.dropna(inplace=True)

#------------------------------------------------------------------------------------------------------------------------------------#

### Check for a relationship between "GDP" and "Income Inequality (Gini Coef.)"

## Calculate Pearson correlation
corr_coeff = df["GDP per capita"].corr(df["Gini coefficient"])                          # Outcome is '-0.43'
print("Correlation coefficient between 'GDP' and 'Gini coefficient' =", corr_coeff)
print("According the guidelines there is a 'Relatively Strong' correlation.\n")

## Calculate Pearson correlation incl. P-value
corr_pvalue = pearsonr(df["GDP per capita"], df["Gini coefficient"])                    # Outcome is very close to '0'
print("The P-value between 'GDP' and 'Gini coefficient' =", corr_pvalue[1])
print("According the guidelines there is a 'Relatively High' chance of a relationship between these two values.\n")

## Scatter plot
fig = plt.figure(figsize=(9, 6))
plt.style.use("fivethirtyeight")

plt.scatter(df["GDP per capita"], df["Gini coefficient"], s=5)
plt.xlabel("GDP per capita")
plt.ylabel("Gini coefficient")

plt.title("Relationship:\nGDP and Income Inequality (Gini Coeff.)")
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------#

### Apply cross-validation (resampling) to assess the accuracy and generalizability (grouping data by "Year")
logo = LeaveOneGroupOut().split(df["GDP per capita"], groups=df["Year"])

correlations = []
pvalues = []

# Perform leave-one-out cross-validation
for train_idx, test_idx in logo:
    # Split data into training and test sets
    X_train = df.iloc[train_idx]["GDP per capita"]
    y_train = df.iloc[train_idx]["Gini coefficient"]
    X_test = df.iloc[test_idx]["GDP per capita"]
    y_test = df.iloc[test_idx]["Gini coefficient"]
    
    # Calculate correlation coefficient and p-value
    corr_coeff, p_value = pearsonr(X_train, y_train)
    
    # Append results to lists
    correlations.append(corr_coeff)
    pvalues.append(p_value)

# Calculate average correlation coefficient and p-value
avg_corr_coeff = sum(correlations) / len(correlations)
avg_pvalue = sum(pvalues) / len(pvalues)

print("Cross-validation results:")
print("Average correlation coefficient:", avg_corr_coeff)
print("Average p-value:", avg_pvalue)

#------------------------------------------------------------------------------------------------------------------------------------#

### Conclusion
print("\nConclusion:")

print('''
The original correlation coefficient is -0.4322 and the p-value is 2.7052-83 (near "0"). This suggests negative 
correlation between GDP per capita and income inequality, meaning that higher levels of GDP per capita tend to be 
associated with lower levels of income inequality.''')

print('''
After cross-validation the correlation coefficient seems stable, which means the orginal result is likely to be reliable.''')

print('''
However, the P-value increased significantly: 2.7052-83 to 7.2845-80. Which means the evidence of a correlation has weakened, 
and that the relationship between GDP per capita and income inequality may be less strong than originally thought. 
There is a higher chance that the observed correlation coefficient could be due to random chance.''')

print('''
Overall, these results suggest that the negative correlation between GDP per capita and Gini coefficient is likely a true 
underlying relationship, it is stable and generalizable across different subsets of the data. However, the evidence against 
the null hypothesis may be slightly weaker than previously thought and the strength of the relationship may be influenced 
by other factors not accounted for in the analysis.''')

