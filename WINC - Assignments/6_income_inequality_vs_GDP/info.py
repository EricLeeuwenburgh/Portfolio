import numpy as np
import pandas as pd
from scipy.stats import pearsonr

'''
Pearson correlation:

The Pearson correlation is a statistical measure that describes the strength and direction of the relationship 
between two continuous variables. It ranges from -1 to 1, where a value of 1 indicates a perfect positive relationship 
(as one variable increases, so does the other), a value of -1 indicates a perfect negative relationship 
(as one variable increases, the other decreases), and a value of 0 indicates no relationship between the two variables.

To calculate the Pearson correlation, you need to have two sets of data, and then you find the average of both sets, 
the standard deviation of each set, and then you multiply the deviations of both sets and sum them. Finally, you divide 
the sum by the product of the two standard deviations.

The Pearson correlation is often used in research to determine whether there is a relationship between two variables, 
such as the relationship between height and weight, or between age and income. It can help us understand how closely two 
variables are related and can be used to make predictions or draw conclusions based on the data.

Absolute Value      Interpretation
0.00 < 0.10         Negligible
0.10 < 0.20         Weak
0.20 < 0.40         Moderate
0.40 < 0.60         Relatively Strong
0.60 < 0.80         Strong
0.80 <= 1.00        Very Strong

----------------------------------------------------------------------------------------------------------------------------

P-value:
The p-value is a way to determine if the results of a study are meaningful or just happened by chance. It's like rolling 
a dice and getting a six, and then asking yourself: "Is this just luck or is the dice biased?". The p-value helps to answer 
that question by giving you a number that tells you how likely it is that the dice is biased. If the p-value is low, 
then it's unlikely that the result happened by chance, and you can conclude that the dice is probably biased. If the 
p-value is high, then it's likely that the result happened by chance, and you can't conclude that the dice is biased.

Similarly, in statistical analysis, if the p-value is low (usually less than 0.05), then it's unlikely that the observed 
results happened by chance alone, and you can conclude that there is a significant difference or relationship between the 
variables being studied. If the p-value is high (greater than 0.05), then it's likely that the observed results happened 
by chance, and you can't conclude that there is a significant difference or relationship between the variables.

Interpretation of P-value (standard)
Below 0.05 = Less likely to be based on chance alone (there is a relationship between the two variables)
Above 0.05 = More likely to be based on chance alone (there is no relationship between the two variables)

More on P-values: https://www.youtube.com/watch?v=kyjlxsLW1Is&list=PLICW5UpCwEj0duPGdUdkzkvbZ9zlhBbi_&index=1&t=567s

'''
# Generate data 
np.random.seed(1)  # for reproducibility

data = {
    'Start_Salary': np.random.randint(low=30000, high=50000, size=10),
    'Current_Salary': np.random.randint(low=50000, high=90000, size=10),
}

df = pd.DataFrame(data)
#print(df)

## Pandas correlation (doesn't show p-value)
print(df.corr())

## NumPy correlation (doesn't show p-value)
print(np.corrcoef(df["Start_Salary"], df["Current_Salary"]))

## SciPy correlation (incl. p-value)
print(pearsonr(df["Start_Salary"], df["Current_Salary"]))


'''
Difference between Correlation and Causation:

Correlation and causation are two concepts in statistics that are often confused with each other. While both of these terms 
are related to the relationship between two variables, they are not the same thing.

Correlation refers to a statistical measure that indicates how strongly two variables are related to each other. 
A correlation coefficient can range from -1 to +1, with a value of 0 indicating no correlation and a value of 1 indicating 
a perfect positive correlation. However, correlation does not imply causation. In other words, just because two variables are 
strongly correlated does not mean that one causes the other.

Causation, on the other hand, refers to a relationship between two variables in which one variable (the cause) directly 
affects the other variable (the effect). Causation implies correlation, but correlation does not necessarily imply causation. 
In order to establish causation, a researcher must conduct experiments or studies that manipulate the supposed cause and 
observe the effect.

In summary, correlation measures the strength of the relationship between two variables, while causation indicates that 
one variable directly affects another. Correlation does not imply causation, and establishing causation requires additional 
evidence beyond a correlation coefficient.

'''
#------------------------------------------------------------------------------------------------------------------------------------#

'''
Gini coefficient:

A Gini coefficient of 0 indicates perfect equality, meaning everyone in the population has the same income (or wealth, or education, 
etc.), while a Gini coefficient of 1 indicates perfect inequality, meaning one person has all the income (or wealth, or education, etc.) while everyone else has none.

A Gini coefficient between 0 and 1 represents the degree of inequality in the population, where higher values indicate greater 
inequality. For example, a Gini coefficient of 0.4 means that the top 20% of the population holds 60% of the income (or wealth, 
or education, etc.), while the bottom 20% holds only 5%.

In summary, the higher the Gini coefficient, the greater the degree of inequality in the population.

Common way to interpret Gini coefficient values:

 - A Gini coefficient of 0 to 0.2:
   indicates a relatively equal distribution of income, wealth, or education. 
   This means that the population is relatively homogeneous in terms of the measured variable, 
   and there is less inequality in the distribution.

 - A Gini coefficient of 0.2 to 0.4:
   indicates a moderate level of inequality in the distribution of income, wealth, or education. 
   In this range, there is a notable difference in the distribution of the measured variable, 
   with a significant portion of the variable concentrated in a relatively small part of the population.

 - A Gini coefficient of 0.4 and above: 
   indicates a high level of inequality in the distribution of income, wealth, or education. 
   In this range, the concentration of the measured variable is more extreme, with a significant amount 
   of the variable held by a small portion of the population, while the rest of the population has less access to it.

However, it's important to note that these are general guidelines and should be interpreted with caution. 
Gini coefficients may also be affected by factors such as the size and composition of the population, 
the measurement of the variable, and the time period studied.

'''
#------------------------------------------------------------------------------------------------------------------------------------#

'''
Re-sampling:

Resampling is a statistical technique in which a new sample is drawn from an existing dataset. Resampling can be useful for a 
variety of purposes, such as estimating the variability of a statistic, validating a model, or testing a hypothesis.

There are two main types of resampling:

- Bootstrap resampling: In bootstrap resampling, new samples are drawn with replacement from the original dataset. 
This means that each new sample may contain duplicate observations from the original dataset. Bootstrap resampling 
can be used to estimate the sampling distribution of a statistic, which can be used to construct confidence intervals 
or conduct hypothesis tests.

- Cross-validation: In cross-validation, the original dataset is divided into two or more subsets, and models are trained 
and evaluated on different subsets. Cross-validation can be used to validate a model, estimate its performance on new data, 
or tune its hyperparameters.

'''