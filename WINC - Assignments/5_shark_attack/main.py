import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#### Assignment: Shark Attacks

### Question 1: What are the most dangerous types of sharks to humans?
### Question 2: Are children more likely to be attacked by sharks? 
### Question 3: Are shark attacks where sharks were provoked more or less dangerous?
### Question 4: Are certain activities more likely to result in a shark attack?

'''
If you feel you can't answer a question based on the dataset alone, feel free to find other datasets and use them in answering the questions.
For each answer you give not only answer the question but also write about the assumptions you made in answering the question. 
If an assumption or decision possibly created a bias please write about this as well.

'''

# Using datatables in Google Colab (sorting columns: looking at .head and .tail values)
# %load_ext google.colab.data_table

# Reading in the dataset (global data on shark attacks)
path = "attacks.csv"
df = pd.read_csv(path, encoding='ISO-8859-1')   # "ISO-8859" needed to be able to read the downloaded file

# Filter out columns which are not needed for this assignment
df.drop(columns=["Case Number", "Date", "Year", "Country", "Area", "Name", "Location", "Time", "Investigator or Source", "pdf", 
                 "href formula", "href", "Case Number.1", "Case Number.2", "original order", "Unnamed: 22", "Unnamed: 23"], 
                 inplace=True)

# Filter out rows where all values are NaN
df = df.dropna(how="all")                                           # Total rows = 6302

#------------------------------------------------------------------------------------------------------------------------------------#

### Cleaning data per column
# "Type" column
df["Type"].replace([np.nan, "Invalid"], None, inplace=True)         # "NaN" and "Invalid" are not fitting for this column
df["Type"].replace("Boatomg", "Boating", inplace=True)              # Assuming "Boatomg" is a typo (count = 1) 

# "Activity" column
len(df["Activity"].unique())                                                            # 1533 different kind (unique) of values
df["Activity"].value_counts().head(60).sort_index()                                     # Look for similar values
df["Activity"].replace("Walking.*", "Walking",regex=True, inplace=True)                 # Fixing small typo differences
df["Activity"].replace("Wade..ishing", "Wade fishing",regex=True, inplace=True)         # Fixing small typo differences
df["Activity"].replace("Swimming.*", "Swimming", regex=True, inplace=True)              # Fixing small typo differences
df["Activity"].replace("Surf.skiing.*", "Surf skiing", regex=True, inplace=True)        # Fixing small typo differences
df["Activity"].replace("Surfing.*", "Surfing", regex=True, inplace=True)                # Fixing small typo differences
df["Activity"].replace("Spearfishing.*", "Spearfishing", regex=True, inplace=True)      # Fixing small typo differences
df["Activity"].replace("Skin diving", "Skindiving", inplace=True)                       # Fixing small typo differences
df["Activity"].replace("Sea Disaster", "Sea disaster", inplace=True)                    # Fixing small typo differences
df["Activity"].replace("S.... .iving.*", "Scuba diving", regex=True, inplace=True)      # Fixing small typo differences
df["Activity"].replace("Playing.*", "Playing",regex=True, inplace=True)                 # Fixing small typo differences
df["Activity"].replace("Pearl diving", "Diving", inplace=True)                          # Fixing small typo differences
df["Activity"].replace("Kite Surfing", "Kite surfing", inplace=True)                    # Fixing small typo differences
df["Activity"].replace("Kayak.*Fishing", "Kayak Fishing", regex=True, inplace=True)     # Fixing small typo differences
df["Activity"].replace("Jumping", "Jumped into the water", inplace=True)                # Fixing small typo differences
df["Activity"].replace("Freedom Swimming", "Freedom swimming", inplace=True)            # Fixing small typo differences
df["Activity"].replace("Free diving.*", "Free diving", regex=True, inplace=True)        # Fixing small typo differences
df["Activity"].replace("Freediving", "Free diving", inplace=True)                       # Fixing small typo differences
df["Activity"].replace("Floating.*", "Floating", regex=True, inplace=True)              # Fixing small typo differences
df["Activity"].replace("Fishing.*", "Fishing", regex=True, inplace=True)                # Fixing small typo differences
df["Activity"].replace("Fell.*", "Fell overboard", regex=True, inplace=True)            # Fixing small typo differences
df["Activity"].replace("Diving for.*", "Diving", regex=True, inplace=True)              # Fixing small typo differences
df["Activity"].replace("Boogie..oarding", "Boogie boarding",regex=True, inplace=True)   # Fixing small typo differences
df["Activity"].replace("Body..oarding", "Body boarding",regex=True, inplace=True)       # Fixing small typo differences
df["Activity"].replace("boat capsized", "Boat capsized", inplace=True)                  # Fixing small typo differences
df["Activity"].replace("Boat", "Boating", inplace=True)                                 # Fixing small typo differences
df["Activity"].replace("Bathing.*", "Bathing",regex=True, inplace=True)                 # Fixing small typo differences

# "Sex" column (Male, Female or Non-Binary)
df.rename(columns={'Sex ': 'Sex'}, inplace=True)                    # Fixing typo in column name
df["Sex"].replace("M ", "M", inplace=True)                          # Fixing typo "M " 
df["Sex"].replace("lli", "N", inplace=True)                         # "lli" to Non_Binary
df["Sex"].replace([np.nan, "."], None, inplace=True)                # "NaN" and "." is not fitting for this column

# "Age" column
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')               # Convert all values to numeric

# "Injury" column
df.loc[df["Injury"].str.contains("PROVOKED", na = False), "Type"] = "Provoked"      # Overwriting "Type" column values due comments in "Injury" column
df.loc[df["Injury"].str.contains("FATAL", na = False), "Fatal (Y/N)"] = "Y"         # Overwriting "Fatal (Y/N)" column values due comments in "Injury" column
df["Injury"] = df["Injury"].astype(str)                                             # Convert all values to strings

# "Fatal (Y/N)" column
df["Fatal (Y/N)"].replace(["M", " N", "N "], "N", inplace=True)                     # Assuming "M"(count = 1), " N"(count = 7), "N "(count = 1) are typos
df["Fatal (Y/N)"].replace(["UNKNOWN", "2017", np.nan], None, inplace=True)          # "NaN", "2017" and "UNKNOWN" are not fitting for this column
df["Fatal (Y/N)"].replace("Y", True, inplace=True)                                  # Converting to boolean for easier use with calculations
df["Fatal (Y/N)"].replace("N", False, inplace=True)                                 # Converting to boolean for easier use with calculations

# "Species" column (info on shark species: https://en.wikipedia.org/wiki/List_of_sharks)
df.rename(columns={'Species ': 'Species'}, inplace=True)            # Fixing typo in column name
df["Species"].replace([np.nan, "Invalid"], None, inplace=True)      # "NaN" and "Invalid" are not fitting for this column

len(df["Species"].unique())                                         # 1549 different kind (unique) of values
df["Species"].isnull().sum()                                        # 2940 null values (almost 50%)

#------------------------------------------------------------------------------------------------------------------------------------#

### Question 1: What are the most dangerous types of sharks to humans?
## Answers to Question 1
print("Question 1: What are the most dangerous types of sharks to humans?")
q1_df = df.groupby("Species")["Fatal (Y/N)"].sum().sort_values(ascending=False).rename("Nr. of Fatal attacks")
print("Top 3 - Most lethal shark",q1_df.head(3))

## Bar chart
fig = plt.figure(figsize=(8, 6))
plt.style.use("fivethirtyeight")
plt.bar(q1_df.head(3).index, q1_df.head(3).values)

plt.xlabel("Shark species")
plt.ylabel("Amount of fatal attacks")

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.title("Most lethal shark species to humans")
plt.tight_layout()
#plt.show()

"""
Answer =    "1: White shark, 2: Tiger shark, 3: Bull shark" (see bar chart)

Assumptions:    (Main) In general the "(Great) White shark" is know to be the most deadly shark to humans.
                I have interpreted 'most dangerous' as 'most deadly' == most fatal injuries.
                If the "Injury" column would contain the word "FATAL" I updated the column "Fatal Y/N" to 'True' even if 
                the original value was 'False'. 

Reasoning:      I didn't want to create a certain bias by cleaning/transforming the dataset to much. As shown above there are
                a lot of different kind (unique) of values in the "Species" column which are also hard to group. 
                By looking at the top 20 a lot more values containing 'white shark' are found, but often in combination with 
                uncertainty (like: '?'). Because of this uncertainty I didn't do any more transformations on the values.
                Note: almost 50% of all values in the "Species" column are null values. 
                The current outcome is already according my main assumption.
"""

#------------------------------------------------------------------------------------------------------------------------------------#

### Question 2: Are children more likely to be attacked by sharks?

# Filter out certain types for this question
q2_df = df.loc[~df["Type"].isin(["Sea Disaster", "Boating", "Boat"])].copy()

# Create age bins | Total values: 3292 (no null values)
bins = [0, 17, 35, 53, 71, 99]
q2_df["Age_bins"] = pd.cut(q2_df["Age"], bins, labels=["child", "young adult", "middle-age adult", "older adult", "senior adult"])       # Create bins to see how many people are child/adult
q2_df["Age_bins"].value_counts(dropna=False)

# Count number of children/adults
child_count = (q2_df["Age_bins"].str.contains("child")).sum()
young_adult_count = (q2_df["Age_bins"].str.contains("young")).sum()
middle_age_adult_count = (q2_df["Age_bins"].str.contains("middle")).sum()
older_adult_count = (q2_df["Age_bins"].str.contains("older")).sum()
senior_adult_count = (q2_df["Age_bins"].str.contains("senior")).sum()
all_adult_count = (q2_df["Age_bins"].str.contains("adult")).sum()

# Calculate the percentage based on total
child_percentage = child_count / (child_count + all_adult_count) * 100          
adult_percentage = all_adult_count / (child_count + all_adult_count) * 100

## Answers to Question 2
print("\nQuestion 2: Are children more likely to be attacked by sharks?")
print(f"{round(child_percentage, 1)}% of all shark attacks affect children (age < 18)")

## Pie chart
fig = plt.figure(figsize=(12, 6))
plt.style.use("fivethirtyeight")

slices = [child_count, young_adult_count, middle_age_adult_count, older_adult_count, senior_adult_count]
labels = ["Children (0-17)", "Young Adults (18-35)", "Middle-age Adult (36-53)", "Older Adult (54-71)", "Senior Adult (72+)"]
explode = [0.1, 0, 0.1, 0.1, 0.2]

plt.pie(slices, labels=None, explode=explode, shadow=True, 
        startangle=90, autopct="%1.1f%%",
        wedgeprops={"edgecolor": "black"})

plt.title("Shark Attacks by Age-group")
plt.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
#plt.show()

"""
Answer =    "No" because 27.3% of all shark attacks affect children.
            "Yes" (see Pie chart), while "Young Adults" are attacked most often, "Children" come second. So children are attacked
            more often than "Middle-age Adults (36-53)", "Older Adults (54-71)" and "Senior Adults (72+)".

Assumptions:    I have assumed 'children' to be persons under the age of 18. (like most international institutions do)
                I have assumed 'more likely' as 'more often' since there isn't enough data to check for any child specific 
                relationships. More data would be needed (like: 'weight', 'height').
Reasoning:      I have transformed the "Age" column to single numbers. This means rows considering multiple victims are left out.
                Rows from "Age" column that contained null values are excluded from the calculation since we can't categorize them.
                I have interpreted 'attacked' to be all types exluding: "Boat", "Boating" and "Sea Disaster". I did this because
                I wanted the results to be based on an 'initial attack' from shark to human and not any attacks which are caused by
                a sea disaster or boat attack first. 
                I made a Pie chart because I wanted to make a fair age group split. All bins contain 18 years. Comparing age groups
                0-17 and 18-90 didn't seem fair to me since the adult group is much bigger (age count wise).
"""

#------------------------------------------------------------------------------------------------------------------------------------#

### Question 3: Are shark attacks where sharks were provoked more or less dangerous?

# Filter out certain types for this question
q3_df = df.loc[~df["Type"].isin(["Sea Disaster", "Boating", "Boat", "Questionable", None])].copy()    

# Count number of provoked/unprovoked attacks
provoked_count = q3_df["Type"].str.contains("Provoked").sum()                       
unprovoked_count = q3_df["Type"].str.contains("Unprovoked").sum()
provoked_fatal_count = (q3_df[q3_df["Type"].str.contains("Provoked") & q3_df["Fatal (Y/N)"]==True]).shape[0]
unprovoked_fatal_count = (q3_df[q3_df["Type"].str.contains("Unprovoked") & q3_df["Fatal (Y/N)"]==True]).shape[0]

# Calculate % of total results
percentage_provoked_total = provoked_count / (provoked_count + unprovoked_count) * 100    
percentage_provoked_fatal = (provoked_fatal_count / provoked_count) * 100
percentage_unprovoked_fatal = (unprovoked_fatal_count / unprovoked_count) * 100

# Count attacks for each type of injury (provoked)
q3_df_provoked = (q3_df.loc[q3_df["Type"].str.contains("Provoked")]).copy()
provoked_injury_no_count = (q3_df_provoked["Injury"].str.contains("no injury", case=False)).sum()      
provoked_injury_fatal_count = (q3_df_provoked["Fatal (Y/N)"]==True).sum()
provoked_injury_count = (q3_df_provoked.loc[~(q3_df_provoked["Injury"].str.contains("no injury", case=False) | q3_df["Fatal (Y/N)"]==True)]).shape[0]

# Count attacks for each type of injury (unprovoked)
q3_df_unprovoked = (q3_df.loc[q3_df["Type"].str.contains("Unprovoked")]).copy()
unprovoked_injury_no_count = q3_df_unprovoked["Injury"].str.contains("no injury", case=False).sum()       
unprovoked_injury_fatal_count = (q3_df_unprovoked["Fatal (Y/N)"]==True).sum()
unprovoked_injury_count = (q3_df_unprovoked.loc[~(q3_df_unprovoked["Injury"].str.contains("no injury", case=False) | q3_df["Fatal (Y/N)"]==True)]).shape[0]

# Calculate % for each injury type (provoked & unprovoked)
percentage_provoked_injury_no = (provoked_injury_no_count / provoked_count) * 100
percentage_unprovoked_injury_no = (unprovoked_injury_no_count / unprovoked_count) * 100
percentage_provoked_injury = (provoked_injury_count / provoked_count) * 100
percentage_unprovoked_injury = (unprovoked_injury_count / unprovoked_count) * 100
percentage_provoked_fatal = (provoked_injury_fatal_count / provoked_count) * 100
percentage_unprovoked_fatal = (unprovoked_injury_fatal_count / unprovoked_count) * 100

result_provoked = [percentage_provoked_injury_no, percentage_provoked_injury, percentage_provoked_fatal]
result_unprovoked = [percentage_unprovoked_injury_no, percentage_unprovoked_injury, percentage_unprovoked_fatal]

## Answers to Question 3
print("\nQuestion 3: Are shark attacks where sharks were provoked more or less dangerous?")
print(f"{round(percentage_provoked_total,1)}% of all shark attacks are provoked")
print(f"{round(percentage_provoked_fatal,1)}% of provoked attacks end up to be fatal")
print(f"{round(percentage_unprovoked_fatal,1)}% of unprovoked attacks end up to be fatal")

## (vertical) Bar chart
fig = plt.figure(figsize=(8, 6))
plt.style.use("fivethirtyeight")

width = 0.35
pos = [1,2,3]

plt.bar(pos, height=result_provoked, width=width, color="#7e2b11", label="Provoked") 
plt.bar([x + width for x in pos], height=result_unprovoked, width=width, color="#1f456e", label="Unprovoked")

plt.xlabel("Types of injury")
plt.ylabel("Relative result of attack")
plt.title("'Provoked' and 'Unprovoked' shark attacks")

plt.xticks(ticks=[1.175,2.175,3.175], labels=["No injury", "Injured", "Fatal"])
plt.yticks(ticks=range(0, 101, 10), labels=[f"{i}%" for i in range(0, 101, 10)])

plt.legend()
plt.tight_layout()
#plt.show()

"""
Answer =        "Less dangerous" because provoked attacks are relatively less fatal.
                "More dangerous" because provoked attacks relatively cause more injuries.

Assumptions:    I have assumed 'less dangerous' to be 'less fatal'. 
                If the "Injury" column would contain the word "PROVOKED" I updated the column "Type" to 'Provoked' even if 
                the original value was 'Unprovoked' or something else.
                If the "Injury" column would contain the word "FATAL" I updated the column "Fatal Y/N" to 'True' even if 
                the original value was 'False'.
                I've split the type of injuries ("Injury" column) into three groups. If column(string) would contain any form of 
                "no injury" in it, the type would be: "no injury". If the "Fatal (Y/N)" column is true then the injury type would 
                be "fatal". All the others are simply put into type "injury", so the "injury" type is positively biased.

Reasoning:      I've split the "Injury" column in a few types (bins) the get a better feeling for the damage being done.
                The sum of all kind of injuries = 5170. 'No injuries' = 424, 'fatal' = 1202. So 'injuries' should be 3544, instead it
                is 3545, 1 more than expected. I haven't been able to find this bug, but due the minor difference I've neglected it.
"""

#------------------------------------------------------------------------------------------------------------------------------------#

### Question 4: Are certain activities more likely to result in a shark attack?

# Filter out all columns but "Activity" for this question (total rows = 6302)
q4_df = df["Activity"].copy()   

# Filter out all rows with a null value (total rows = 5758)
q4_df.dropna(inplace=True)

## Answers to Question 4
print("\nQuestion 4: Are certain activities more likely to result in a shark attack?")
print("Top 10 - Nr. of shark attacks per", q4_df.value_counts().head(10))
print(q4_df.value_counts().head(30).sum())
## (horizontal) Bar chart

fig = plt.figure(figsize=(10,7))
plt.style.use("fivethirtyeight")

q4_data = q4_df.value_counts().head(30)
q4_data = q4_data.iloc[::-1]

plt.barh(range(len(q4_data)), q4_data.values, align='center')
plt.xticks(range(0, max(q4_data.values)+100, 100), fontsize=8)
plt.yticks(range(len(q4_data)), q4_data.index, fontsize=8)

plt.xlabel("Number of shark attacks")
plt.ylabel("Type of Activity")
plt.title("Top 30 - Activities subject to shark attacks")

plt.legend()
plt.tight_layout()
plt.show()

"""
Answer =        "See (horizontal) bar chart". 

Assumptions:    (see cleaning data) I have tried to keep the types of activities as 'pure' as possible. Like "Diving", this is a very
                broad term, I have only transformed values to "Diving" where the original value was pointing towards diving 'for a
                certain thing/object'. Types are diving like 'Free diving', 'Scuba diving' are still seperated.

Reasoning:      Due the wide variety of activities I've decided to pick the top 60 from .valuecounts() and started looking (manually)
                for different kind of typos and fixing those one by one. Since the bottom valuecounts ends at 3 if feel that I haven't
                neglected to many values. Transforming all unique values would be very time consuming and I don't feel this would 
                change the outcome of the result.
                To not have to many bars on the resulting bar chart (keep it readable) I've only selected the top 30 results. I think
                this gives a good (enough) answer to the question since the top value contains 1000+ attacks while the bottom values are 
                near 10 attacks (= 1%).
                The sum of the top 30 == 80% of all the values.
"""

#------------------------------------------------------------------------------------------------------------------------------------#