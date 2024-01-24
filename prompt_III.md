# Practical Application III: Comparing Classifiers

**Overview**: In this practical application, your goal is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.  We will utilize a dataset related to marketing bank products over the telephone.  



### Getting Started

Our dataset comes from the UCI Machine Learning repository [link](https://archive.ics.uci.edu/ml/datasets/bank+marketing).  The data is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns.  We will make use of the article accompanying the dataset [here](CRISP-DM-BANK.pdf) for more information on the data and features.



### Problem 1: Understanding the Data

To gain a better understanding of the data, please read the information provided in the UCI link above, and examine the **Materials and Methods** section of the paper.  How many marketing campaigns does this data represent?

### Problem 2: Read in the Data

Use pandas to read in the dataset `bank-additional-full.csv` and assign to a meaningful variable name.


```python
###  Given the original dataset 17 campaigns corresponding to a total of 79354 contacts and 59 client attributes
### and the learning objective of classification modeling task of 4 classifiers in a reasonable run-time, 
### this data used in this study is a scaled-down version corresponding to 4119 contacts and 19 attributes 
```


```python
import pandas as pd
```


```python
# Due to computationally demanding machine learning algorithms (e.g., SVM), a smaller data set is used.
df = pd.read_csv('data/bank-additional.csv', sep = ';')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>...</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>emp.var.rate</th>
      <th>cons.price.idx</th>
      <th>cons.conf.idx</th>
      <th>euribor3m</th>
      <th>nr.employed</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>basic.9y</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>may</td>
      <td>fri</td>
      <td>...</td>
      <td>2</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-1.8</td>
      <td>92.893</td>
      <td>-46.2</td>
      <td>1.313</td>
      <td>5099.1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>39</td>
      <td>services</td>
      <td>single</td>
      <td>high.school</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>fri</td>
      <td>...</td>
      <td>4</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.1</td>
      <td>93.994</td>
      <td>-36.4</td>
      <td>4.855</td>
      <td>5191.0</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>services</td>
      <td>married</td>
      <td>high.school</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>telephone</td>
      <td>jun</td>
      <td>wed</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.4</td>
      <td>94.465</td>
      <td>-41.8</td>
      <td>4.962</td>
      <td>5228.1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38</td>
      <td>services</td>
      <td>married</td>
      <td>basic.9y</td>
      <td>no</td>
      <td>unknown</td>
      <td>unknown</td>
      <td>telephone</td>
      <td>jun</td>
      <td>fri</td>
      <td>...</td>
      <td>3</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>1.4</td>
      <td>94.465</td>
      <td>-41.8</td>
      <td>4.959</td>
      <td>5228.1</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47</td>
      <td>admin.</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>nov</td>
      <td>mon</td>
      <td>...</td>
      <td>1</td>
      <td>999</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>-0.1</td>
      <td>93.200</td>
      <td>-42.0</td>
      <td>4.191</td>
      <td>5195.8</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
#Step 1: Checking if  missing values 
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4119 entries, 0 to 4118
    Data columns (total 21 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   age             4119 non-null   int64  
     1   job             4119 non-null   object 
     2   marital         4119 non-null   object 
     3   education       4119 non-null   object 
     4   default         4119 non-null   object 
     5   housing         4119 non-null   object 
     6   loan            4119 non-null   object 
     7   contact         4119 non-null   object 
     8   month           4119 non-null   object 
     9   day_of_week     4119 non-null   object 
     10  duration        4119 non-null   int64  
     11  campaign        4119 non-null   int64  
     12  pdays           4119 non-null   int64  
     13  previous        4119 non-null   int64  
     14  poutcome        4119 non-null   object 
     15  emp.var.rate    4119 non-null   float64
     16  cons.price.idx  4119 non-null   float64
     17  cons.conf.idx   4119 non-null   float64
     18  euribor3m       4119 non-null   float64
     19  nr.employed     4119 non-null   float64
     20  y               4119 non-null   object 
    dtypes: float64(5), int64(5), object(11)
    memory usage: 675.9+ KB


### Problem 3: Understanding the Features


Examine the data description below, and determine if any of the features are missing values or need to be coerced to a different data type.


```
Input variables:
# bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
```



### Problem 4: Understanding the Task

After examining the description and data, your goal now is to clearly state the *Business Objective* of the task.  State the objective below.


```python
### Approach:  Best Machine Learning Model (Accuracy and Intepretability)  is obtained from simplfication for extracting  knowledge sucessfully.
### thus , the objective is to transform the output from multi-class to binary classification problem
```


```python
### Since the data set have 
### a) 'poutcome' = outcome  with {'failure', 'nonexistent',  'success' } from the previous marketing campaign, 
### b) 'y' (whether the client subscribed to a term deposit with  {'yes' ,  'no' } from in the present campain)

```


```python
### Business Objective : Predict whether the client subscribes to a term deposit in the current campaign
### Thus the task is to use only the conclusive results  (Positive Outcome is 'Yes' in 'y' )
### by creating a new target variable,  y_concluded 

```

### Problem 5: Engineering Features

Now that you understand your business objective, we will build a basic model to get started.  Before we can do this, we must work to encode the data.  Using just the bank information features (columns 1 - 7), prepare the features and target column for modeling with appropriate encoding and transformations.


```python
# Step 1.2 : Since no missing values, "Label Encoding" for Ordinal Categorical Variables
from sklearn.preprocessing import LabelEncoder
columns_to_encode = ['job', 'marital', 'education', 'default','housing', 'loan','month','day_of_week','poutcome']

label_encoder = LabelEncoder()

for column in columns_to_encode:
    df[column + '_encoded'] = label_encoder.fit_transform(df[column])

# Keep only the encoded columns and drop the original ones
df_encoded = df.drop(columns=columns_to_encode)
```


```python
#Step 1.3: One-Hot Encoding for Nominal (no inherent order) Categorical Variables

mydf = pd.get_dummies(df_encoded, columns=['contact'], prefix='contact')
```


```python
# Step 1.3: Convert multiple-class classification into a binary classification by focusing on the concluded contacts
# Step 1: Create a new binary target variable for concluded contacts
mydf['y_concluded'] = mydf['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Step 2: Drop the 'duration' column as suggested for realistic predictive modeling
mydf = mydf.drop('duration', axis=1)
```


```python
# Visualization 
import matplotlib.pyplot as plt
import seaborn as sns

# Display basic statistics of the dataset
print(mydf.describe())

# Visualize the distribution of the target variable
sns.countplot(x='y_concluded', data=mydf)
plt.title('Distribution of Target Variable')
plt.show()
```

                   age     campaign        pdays     previous  emp.var.rate  \
    count  4119.000000  4119.000000  4119.000000  4119.000000   4119.000000   
    mean     40.113620     2.537266   960.422190     0.190337      0.084972   
    std      10.313362     2.568159   191.922786     0.541788      1.563114   
    min      18.000000     1.000000     0.000000     0.000000     -3.400000   
    25%      32.000000     1.000000   999.000000     0.000000     -1.800000   
    50%      38.000000     2.000000   999.000000     0.000000      1.100000   
    75%      47.000000     3.000000   999.000000     0.000000      1.400000   
    max      88.000000    35.000000   999.000000     6.000000      1.400000   
    
           cons.price.idx  cons.conf.idx    euribor3m  nr.employed  job_encoded  \
    count     4119.000000    4119.000000  4119.000000  4119.000000  4119.000000   
    mean        93.579704     -40.499102     3.621356  5166.481695     3.824958   
    std          0.579349       4.594578     1.733591    73.667904     3.606319   
    min         92.201000     -50.800000     0.635000  4963.600000     0.000000   
    25%         93.075000     -42.700000     1.334000  5099.100000     1.000000   
    50%         93.749000     -41.800000     4.857000  5191.000000     3.000000   
    75%         93.994000     -36.400000     4.961000  5228.100000     7.000000   
    max         94.767000     -26.900000     5.045000  5228.100000    11.000000   
    
           ...  education_encoded  default_encoded  housing_encoded  loan_encoded  \
    count  ...        4119.000000      4119.000000      4119.000000   4119.000000   
    mean   ...           3.780286         0.195436         1.081573      0.348386   
    std    ...           2.149588         0.397196         0.983915      0.741647   
    min    ...           0.000000         0.000000         0.000000      0.000000   
    25%    ...           2.000000         0.000000         0.000000      0.000000   
    50%    ...           3.000000         0.000000         2.000000      0.000000   
    75%    ...           6.000000         0.000000         2.000000      0.000000   
    max    ...           7.000000         2.000000         2.000000      2.000000   
    
           month_encoded  day_of_week_encoded  poutcome_encoded  contact_cellular  \
    count    4119.000000          4119.000000       4119.000000       4119.000000   
    mean        4.294975             2.009711          0.924253          0.643846   
    std         2.305188             1.389233          0.372816          0.478920   
    min         0.000000             0.000000          0.000000          0.000000   
    25%         3.000000             1.000000          1.000000          0.000000   
    50%         4.000000             2.000000          1.000000          1.000000   
    75%         6.000000             3.000000          1.000000          1.000000   
    max         9.000000             4.000000          2.000000          1.000000   
    
           contact_telephone  y_concluded  
    count        4119.000000  4119.000000  
    mean            0.356154     0.109493  
    std             0.478920     0.312294  
    min             0.000000     0.000000  
    25%             0.000000     0.000000  
    50%             0.000000     0.000000  
    75%             1.000000     0.000000  
    max             1.000000     1.000000  
    
    [8 rows x 21 columns]



    
![png](prompt_III_files/prompt_III_18_1.png)
    



```python
# Visualize the correlation matrix
corr_matrix = mydf.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
```

    /var/folders/wk/p6xy78b50699b5qtxgv6t_6c0000gn/T/ipykernel_17053/2360327039.py:2: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      corr_matrix = mydf.corr()



    
![png](prompt_III_files/prompt_III_19_1.png)
    



```python
# Visualize the distribution of numerical features
num_features = mydf.select_dtypes(include=['float64', 'int64']).columns
mydf[num_features].hist(figsize=(12, 12))
plt.suptitle('Distribution of Numerical Features', y=0.92)
plt.show()
```


    
![png](prompt_III_files/prompt_III_20_0.png)
    


### Problem 6: Train/Test Split

With your data prepared, split it into a train and test set.


```python
# Identify features (X) and target variable (y)
# Drop the y column to focus on the binary classification task

mydf.drop(columns='y', inplace=True)
X = mydf.drop('y_concluded', axis=1)  
y = mydf['y_concluded'] 
```


```python
# Assuming X, y are your features and target variable
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Problem 7: A Baseline Model

Before we build our first model, we want to establish a baseline.  What is the baseline performance that our classifier should aim to beat?

### Initialize and fit a DummyClassifier


```python

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dummy_classifier = DummyClassifier(strategy='most_frequent')
dummy_classifier.fit(X_train, y_train)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DummyClassifier(strategy=&#x27;most_frequent&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">DummyClassifier</label><div class="sk-toggleable__content"><pre>DummyClassifier(strategy=&#x27;most_frequent&#x27;)</pre></div></div></div></div></div>




```python
# Make predictions on the test set
y_pred = dummy_classifier.predict(X_test)

# Calculate baseline performance metrics
baseline_accuracy = accuracy_score(y_test, y_pred)
baseline_precision = precision_score(y_test, y_pred,zero_division=1)
baseline_recall = recall_score(y_test, y_pred)
baseline_f1 = f1_score(y_test, y_pred)
```

### Print baseline metrics


```python

print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"Baseline Precision: {baseline_precision:.4f}")
print(f"Baseline Recall: {baseline_recall:.4f}")
print(f"Baseline F1 Score: {baseline_f1:.4f}")
```

    Baseline Accuracy: 0.8883
    Baseline Precision: 1.0000
    Baseline Recall: 0.0000
    Baseline F1 Score: 0.0000


### Problem 8: A Simple Model

Use Logistic Regression to build a basic model on your data.  


```python
# Use Logistic Regression model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

### Problem 9: Score the Model

What is the accuracy of your model?


```python
# Evaluate the performance of the best model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print performance metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
```

    Accuracy: 0.8968
    Precision: 0.6522
    Recall: 0.1630
    F1 Score: 0.2609


### Problem 10: Model Comparisons

Now, we aim to compare the performance of the Logistic Regression model to our KNN algorithm, Decision Tree, and SVM models.  Using the default settings for each of the models, fit and score each.  Also, be sure to compare the fit time of each of the models.  Present your findings in a `DataFrame` similar to that below:

| Model | Train Time | Train Accuracy | Test Accuracy |
| ----- | ---------- | -------------  | -----------   |
|     |    |.     |.     |


```python
# Establish default Model Comparisions with all 19 parameters
### b) StandardScaler needed in LogisticRegression
### c) A metric table Classifier , Train Score, Test score 
```


```python
import time
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# Start metrics table with default LogisticRegression using cv = 5 

# Summary table columns
summary_columns = ['Default_Classifier','Train Score', 'Test Score', 'Fit Time']

summary_df_list = []

X_test_scaled = scaler.transform(X_test)

# LogisticRegression with cross-validation
start_time = time.time()

lgr = LogisticRegression().fit(X_train_scaled,y_train)

lgr_train_time = time.time() - start_time

# Get the results and append them to the summary DataFrame
lgr_results = {
        'Default_Classifier': 'LogisticRegression',
        'Train Score': cross_val_score(lgr, X_train_scaled, y_train, cv=5, scoring='accuracy').mean(),
        'Test Score': lgr.score(X_test_scaled, y_test),
        'Fit Time': lgr_train_time,
    }
    
#summary_df = summary_df.append(results, ignore_index=True)
summary_df_list.append(pd.DataFrame([lgr_results]))

# Concatenate the list of DataFrames into the summary DataFrame
summary_df = pd.concat(summary_df_list, ignore_index=True)

# Display the summary DataFrame
print(summary_df)
```

       Default_Classifier  Train Score  Test Score  Fit Time
    0  LogisticRegression     0.901062    0.896845  0.030003



```python
# KNN with cross-validation
start_time = time.time()

knn = KNeighborsClassifier().fit(X_train_scaled,y_train)

knn_train_time = time.time() - start_time

# Get the results and append them to the summary DataFrame
knn_results = {
        'Default_Classifier': 'KNN',
        'Train Score': cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy').mean(),
        'Test Score': knn.score(X_test_scaled, y_test),
        'Fit Time': knn_train_time,
    }
    
#summary_df = summary_df.append(results, ignore_index=True)
summary_df_list.append(pd.DataFrame([knn_results]))

# Concatenate the list of DataFrames into the summary DataFrame
summary_df = pd.concat(summary_df_list, ignore_index=True)

# Display the summary DataFrame
print(summary_df)
```

       Default_Classifier  Train Score  Test Score  Fit Time
    0  LogisticRegression     0.901062    0.896845  0.030003
    1                 KNN     0.896510    0.889563  0.002264



```python
# SVM with cross-validation
start_time = time.time()

svm = SVC().fit(X_train_scaled,y_train)

svm_train_time = time.time() - start_time

# Get the results and append them to the summary DataFrame
svm_results = {
        'Default_Classifier': 'SVM',
        'Train Score': cross_val_score(svm, X_train_scaled, y_train, cv=5, scoring='accuracy').mean(),
        'Test Score': svm.score(X_test_scaled, y_test),
        'Fit Time': svm_train_time,
    }
    
#summary_df = summary_df.append(results, ignore_index=True)
summary_df_list.append(pd.DataFrame([svm_results]))

# Concatenate the list of DataFrames into the summary DataFrame
summary_df = pd.concat(summary_df_list, ignore_index=True)

# Display the summary DataFrame
print(summary_df)
```

       Default_Classifier  Train Score  Test Score  Fit Time
    0  LogisticRegression     0.901062    0.896845  0.030003
    1                 KNN     0.896510    0.889563  0.002264
    2                 SVM     0.901973    0.894417  0.233464



```python
# DecisionTree with cross-validation. No need input Featuring , i.e., no need StandardScaler 
start_time = time.time()

dt = DecisionTreeClassifier().fit(X_train,y_train)

dt_train_time = time.time() - start_time

# Get the results and append them to the summary DataFrame
dt_results = {
        'Default_Classifier': 'DecisionTree',
        'Train Score': cross_val_score(dt, X_train, y_train, cv=5, scoring='accuracy').mean(),
        'Test Score': dt.score(X_test, y_test),
        'Fit Time': dt_train_time,
    }
    
#summary_df = summary_df.append(results, ignore_index=True)
summary_df_list.append(pd.DataFrame([dt_results]))

# Concatenate the list of DataFrames into the summary DataFrame
summary_df = pd.concat(summary_df_list, ignore_index=True)

```


```python
# Display the summary with 4 default Classifiers 
```


```python
print(summary_df)
```

       Default_Classifier  Train Score  Test Score  Fit Time
    0  LogisticRegression     0.901062    0.896845  0.030003
    1                 KNN     0.896510    0.889563  0.002264
    2                 SVM     0.901973    0.894417  0.233464
    3        DecisionTree     0.829439    0.845874  0.021923


### Problem 11: Improving the Model

Now that we have some basic models on the board, we want to try to improve these.  Below, we list a few things to explore in this pursuit.

- More feature engineering and exploration.  For example, should we keep the gender feature?  Why or why not?
- Hyperparameter tuning and grid search.  All of our models have additional hyperparameters to tune and explore.  For example the number of neighbors in KNN or the maximum depth of a Decision Tree.  
- Adjust your performance metric


```python
# Improvement the models:
### a) Dimensionality Reduction: Reduce 19 parameters to 5 parameters
###    From correlation matrix, assume threhold =0.2, pick the important parameters
### b) Pipeline using Custom Hyperparameter Tunning,StandardScaler as needed and GridSearchCV
### c) New metric Classifier , Train Score, Test score and Best Parameter 
# Five parameters {pdays, previous,emp.var.rate, euribor3m,nr.employed" are used . 
```


```python
# List of column names to select
selected_columns = ['pdays', 'previous','emp.var.rate', 'euribor3m','nr.employed', 'y_concluded']

# Using filter method
filtered_df = mydf.filter(items=selected_columns)
# Display the filtered DataFrames
print("Filtered DataFrame (using filter method):")
print(filtered_df)

```

    Filtered DataFrame (using filter method):
          pdays  previous  emp.var.rate  euribor3m  nr.employed  y_concluded
    0       999         0          -1.8      1.313       5099.1            0
    1       999         0           1.1      4.855       5191.0            0
    2       999         0           1.4      4.962       5228.1            0
    3       999         0           1.4      4.959       5228.1            0
    4       999         0          -0.1      4.191       5195.8            0
    ...     ...       ...           ...        ...          ...          ...
    4114    999         0           1.4      4.958       5228.1            0
    4115    999         0           1.4      4.959       5228.1            0
    4116    999         1          -1.8      1.354       5099.1            0
    4117    999         0           1.4      4.966       5228.1            0
    4118    999         0          -0.1      4.120       5195.8            0
    
    [4119 rows x 6 columns]



```python
X = filtered_df.drop('y_concluded', axis=1)  
y = filtered_df['y_concluded']
```


```python
# Assuming X, y are your features and target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```


```python
### GridSearch with LogisticRegression
```


```python

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# LogisticRegression
# Summary table columns
imp_summary_columns = ['Tuned_Classifier','Train Score', 'Test Score', 'Fit Time', 'Best Parameters']

# DataFrame to store summary
imp_summary_df = pd.DataFrame(columns=imp_summary_columns)

# KNN
# DataFrame to store summary
imp_summary_df_list = []
lgr_params = {
    'lgr__max_iter' : [100, 500, 2500],
    'lgr__C' : [0.001, 0.01, 0.1, 1, 10, 100]
    }

lgr_pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('lgr', LogisticRegression())
])
    
# GridSearchCV with cross-validation
lgr_search = GridSearchCV(lgr_pipeline, param_grid = lgr_params, cv=5 ).fit(X_train,y_train)
    
# Get the results and append them to the summary DataFrame
lgr_results = {
        'Tuned_Classifier': 'LogisticRegression',
        'Train Score': lgr_search.best_estimator_.score(X_train, y_train),
        'Test Score': lgr_search.best_estimator_.score(X_test, y_test),
        'Fit Time': lgr_search.cv_results_['mean_fit_time'].mean(),
        'Best Parameters': lgr_search.best_params_
    }
    
imp_summary_df_list.append(pd.DataFrame([lgr_results]))

# Concatenate the list of DataFrames into the summary DataFrame
imp_summary_df = pd.concat(imp_summary_df_list, ignore_index=True)

# Display the summary DataFrame
#print(summary_df)
```


```python
### GridSearch with KNN 
```


```python

KNN_params = {
    'knn__n_neighbors': [5, 7, 9],
    'knn__weights'    : ['uniform', 'distance']
        }

KNN_pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('knn', KNeighborsClassifier())
])
  
# GridSearchCV with cross-validation
KNN_search = GridSearchCV(KNN_pipeline, param_grid = KNN_params, cv=5 ).fit(X_train,y_train)
    
# Get the results and append them to the summary DataFrame
KNN_results = {
        'Tuned_Classifier': 'KNN',
        'Train Score': KNN_search.best_estimator_.score(X_train, y_train),
        'Test Score': KNN_search.best_estimator_.score(X_test, y_test),
        'Fit Time': KNN_search.cv_results_['mean_fit_time'].mean(),
        'Best Parameters': KNN_search.best_params_
    }
    
#summary_df = summary_df.append(results, ignore_index=True)
imp_summary_df_list.append(pd.DataFrame([KNN_results]))

# Concatenate the list of DataFrames into the summary DataFrame
imp_summary_df = pd.concat(imp_summary_df_list, ignore_index=True)

# Display the summary DataFrame
#print(summary_df)
```


```python
### GridSearch with SVM
```


```python

svc_params = {
     'svm__kernel': ['poly', 'rbf', 'sigmoid'],
     'svm__C': [0.1, 1.0, 10.0]
          }
svc_pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('svm', SVC())
])
   
# GridSearchCV with cross-validation
svc_search = GridSearchCV(svc_pipeline, param_grid = svc_params, cv = 5).fit(X_train,y_train)
    
# Get the results and append them to the summary DataFrame
svc_results = {
        'Tuned_Classifier': 'SVM',
        'Test Score': svc_search.best_estimator_.score(X_test, y_test),
        'Fit Time': svc_search.cv_results_['mean_fit_time'].mean(),
        'Best Parameters': svc_search.best_params_
    }
    
imp_summary_df_list.append(pd.DataFrame([svc_results]))

# Concatenate the list of DataFrames into the summary DataFrame
imp_summary_df = pd.concat(imp_summary_df_list, ignore_index=True)


```


```python
### GridSearch with Decision Tree
```


```python
# Decision Tree
dt_params = {
     'max_depth': [5, 10, 15], 
     'min_samples_split': [2, 5, 10], 
     'min_samples_leaf': [1, 2, 4]
}
    
# GridSearchCV with cross-validation
dt_search = GridSearchCV(DecisionTreeClassifier(), param_grid = dt_params, cv=5 ).fit(X_train,y_train)
    
# Get the results and append them to the summary DataFrame
dt_results = {
        'Tuned_Classifier': 'DecisionTree',
        'Train Score': dt_search.best_estimator_.score(X_train, y_train),
        'Test Score': dt_search.best_estimator_.score(X_test, y_test),
        'Fit Time': dt_search.cv_results_['mean_fit_time'].mean(),
        'Best Parameters': dt_search.best_params_
    }

imp_summary_df_list.append(pd.DataFrame([dt_results]))

# Concatenate the list of DataFrames into the summary DataFrame
imp_summary_df = pd.concat(imp_summary_df_list, ignore_index=True)

```


```python
### Display the summary Data Frame with improved Classifiers ( GridSearch and Hyper Parameter Tunning) 
```


```python
print(imp_summary_df)  
```

         Tuned_Classifier  Train Score  Test Score  Fit Time  \
    0  LogisticRegression     0.902276    0.902913  0.007356   
    1                 KNN     0.905615    0.898058  0.003688   
    2                 SVM          NaN    0.902913  0.569773   
    3        DecisionTree     0.909863    0.898058  0.002633   
    
                                         Best Parameters  
    0             {'lgr__C': 0.01, 'lgr__max_iter': 100}  
    1  {'knn__n_neighbors': 9, 'knn__weights': 'unifo...  
    2             {'svm__C': 1.0, 'svm__kernel': 'poly'}  
    3  {'max_depth': 5, 'min_samples_leaf': 1, 'min_s...  



```python
### For Comparision between default and improved models, display the summary DataFrame with default Classifiers 
```


```python
print(summary_df)  
```

       Default_Classifier  Train Score  Test Score  Fit Time
    0  LogisticRegression     0.901062    0.896845  0.030003
    1                 KNN     0.896510    0.889563  0.002264
    2                 SVM     0.901973    0.894417  0.233464
    3        DecisionTree     0.829439    0.845874  0.021923



```python
# Conclusion: Given the business decision of 
## 1.0 Given the simplification of 5 input features and Corresponding Classifiers (Default or Tuned): 
### In terms of Fit Time, KNN is fastest while SVM is slowest
### In terms of Test_Score, LogisticRegression is highest while DecisionTree is lowest
##
## 2.0 I would deploy KNN as the model of choice for its combined test_score and speed
# 
## 3.0 Five important variables: 'pdays', 'previous','emp.var.rate', 'euribor3m','nr.employed'
#
```


```python
# Next Step/Recommendations to further enhance Model Performance 
## 1.0 Feature Interaction: Create interaction terms or polynomial features to capture non-linear relationships
## 2.0 Outlier Handling: Investigate and handle outliers in the data because they can affect model performance
```

##### Questions


```python

```
