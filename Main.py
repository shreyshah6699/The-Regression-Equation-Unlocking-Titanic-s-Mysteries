#!/usr/bin/env python
# coding: utf-8

# # Final project AAI-551
# name : Shrey Shah   
# CWid : 20009523

# # Data analysis on Titanic Dataset

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv('titanic.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.describe()


# # Printing total null value in dataset

# In[6]:


print(df.isnull().sum())


# ### Taking mean for filling null value in 'Age' column. removing whole row where null in 'Embarked'column and Cabin data is not useful for us here. 

# In[7]:


#df = df.dropna(subset=["Age"])
df = df.dropna(subset=["Embarked"])
df["Age"].fillna(df["Age"].mean(), inplace=True)


# In[8]:


print(df.isnull().sum())


# # Creating custom function for summary of dataset

# In[9]:


def custom_summry(df):
    
    result = []         # Creating an empty list called result 
    
    # Iterating all the columns in the data for studying Descriptive stats
    
    for col in ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp',
       'Parch',  'Fare']:
        stats = OrderedDict({"Feature Name":col,
                             'Count':df[col].count(),
                            'Minimum':df[col].min(),
                            'Quartile1':df[col].quantile(0.25),
                            'Mean':df[col].mean(),
                            'Median':df[col].median(),
                            'Quartile3':df[col].quantile(0.75),
                            'IQR':(df[col].quantile(0.75)-df[col].quantile(0.25)),
                            'Maximum':df[col].max(),
                            'Variance':df[col].var(),
                            'Standard Deviation':df[col].std(),
                            'Skewness':df[col].skew(),
                            'Kurtosis':df[col].kurt()})
        
        #custom comment for identifying skew 
        
        if df[col].skew()<-1:
            sk_label = 'highly Negatively Skewed'
        elif -1<= df[col].skew() < -0.5:
            sk_label = 'Moderately Negatively Skewed'
        elif -0.5 <= df[col].skew() <0:
            sk_label = 'Fairly Symmetric(-ve)'
        elif 0 <= df[col].skew() < 0.5:
            sk_label = 'Fairly Symmetric(+ve)'
        elif 0.5 <= df[col].skew() <1:
            sk_label = 'Moderately Skewed (+ve)'
        elif df[col].skew() >1:
            sk_label = 'Higly (+ve) Skewed'
        else:
            sk_label = 'error'
        stats['Skeweness Comment'] = sk_label
        
        #custom comment for identifying Outliers
        
        uplim = stats['Quartile3'] + 1.5 * stats['IQR']
        lowlim = stats['Quartile1'] - 1.5 * stats['IQR']
        if len([x for x in df[col] if x < lowlim or x > uplim])> 0:
            outlier_comment = 'Has outlier'
        else:
            outlier_comment = 'No outlier'
            
        stats['outlier_comment']= outlier_comment
        
            
        result.append(stats)
    summary_df = pd.DataFrame(data=result)
    return summary_df


# In[10]:


custom_summry(df)


# # Replacing value to 0 and 1 in "Sex" column.(one hot encoding)

# In[11]:


df['Sex'] = df['Sex'].replace({'male': 1, 'female': 0})


# In[12]:


plt.hist(df['Age'])
plt.show()


# In[13]:


plt.scatter(df['Age'], df['Fare'])
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()


# In[14]:


pivot_table = df.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='mean')
print(pivot_table)


# In[15]:


grouped = df.groupby(['Sex'])
print(grouped['Survived'].mean())


# In[16]:


def calculate_range(column):
    return column.max() - column.min()


# In[17]:


for column in ['Age', 'Fare']:
    print(f"Range of {column}: {calculate_range(df[column])}")


# In[18]:


corr = df.corr()
print(corr)


# In[19]:


sns.heatmap(corr, annot=True)


# In[20]:


sns.pairplot(df, hue='Survived')


# ## Creating Linear Regression model for the dataset

# In[21]:


df = df.dropna()
X = df[['Age']]
y = df['Fare']

model = LinearRegression()
model.fit(X, y)


# In[22]:


y_pred_lr = model.predict(X)


# In[23]:


mse_lr = mean_squared_error(y, y_pred_lr)
print(f"Mean Squared Error (Linear Regression): {mse_lr}")


# In[24]:


print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_}")


# # Creating Logistic Regression model for the dataset

# In[25]:


X = df[['Age', 'Fare']]
y = df['Survived']

model = LogisticRegression()
model.fit(X, y)


# In[26]:


print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")


# In[27]:


y_pred_logr = model.predict(X)


# In[28]:


acc_logr = accuracy_score(y, y_pred_logr)
print(f"Accuracy (Logistic Regression): {acc_logr}")


# In[29]:


# Create a single DataFrame with the true values and predictions
predictions = pd.DataFrame({'Linear Regression': y_pred_lr, 'Logistic Regression': y_pred_logr, 'True Value': y})


# # Ploting the predictions

# In[30]:


plt.plot(y_pred_lr, label='Linear Regression', color='black')
plt.plot(y_pred_logr, label='Logistic Regression', color='red')
plt.legend()
plt.show()


# # Create a distribution plot

# In[31]:


sns.distplot(predictions['True Value'] - predictions['Linear Regression'], label='Linear Regression')
sns.distplot(predictions['True Value'] - predictions['Logistic Regression'], label='Logistic Regression')
plt.legend()
plt.show()


# # Create a scatter plot with the true values and the predictions

# In[32]:


sns.scatterplot(data=predictions, x='True Value', y='Linear Regression', hue='Linear Regression')
sns.scatterplot(data=predictions, x='True Value', y='Logistic Regression', hue='Logistic Regression')
plt.show()


# In[ ]:




