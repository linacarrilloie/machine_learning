#!/usr/bin/env python
# coding: utf-8

# In[300]:

### Class Project for Machine Learning 2 where the challenge was apply feature engineering techniques
### for a Logistic Regression model to estimate Churn at a company. 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbs
import sklearn
from dataset import Dataset as dataset
from sklearn.linear_model import LogisticRegression
from typing import List
from skrebate import ReliefF
from sklearn import datasets
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p
from sklearn.neighbors import LocalOutlierFactor
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV


# # 1. Data Loading

# In[1195]:


hr_raw = pd.read_csv('/Users/lina/Downloads/turnover 2.csv')


# # 2. Exploratory Data Analysis

# ### 2.0 First Exploration
# 

# In[1981]:


print(len(hr_raw))
#hr_raw.describe() #I am gonna hide this to not have too many pages!


# For this dataset, satisfaction_level, last_evaluation, average monthly hours, number of projects and time spent at company are numerical, however  number_project and time spend company are discrete as they have a low range and only increment by one unit.
# Work accident , left (my target) and promotion in the last 5 years are binary 
# Sales and salary are categorical, but require some encoding as they appear as strings.

# ### 2.1 Hunting for NA's

# In[1172]:


hr_raw.info()


# All of the values are not null so missing values won't be an issue for this model

# ### 2.2 One Hot Encoding for Sales and Salary
# As mentioned before, for "sales" and "salary" we will do a one hot encoding, I will do the same for number_project and time_spent_company
# This is the first change I am going to have on the data set so for this stage I will work with the pandas df "hr"

# In[2818]:


hr = pd.get_dummies(hr_raw, columns = ['sales'], prefix ='sales') 
hr = pd.get_dummies(hr, columns = ['salary'], prefix ='salary')
#hr.head() #to verify that the variables were correctly encoded


# ### 2.3 Dropping Outliers

# I am going to use the outliers function found on the datasets notebook and drop
# outliers for numerical variables

# In[2819]:


def outliers(self, n_neighbors=2000):
    
        """
        Find outliers, using LOF criteria, from the numerical features.
        Returns a list of indices where outliers are present
        :param n_neighbors: Number of neighbors to use by default for
            kneighbors queries. If n_neighbors is larger than the number
            of samples provided, all samples will be used.
        # TODO Implement a simple set of methods to select from in order to
               detect outliers.
        """
        X = hr_fe_numerical
        
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
        y_pred = lof.fit_predict(X)
        outliers = np.where(y_pred == -1)
        return outliers[0]
    
#dataframe for only my numerical variables    
hr_numerical = hr[['satisfaction_level',"last_evaluation","average_montly_hours"]] 

#following the process in datasets notebook
X = hr_numerical
lof = LocalOutlierFactor(n_neighbors=2000, contamination='auto')
y_pred = lof.fit_predict(X)
outliers = np.where(y_pred == -1)

hr = hr.drop(outliers[0]) #remove them from my db
print('Rows Removed: {:.0f}'.format(len(outliers[0])))


# ### 2.4 Plotting to analyze possible scale and skewness

# In[2741]:


## For Numerical Variables

hr_numerical = hr[['satisfaction_level',"last_evaluation" ,"number_project","average_montly_hours","time_spend_company"]]

hr_numerical.hist(figsize=(12, 12), bins=50, xlabelsize=8, ylabelsize=8)


# #### Even though number_project and time_spend_company are numerical, the range is very low so it would be interested to consider them as categorical further. I will not be correcting for normality at the moment but I will rather discretize them in the feature engineering process.

# In[2742]:


## EDA For Binary
sales_cols = hr.loc[:,(hr.columns.str.startswith("sales"))]
salary_cols = hr.loc[:,(hr.columns.str.startswith("salary"))]
hr_binary= pd.concat([sales_cols.reset_index(drop=True), salary_cols,hr_raw[["left","Work_accident","promotion_last_5years"]]], axis=1)
hr_binary.hist(figsize=(15, 15), bins=2, xlabelsize=6, ylabelsize=8)


# #### Clearly there is a high unbalance for most of our dummy variables. For out target, left , the unbalance does not seem to be high enough to treat it (like oversampling, for example), but for the enconding for the variable sales, there are many departments and many of the highly underrepresented. This will be taken into consideration for feature enginering as well.

# ### 2.5 Scaling

# Only the next 3 variables will be considered to scale.

# In[2820]:


scaler = MinMaxScaler()
scaler.fit(hr[['average_montly_hours']])
scaler.fit(hr[['last_evaluation']])
scaler.fit(hr[['satisfaction_level']])

# Transforming my data
hr['average_montly_hours'] = scaler.transform(hr[['average_montly_hours']])
hr['last_evaluation'] = scaler.transform(hr[['last_evaluation']])
hr['satisfaction_level'] = scaler.transform(hr[['satisfaction_level']])

#redefine my numerical df to include changes
hr_numerical = hr[['satisfaction_level',"last_evaluation"                         ,"number_project","average_montly_hours","time_spend_company"]]


# ### 2.6 Skewness

# In[2821]:


def skewed_features(self, threshold=0.75, fix=False, return_series=True):

        df = self
        feature_skew = df.apply(
            lambda x: skew(x)).sort_values(ascending=False)

        if fix is True:
            high_skew = feature_skew[feature_skew > threshold]
            skew_index = high_skew.index
            for feature in skew_index:
                self = boxcox1p(
                    df[feature], boxcox_normmax(df[feature] + 1))
        if return_series is True:
            return feature_skew
        
print(skewed_features(hr_numerical))


# In[2822]:


## Lets fix skewness for time spent at company, we still can discretize further.
pt = PowerTransformer(method='yeo-johnson')

pt.fit(hr[['time_spend_company']])

hr['time_spend_company'] = pt.transform(hr[['time_spend_company']])

#redefine my numerical to include changes
hr_numerical = hr[['satisfaction_level',"last_evaluation"  ,"number_project","average_montly_hours","time_spend_company"]]


# # 3. Baseline

# In[2823]:


target = hr[['left']]
features = hr.loc[:, hr.columns != 'left']

X = features
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=27)
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))


# ##### The metric that I will be trying to improve through this process is the f-1 score for the weighted average. I want to have a good prediction level for both workers that Churn and don't churn. As expected the f-1 score is much higher for my class0, as this data presents some class imbalance.

# ## 4. Feature Engineering

# ## 4.0 Setting up cross validation

# For now, my cross-validation will be very straightforward. I will have 10 folds and score according to the f1 weighted score, as mentioned before. Lets start with cross validating the baseline and storing the results in a new dataframe.

# In[2824]:


scores = cross_validate(logreg, X_train, y_train, cv=10, scoring='f1_weighted')
results = {'model':["baseline"],"test_score": 0.79, "mean_cv_score": scores['test_score'].mean(), "max_cv_score" : scores['test_score'].max()}
results_df = pd.DataFrame(results , columns = ["model","test_score","mean_cv_score","max_cv_score"])
results_df.plot(x='model', y=["test_score","mean_cv_score"], kind="bar")
print('Best F1: {:.5f}'.format(results_df["max_cv_score"][0]))
print('Average F1: {:.5f}'.format(results_df["mean_cv_score"][0]))


# ### Good! The average F1 in my cross validation is slightly higher than the one in the test data. The average is 0.78 and it reaches a max of 0.80 across the 10 folds. This is my baseline.

# # 4.1 Feature Construction

# ### 4.1.1 Clustering with target = sales

# As we saw on the EDA, the variable "sales", which represents the department of the workers the classes are highly unbalanced, and we have 13 of them. Why if we try clustering our model, ignoring this column, to try to capture relationships between departments which are not reflected in the "sales" variable? 

# First I am going to split my dataset into training and testing and calculate the clusters for each.
# As we learned in class, we can't train a cluster model with unseen data! 

# In[2825]:


cluster_train = X_train
cluster_train['left'] = y_train
cluster_train = cluster_train.loc[:,~cluster_train.columns.str.startswith('sales')] #only dropping columns which start with sales
cluster_test = X_test

cluster_test['left'] = y_test 
cluster_test = cluster_test.loc[:,~cluster_test.columns.str.startswith('sales')]


# In[2826]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=0).fit(cluster_train) #cluster for train data
X_train["kmeans"] = kmeans.labels_
kmeans_test = KMeans(n_clusters=5, random_state=0).fit(cluster_test) # cluster for test data
X_test["kmeans"] = kmeans_test.labels_

temp = pd.concat((X_train["kmeans"],X_test["kmeans"]), axis=0) #putting them together
temp = pd.DataFrame(temp)

hr_fe = pd.merge(hr,temp, left_index=True, right_index=True) #merging my index on hr_fe, which will keep changes of feature engineering


hr_fe = pd.get_dummies(hr_fe, columns = ['kmeans'], prefix ='kmeans') #being a categorical variable, I will get the dummies for it as well
hr_fe = hr_fe.loc[:,~hr_fe.columns.str.startswith('sales')] #and drop the sales variables.


# In[2827]:


#Lets define X and y again as the values have changed
target = hr_fe[['left']]
features = hr_fe.loc[:, hr_fe.columns != 'left']

X = features
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=1)
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))


# In[2828]:


scores = cross_validate(logreg, X_train, y_train, cv=10, scoring='f1_weighted')
scores["test_score"].mean()

print('Best F1: {:.5f}'.format(scores["test_score"].max()))
print('Average F1: {:.5f}'.format(scores["test_score"].mean()))


# ## Nice! On our cross validation our best score went up to 0.82. It is consistent with the f1-score in out test set of 0.81. Clustering after dropping the departments worked.

# ### 4.1.2 Lets binarize the time spent at the company
# 

# In[2493]:


hr_fe["time_spend_company"].hist(figsize=(4, 4), bins=50, xlabelsize=8, ylabelsize=8)


# I will use the benchmarks on the histogram to categorize into a high, medium and low time spent at the company and with these values try to have a better balance between the groups of people.

# In[2829]:


hr_fe["time_spend_company_low"] = np.where(hr_fe['time_spend_company']<-1,1,0)
hr_fe["time_spend_company_medium"] = np.where(((hr_fe['time_spend_company']>-1) & (hr_fe['time_spend_company']<1) ),1,0)
hr_fe["time_spend_company_high"] = np.where(hr_fe['time_spend_company']>1,1,0)


# In[2830]:


features = hr_fe.loc[:, hr_fe.columns != 'left']
features = features.loc[:, features.columns != 'time_spend_company'] #lets drop our original variable for this model

X = features
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=27)
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
scores = cross_validate(logreg, X_train, y_train, cv=10, scoring='f1_weighted')
scores["test_score"].mean()

y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))


# In[2831]:


scores = cross_validate(logreg, X_train, y_train, cv=10, scoring='f1_weighted')
scores["test_score"].mean()

print('Best F1: {:.5f}'.format(scores["test_score"].max()))
print('Average F1: {:.5f}'.format(scores["test_score"].mean()))


# ### Good. Both our testing and average training score improved. Grouping the time spent at company seems to better predict the effect on churning than considering the amount of years (corrected by skewness). 

# ### 4.1.2 Now lets binarize number of projects
# 

# In[2499]:


hr_fe["number_project"].hist(figsize=(4, 4), bins=50, xlabelsize=8, ylabelsize=8)


# Let's do something similar to what we did with the time spent and group by low medium and high. This could help to balance the number of observations for each value and better capture the workload itself instead of just a number of projects

# In[2832]:


#collections.Counter(hr_raw["time_spend_company"])
hr_fe["number_projects_low"] = np.where(hr_fe['number_project']<3,1,0)
hr_fe["number_projects_medium"] = np.where(((hr_fe['number_project']==3)| (hr_fe['number_project']==4) ),1,0)
hr_fe["number_projects_high"] = np.where(hr_fe['number_project']>4,1,0)


# In[2833]:


target =hr_fe[['left']]

features = hr_fe.loc[:, hr_fe.columns != 'left']
features = features.loc[:, features.columns != 'time_spend_company']
features = features.loc[:, features.columns != 'number_project']

X = features
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=27)
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))


# In[2834]:


scores = cross_validate(logreg, X_train, y_train, cv=10, scoring='f1_weighted')
scores["test_score"].mean()

print('Best F1: {:.5f}'.format(scores["test_score"].max()))
print('Average F1: {:.5f}'.format(scores["test_score"].mean()))


# ## Good, there is some significant improvement. Grouping the number of projects by a smaller number of groups seem to better predict the Churn than just using each value.

# ### 4.1.3 Adding variable interactions
# 

# When adding variable interactions, we are capturing the effect of one variable (lets say, satisfaction levels), at different levels of another variable (lets say, Work accidents).The hyphothesis is that depending on if the person had an accident or not, the effect of the satisfaction level on the Churn will be different. For these variables I will try a few combinations : we have many binary features and groups of features that impact the effect of other features at different levels.

# In[2835]:


hr_fe["1_1"] = hr_fe["satisfaction_level"] * hr_fe["last_evaluation"]
hr_fe["1_2"] = hr_fe["satisfaction_level"] * hr_fe["time_spend_company_high"]
hr_fe["1_3"] = hr_fe["last_evaluation"] * hr_fe["average_montly_hours"]
hr_fe["1_4"] = hr_fe["last_evaluation"] * hr_fe["kmeans_0"]
hr_fe["1_5"] = hr_fe["last_evaluation"] * hr_fe["time_spend_company_medium"]
hr_fe["1_6"] = hr_fe["average_montly_hours"] * hr_fe["salary_medium"]
hr_fe["1_7"] = hr_fe["average_montly_hours"] * hr_fe["number_projects_low"]
hr_fe["1_8"] = hr_fe["last_evaluation"] * hr_fe["promotion_last_5years"]
hr_fe["1_9"] = hr_fe["last_evaluation"] * hr_fe["salary_medium"]
hr_fe["1_10"] = hr_fe["satisfaction_level"] * hr_fe["salary_medium"]
hr_fe["1_11"] = hr_fe["salary_high"] * hr_fe["number_projects_medium"]
hr_fe["1_12"] = hr_fe["time_spend_company_low"] * hr_fe["number_projects_high"]


# In[2836]:


target =hr_fe[['left']]

features = hr_fe.loc[:, hr_fe.columns != 'left']
features = features.loc[:, features.columns != 'time_spend_company']
features = features.loc[:, features.columns != 'number_project']


X = features
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=27)
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))


# In[2837]:


scores = cross_validate(logreg, X_train, y_train, cv=10, scoring='f1_weighted')
scores["test_score"].mean()

print('Best F1: {:.5f}'.format(scores["test_score"].max()))
print('Average F1: {:.5f}'.format(scores["test_score"].mean()))


# ### Disclaimer : the process of variables interaction creation was manual and iterative, I tried a few combinations that maximized my score. Overall, both the test and train f1 scores improved. Now that we are done with feature creation, lets add the results to compare to the baseline.

# In[2841]:


results = {'model':["feature_creation"],"test_score": 0.90, "mean_cv_score": scores['test_score'].mean(), "max_cv_score" : scores['test_score'].max()}
results_df = results_df.append(results,ignore_index=True)
#results_df
#results_df = results_df.drop(1)
results_df.plot(x='model', y=["test_score","mean_cv_score"], kind="bar")


# # 4.3 Feature Selection

# ### 4.3.1 Relief

# Here we are going to use an adapted version(for pandas) found on the dataset github repository for the class. We have created a lot of new features (numerical) so lets redefine our numerical features. This algorithm throws the most importance features to do feature selection on our model.

# In[2786]:


hr_fe_numerical = hr_fe.select_dtypes(include=['float64'])
hr_fe_numerical =hr_fe_numerical.drop(["time_spend_company"],axis=1) # we are dropping this one for our relief algorithm as we hot encoded it earlier.


# In[2787]:


from skrebate import ReliefF

num_features=10
num_neighbors=200
abs_imp=False

if num_features is None:
            num_features = len(hr_fe_numerical.columns)
if num_neighbors is None:
            num_neighbors = 20
assert num_features <= len(hr_fe_numerical.columns),             "Larger nr of features ({}) than available ({})".format(
                num_features, len(hr_fe_numerical.columns))
assert target is not None,             "Target feature must be specified before computing importance"
assert num_neighbors <= hr_fe.shape[0],             "Larger nr of neighbours than samples ({})".format(
                hr_fe.shape[0])

my_features = hr_fe_numerical.values  # the array inside the dataframe
my_labels = target.values.ravel()  # the target as a 1D array.

fs = ReliefF(n_features_to_select=num_features,
                     n_neighbors=num_neighbors)

fs.fit_transform(my_features, my_labels)

if abs_imp is True:
            importances = abs(fs.feature_importances_[:num_features])
else:
            importances = fs.feature_importances_[:num_features]
        
indices = np.argsort(importances)[:num_features]


# Now, lets identify which are these features. Here are the sorted importance indexes...

# In[2788]:


relief = pd.DataFrame(indices,importances)
relief =relief.reset_index(drop=False)
relief = pd.concat((relief,pd.DataFrame(hr_fe_numerical.columns)), axis=1)
relief.sort_values(by=['index'], ascending=False)


# #### One more time, lets run our model without the features by the ReliefFalgorithm (those with Nan)

# In[2789]:


target =hr_fe[['left']]

#For my dummy encodings, I am going to delete one of each to avoid multicolinearity
features = hr_fe.loc[:, hr_fe.columns != 'left']
features = features.loc[:, features.columns != '1_8']
features = features.loc[:, features.columns != '1_9']
features = features.loc[:, features.columns != '1_10']


X = features
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=27)
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))


# In[2790]:


scores = cross_validate(logreg, X_train, y_train, cv=10, scoring='f1_weighted')
scores["test_score"].mean()

print('Best F1: {:.5f}'.format(scores["test_score"].max()))
print('Average F1: {:.5f}'.format(scores["test_score"].mean()))


# #### Both the test and train f1 score decreased. Maybe this is not the best algorithm to do feature selection for this problem. Let's see what happens if we do something similar only for the categorical variables

# ## 4.3.2 Chi Squared

# We are going to use Chi Squared to filter specifically for categorical variables, as it is the purpose of this method, which uses their correlations to filter by importance. 

# In[2795]:


#hr_fe_cat = hr_fe.select_dtypes(include=['int64'])
hr_fe_cat =hr_fe[["Work_accident","promotion_last_5years","salary_high","salary_low","salary_medium","kmeans_0","kmeans_1","kmeans_2","kmeans_3","kmeans_4","time_spend_company_low",                 "time_spend_company_medium","time_spend_company_high","number_projects_low", "number_projects_medium","number_projects_high","1_11","1_12"]]


# In[2796]:


#from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

hr_fe.columns
#hr_fe_categorical = hr_fe.select_dtypes(include=['uint8'])#hr_fe_categorical  = hr_fe_categorical.drop(["left","number_project"], axis=1)

chi_results = sklearn.feature_selection.chi2(hr_fe_cat, y)

chi2 = pd.concat((pd.DataFrame(hr_fe_cat.columns),pd.DataFrame(chi_results[0])), axis=1)
chi2 = pd.concat((chi2,pd.DataFrame(chi_results[1])), axis=1)
chi2


# All of the correlations have a very small pvalue. Now lets remove from the model the variables with a coefficient (second column), lower than 50. 

# In[2801]:


target =hr_fe[['left']]

#For my dummy encodings, I am going to delete one of each to avoid multicolinearity
features = hr_fe.loc[:, hr_fe.columns != 'left']
features = features.loc[:, features.columns != 'time_spent_company']
features = features.loc[:, features.columns != 'salary_medium']

features = features.loc[:, features.columns != 'kmeans_3']
features = features.loc[:, features.columns != 'time_spend_company_medium']
features = features.loc[:, features.columns != 'number_project']


X = features
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=27)
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))


# In[2802]:


scores = cross_validate(logreg, X_train, y_train, cv=10, scoring='f1_weighted')
scores["test_score"].mean()

print('Best F1: {:.5f}'.format(scores["test_score"].max()))
print('Average F1: {:.5f}'.format(scores["test_score"].mean()))


# #### The f1 score for both testing and training is still lower than before the feature selection. Lets try a wrapper method for this same purpose and check if it performs better.

# ### 4.3.3 Wrapper method : forward/backward selection

# I will use the algorithm in the github of the class. Again this is an adapted version for pandas dataframes.

# In[2803]:


#initial_list=None,
threshold_in=0.01,
threshold_out=0.05,
verbose=False
"""
        Perform a forward/backward feature selection based on p-value from
        statsmodels.api.OLS
        Your features must be all numerical, so be sure to onehot_encode them
        before calling this method.
        Always set threshold_in < threshold_out to avoid infinite looping.
        All features involved must be numerical and types must be float.
        Target variable must also be float. You can convert it back to a
        categorical type after calling this method.
        :parameter initial_list: list of features to start with (column names
            of X)
        :parameter threshold_in: include a feature if its
            p-value < threshold_in
        :parameter threshold_out: exclude a feature if its
            p-value > threshold_out
        :parameter verbose: whether to print the sequence of inclusions and
            exclusions
        :return: List of selected features
        Example::
            my_data.stepwise_selection()
        See <https://en.wikipedia.org/wiki/Stepwise_regression>
        for the details
        Taken from: <https://datascience.stackexchange.com/a/24823>
"""
if initial_list is None:
            initial_list = []
if len(hr_fe_categorical) != 0:
            print('Considering only numerical features')

        # assert self.target.dtype.name == 'float64'

included = list(hr_fe_numerical)

while True:
            changed = False
            # forward step
            excluded = list(set(hr_fe_numerical.columns) - set(included))
            new_pval = pd.Series(index=excluded)
            for new_column in excluded:
                model = sm.OLS(hr_fe['left'], sm.add_constant(hr_fe_numerical[included + [new_column]])).fit()
                new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < 0.01:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print('Add  {:30} with p-value {:.6}'.format(best_feature,
                                                                 best_pval))
            # backward step
            model = sm.OLS(hr_fe['left'], sm.add_constant(
                pd.DataFrame(hr_fe_numerical[included]))).fit()
            # use all coefs except intercept
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()  # null if p-values is empty
            if worst_pval > 0.05:
                changed = True
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                if verbose:
                    print('Drop {:30} with p-value {:.6}'.format(worst_feature,
                                                                 worst_pval))
            if not changed:
                break


# In[2804]:


included


# ### I will run the regression once again considering only the features to include according to the algorithm

# In[2805]:


target =hr_fe[['left']]
features=hr_fe[['satisfaction_level', 'last_evaluation', 'average_montly_hours', '1_1', '1_2', '1_3', '1_4',"1_5",  '1_7', '1_8', '1_9']]
#For my dummy encodings, I am going to delete one of each to avoid multicolinearity


X = features
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=27)
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))




# In[2806]:


scores = cross_validate(logreg, X_train, y_train, cv=10, scoring='f1_weighted')
scores["test_score"].mean()

print('Best F1: {:.5f}'.format(scores["test_score"].max()))
print('Average F1: {:.5f}'.format(scores["test_score"].mean()))


# ## The f1 score for both train and test decreased slightly! Lets go back to our best model so far (before Relief,Chi) and try a very straightforward method for feature selection: only keeping features with a pvalue greater than 0.05

# Disclaimer: for this regression I removed one of the columns for my hot encondings, as multicolinearity, even though does not bias the estimators, might make them unstable and I faced issues to obtain the p-value without removing them.

# In[2842]:


target =hr_fe[['left']]

features = hr_fe.loc[:, hr_fe.columns != 'left']
features = features.loc[:, features.columns != 'time_spend_company']
features = features.loc[:, features.columns != 'number_project']
features = features.loc[:, features.columns != 'number_projects_medium'] #
features = features.loc[:, features.columns != 'time_spend_company_medium']
features = features.loc[:, features.columns != 'kmeans_4']
features = features.loc[:, features.columns != 'salary_medium']

X = features
y = target

logit_model=sm.Logit(y, X)
result=logit_model.fit(method='bfgs')
print(result.summary2())

to_remove = result.pvalues[result.pvalues > 0.05].index.tolist()
to_remove


# In[2843]:


target =hr_fe[['left']]

features = hr_fe.loc[:, hr_fe.columns != 'left']
features = features.loc[:, features.columns != 'time_spend_company']
features = features.loc[:, features.columns != 'number_project']
features = features.loc[:, features.columns != 'number_projects_high'] #
features = features.loc[:, features.columns != 'time_spend_company_high']
features = features.loc[:, features.columns != 'kmeans_4']
features = features.loc[:, features.columns != 'salary_high']

features = features.loc[:, features.columns != 'promotion_last_5years']
features = features.loc[:, features.columns != 'kmeans_0']
features = features.loc[:, features.columns != '1_4']
features = features.loc[:, features.columns != '1_8']
#features = features.loc[:, features.columns != '1_10']
features = features.loc[:, features.columns != '1_11']
features = features.loc[:, features.columns != '1_12']



X = features
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=27)
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))


# In[2844]:


scores = cross_validate(logreg, X_train, y_train, cv=10, scoring='f1_weighted')
scores["test_score"].mean()

print('Best F1: {:.5f}'.format(scores["test_score"].max()))
print('Average F1: {:.5f}'.format(scores["test_score"].mean()))


# In[2845]:


results = {'model':["feature_selection"],"test_score": 0.91, "mean_cv_score": scores['test_score'].mean(), "max_cv_score" : scores['test_score'].max()}
results_df = results_df.append(results,ignore_index=True)
#results_df
#results_df = results_df.drop(1)
results_df.plot(x='model', y=["test_score","mean_cv_score"], kind="bar")


# We have a small improvement and a simpler model. Now lets go to our final step: regularization

# # 5. Hyperparameter tunning (with regularization!)

# We will penalize our model using Ridge (l2) and will do gridsearch to find the best lambda (C). For the gridsearch we will use 50 samples to generate , starting at -5 and going until 5. 

# In[2850]:


grid={"C":np.logspace(-5, 5, 30),"penalty":["l2","none"]}

logreg = LogisticRegression(solver='lbfgs', max_iter=250)

logreg_cv=GridSearchCV(logreg,grid,cv=10,scoring='f1_weighted')

logreg_cv.fit(X_train,y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)
results= pd.DataFrame(logreg_cv.cv_results_)
results


# The gridsearch is telling us that adding a penalty is better than not! What if we use these parameters and plug them in our test data?

# In[2870]:


X = features
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=27)

logreg = LogisticRegression(solver='lbfgs', C=78.84, penalty="l2")

y_pred = logreg.predict(X_test)

print(classification_report(y_test, y_pred))


# In[2871]:


scores = cross_validate(logreg, X_train, y_train, cv=10, scoring='f1_weighted')
scores["test_score"].mean()

print('Best F1: {:.5f}'.format(scores["test_score"].max()))
print('Average F1: {:.5f}'.format(scores["test_score"].mean()))


# In[2848]:


results = {'model':["ridge"],"test_score": 0.89, "mean_cv_score": scores['test_score'].mean(), "max_cv_score" : scores['test_score'].max()}
results_df = results_df.append(results,ignore_index=True)
#results_df
#results_df = results_df.drop(1)
results_df.plot(x='model', y=["test_score","mean_cv_score"], kind="bar")


# ## We have achieved an average f1 weighted score of 0.928 and through gridsearch a maximum of 0.947

# # 6. Model Interpretation

# In[2878]:


logit_model=sm.Logit(y_test, X_test)
result=logit_model.fit(method='bfgs',c=78.84)
result.summary2()


# Our final model reached an average accuracy of 0.928. The accuracy for the class 0 was approximately 0.93 while the accuracy for the 1 class was around 0.75, which is normal for datasets which are not balanced. All and all, its harder to predict people who Churn than people who don't Churn. That being said there are many parameters in our final model which are highly significant, for example the satisfaction level, which has a negative coefficient. This means that higher levels of satisfaction level are associated with less probability to Churn, same for the last evaluation and the average monthly hours. Looking at binary varialbes like Work_accident, it is less likely to Churn if there has not been an accident. For low salaries, it is more likely to Churn for this categorization compared to the other two. When looking at the clusters , it is more likely to Churn for the first and second cluster, but less likely from the third when comparing to the other clusters. For the time spend in the company, for people who have stayed little time, it is more likely to Churn, while people who have spend medium levels are more likely to Churn than the other two. When it comes to the number of projects, people who have a low number of projects are more likely to Churn while people who have a medium number are less likely. 

# In[ ]:




