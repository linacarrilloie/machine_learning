#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


import xgboost as xgb

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# load data
X_train = pd.read_csv('train_features.csv')
y_train = pd.read_csv('train_labels.csv')
X_test = pd.read_csv('test_features.csv')
y_test = pd.read_csv('submission_format.csv')

# merge features and labels on train set
train = X_train.copy()
train = train.merge(y_train, how = 'left', on = 'id')


# In[3]:


# column to always drop
columns_to_drop = [

    'subvillage',
    'region_code',
    'district_code',
    'wpt_name',
    'recorded_by',
    'scheme_name',
    'management_group',
    'payment',
    'extraction_type_group',
    'extraction_type_class',
    'waterpoint_type_group',
    'quality_group',
    'quantity_group',
    'source_type',
    'source_class',
    'num_private', 
    'date_recorded',
  
]


# In[4]:


# columns to drop for now
additional_columns_to_drop = [
    'funder',
    'installer',
    'amount_tsh',
    'lga',
    'ward',
    'scheme_management'
]


# In[5]:


X_train.drop(columns_to_drop, axis = 1, inplace = True)
X_train.drop(additional_columns_to_drop, axis = 1, inplace = True)

X_test.drop(columns_to_drop, axis = 1, inplace = True)
X_test.drop(additional_columns_to_drop, axis = 1, inplace = True)


# In[6]:


# one-hot encoding
X_train = pd.get_dummies(X_train, 
                         prefix = X_train.select_dtypes('object').columns, 
                         columns = X_train.select_dtypes('object').columns,
                         drop_first = True
                        )

X_test = pd.get_dummies(X_test, 
                         prefix = X_test.select_dtypes('object').columns, 
                         columns = X_test.select_dtypes('object').columns,
                         drop_first = True
                        )

# power transformation of numerical columns
numerical_columns = X_train.select_dtypes(['int64', 'float64']).columns

pt = PowerTransformer()
X_train.loc[:,numerical_columns] = pt.fit_transform(X_train.loc[:,numerical_columns])
X_test.loc[:,numerical_columns] = pt.transform(X_test.loc[:,numerical_columns])

# add columns to test set that only exist in train set
X_test[list(set(X_train.columns).difference(set(X_test.columns)))[0]] = 0

# make sure columns are in the same order
X_train = X_train[sorted(X_train.columns)].copy()
X_test = X_test[sorted(X_test.columns)].copy()


# creating a mapping for the classes
classes = {
    'functional' : 0,
    'non functional' : 1,
    'functional needs repair' : 2
}

# create the inverse mapping
classes_inv = {v: k for k, v in classes.items()}

# map the target to numerical
y_train = y_train.status_group.map(classes)


# In[7]:


import featuretools as ft
es = ft.EntitySet(id = 'id')

BOOL = ft.variable_types.Boolean

variable_types = {
    'basin_Lake Nyasa': BOOL,
    'basin_Lake Rukwa': BOOL,
    'basin_Lake Tanganyika': BOOL,
    'basin_Lake Victoria': BOOL,
    'basin_Pangani': BOOL,
    'basin_Rufiji': BOOL,
    'basin_Ruvuma / Southern Coast': BOOL,
    'basin_Wami / Ruvu': BOOL,
    'extraction_type_cemo': BOOL,
    'extraction_type_climax': BOOL,
    'extraction_type_gravity': BOOL,
    'extraction_type_india mark ii': BOOL,
    'extraction_type_india mark iii': BOOL,
    'extraction_type_ksb': BOOL,
    'extraction_type_mono': BOOL,
    'extraction_type_nira/tanira': BOOL,
    'extraction_type_other': BOOL,
    'extraction_type_other - mkulima/shinyanga': BOOL,
    'extraction_type_other - play pump': BOOL,
    'extraction_type_other - rope pump': BOOL,
    'extraction_type_other - swn 81': BOOL,
    'extraction_type_submersible': BOOL,
    'extraction_type_swn 80': BOOL,
    'extraction_type_walimi': BOOL,
    'extraction_type_windmill': BOOL,
    'management_other': BOOL,
    'management_other - school': BOOL,
    'management_parastatal': BOOL,
    'management_private operator': BOOL,
    'management_trust': BOOL,
    'management_unknown': BOOL,
    'management_vwc': BOOL,
    'management_water authority': BOOL,
    'management_water board': BOOL,
    'management_wua': BOOL,
    'management_wug': BOOL,
    'payment_type_monthly': BOOL,
    'payment_type_never pay': BOOL,
    'payment_type_on failure': BOOL,
    'payment_type_other': BOOL,
    'payment_type_per bucket': BOOL,
    'payment_type_unknown': BOOL,
    'permit_True': BOOL,
    'public_meeting_True': BOOL,
    'quantity_enough': BOOL,
    'quantity_insufficient': BOOL,
    'quantity_seasonal': BOOL,
    'quantity_unknown': BOOL,
    'region_Dar es Salaam': BOOL,
    'region_Dodoma': BOOL,
    'region_Iringa': BOOL,
    'region_Kagera': BOOL,
    'region_Kigoma': BOOL,
    'region_Kilimanjaro': BOOL,
    'region_Lindi': BOOL,
    'region_Manyara': BOOL,
    'region_Mara': BOOL,
    'region_Mbeya': BOOL,
    'region_Morogoro': BOOL,
    'region_Mtwara': BOOL,
    'region_Mwanza': BOOL,
    'region_Pwani': BOOL,
    'region_Rukwa': BOOL,
    'region_Ruvuma': BOOL,
    'region_Shinyanga': BOOL,
    'region_Singida': BOOL,
    'region_Tabora': BOOL,
    'region_Tanga': BOOL,
    'source_hand dtw': BOOL,
    'source_lake': BOOL,
    'source_machine dbh': BOOL,
    'source_other': BOOL,
    'source_rainwater harvesting': BOOL,
    'source_river': BOOL,
    'source_shallow well': BOOL,
    'source_spring': BOOL,
    'source_unknown': BOOL,
    'water_quality_fluoride': BOOL,
    'water_quality_fluoride abandoned': BOOL,
    'water_quality_milky': BOOL,
    'water_quality_salty': BOOL,
    'water_quality_salty abandoned': BOOL,
    'water_quality_soft': BOOL,
    'water_quality_unknown': BOOL,
    'waterpoint_type_communal standpipe': BOOL,
    'waterpoint_type_communal standpipe multiple': BOOL,
    'waterpoint_type_dam': BOOL,
    'waterpoint_type_hand pump': BOOL,
    'waterpoint_type_improved spring': BOOL,
    'waterpoint_type_other': BOOL

}


# In[8]:



es_train = es.entity_from_dataframe(entity_id = 'entity_id', dataframe = X_train, 
                              index = 'id', variable_types=variable_types)

es_test = es.entity_from_dataframe(entity_id = 'entity_id1', dataframe = X_test, 
                              index = 'id',variable_types=variable_types)


# In[12]:


features, feature_names = ft.dfs(entityset=es, target_entity='entity_id', agg_primitives=["mean"]
                        ,trans_primitives = ["and"]
                                 , max_depth = 2)

features_test, feature_names_test = ft.dfs(entityset=es_test, target_entity='entity_id1', agg_primitives=["mean"]
                       ,trans_primitives = ["and"]
                                           , max_depth = 2)


# In[91]:


features


# In[92]:


features_test


# In[14]:


features = features.reset_index()
features_test = features_test.reset_index()

features  = features.drop("id", axis=1)
features_test = features_test.drop("id", axis=1)


# In[16]:


from sklearn.decomposition import PCA
pca = PCA(.95)

pca.fit(features)
len(pca.components_)


# In[40]:


pca_2 = PCA(n_components=30)
pca_2.fit(features)
len(pca.components_)


# In[43]:


len(pca_2.components_)


# In[22]:


X_train = pca.transform(features)
X_test = pca.transform(features_test)


# In[38]:


X_test = pca.transform(features_test)


# In[26]:


from sklearn.neighbors import KNeighborsClassifier

# create a knn classifier
knn = KNeighborsClassifier()

# create a param grid
param_grid = {'n_neighbors' : [7]}

# do a grid search to find the best parameter
grid_knn = GridSearchCV(
    estimator = knn,
    param_grid = param_grid,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 10,
    refit = True,
    return_train_score = True
)

# fit the model
grid_knn.fit(X_train, y_train)

# read results of grid search into dataframe
cv_results_df = pd.DataFrame(grid_knn.cv_results_)

# print results
cv_results_df[['params', 'mean_test_score', 'mean_train_score']].sort_values(by = ['mean_test_score'], ascending = False)


# In[ ]:


pca_2 = PCA(n_components=30)
pca_2.fit(features)
len(pca.components_)


# In[44]:


features


# In[46]:


X_train_2 = pca_2.transform(features)
X_test_2 = pca_2.transform(features_test)


# In[49]:


y_train


# In[54]:


from sklearn.neighbors import KNeighborsClassifier

# create a knn classifier
knn = KNeighborsClassifier()

# create a param grid
param_grid = {'n_neighbors' : [1,3,5]}

# do a grid search to find the best parameter
grid_knn = GridSearchCV(
    estimator = knn,
    param_grid = param_grid,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 10,
    refit = True,
    return_train_score = True
)

# fit the model
grid_knn.fit(X_train_2, y_train)

# read results of grid search into dataframe
cv_results_df = pd.DataFrame(grid_knn.cv_results_)

# print results
cv_results_df[['params', 'mean_test_score', 'mean_train_score']].sort_values(by = ['mean_test_score'], ascending = False)


# In[50]:


# XGBOOST


# In[55]:


param_test = {
 'max_depth':[7,8],
 'min_child_weight':[1,2],
 'num_class' : [3]
}


xgb_model = xgb.XGBClassifier(learning_rate=0.1, 
                              n_estimators=140, 
                              gamma=0, 
                              #max_depth = 14,
                              #min_child_weight = 2,
                              num_class = 3,
                              subsample=0.8, 
                              colsample_bytree=0.8,
                              objective= 'multi:softmax', 
                              nthread=4, 
                              scale_pos_weight=1,
                              seed=27)

gsearch = GridSearchCV(estimator = xgb_model, 
                       param_grid = param_test, 
                       scoring='accuracy',
                       n_jobs=4,
                       cv=5,
                       refit = True,
                       return_train_score = True)


# In[56]:



train_model_9 = gsearch.fit(X_train_2, y_train)


# In[57]:


pd.DataFrame(gsearch.cv_results_)[['params', 'mean_test_score', 'mean_train_score']]


# In[ ]:


##### ROUND 2 ######


# In[67]:



features_2, feature_names = ft.dfs(entityset=es, target_entity='entity_id',agg_primitives =["mean","skew","max","std"]
                        ,trans_primitives = ["and"], max_depth = 2)

features_test_2, feature_names_test = ft.dfs(entityset=es_test, target_entity='entity_id1',agg_primitives =["mean","skew","max","std"]
                        ,trans_primitives = ["and"], max_depth = 2)


# In[68]:


features_2 = features_2.reset_index()
features_test_2 = features_test_2.reset_index()

features_2  = features_2.drop("id", axis=1)
features_test_2 = features_test_2.drop("id", axis=1)


# In[75]:


pca_3 = PCA(n_components=30)
pca_3.fit(features_2)


# In[78]:


X_train_3 = pca_3.transform(features_2)
X_test_3 = pca_3.transform(features_test_2)


# In[79]:


from sklearn.neighbors import KNeighborsClassifier

# create a knn classifier
knn = KNeighborsClassifier()

# create a param grid
param_grid = {'n_neighbors' : [3,5,7]}

# do a grid search to find the best parameter
grid_knn = GridSearchCV(
    estimator = knn,
    param_grid = param_grid,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 10,
    refit = True,
    return_train_score = True
)

# fit the model
grid_knn.fit(X_train_3, y_train)

# read results of grid search into dataframe
cv_results_df = pd.DataFrame(grid_knn.cv_results_)

# print results
cv_results_df[['params', 'mean_test_score', 'mean_train_score']].sort_values(by = ['mean_test_score'], ascending = False)


# In[86]:


param_test = {
 'max_depth':[10,11],
 'min_child_weight':[2,3],
 'num_class' : [3]
}


xgb_model = xgb.XGBClassifier(learning_rate=0.1, 
                              n_estimators=140, 
                              gamma=0, 
                              #max_depth = 14,
                              #min_child_weight = 2,
                              num_class = 3,
                              subsample=0.8, 
                              colsample_bytree=0.8,
                              objective= 'multi:softmax', 
                              nthread=4, 
                              scale_pos_weight=1,
                              seed=27)

gsearch = GridSearchCV(estimator = xgb_model, 
                       param_grid = param_test, 
                       scoring='accuracy',
                       n_jobs=4,
                       cv=5,
                       refit = True,
                       return_train_score = True)


# In[87]:



train_model_9 = gsearch.fit(X_train_3, y_train)


# In[88]:


pd.DataFrame(gsearch.cv_results_)[['params', 'mean_test_score', 'mean_train_score']]


# In[89]:


y_pred = grid_knn.best_estimator_.predict(X_test_3)
y_pred_df = pd.DataFrame(y_pred)
y_pred_df[0].value_counts()


# In[90]:


y_pred = gsearch.best_estimator_.predict(X_test_3)
y_pred_df = pd.DataFrame(y_pred)
y_pred_df[0].value_counts()


# In[ ]:




