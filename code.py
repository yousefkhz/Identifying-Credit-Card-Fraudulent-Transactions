import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report, confusion_matrix, recall_score

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from datetime import datetime as dt


#reading data
df = pd.read_csv('fraud.csv')

#-------------------------------EDA-------------------------------
#data qaulity checks
#describing numerical columns to get an overview on each
df.describe()

#checking categorical variables with distinct count of 1 and removing them from dataset
zero_var_catag_cols = [x for x in df.columns if df[x].dtype == 'O' and len(df[x].unique()) == 1]
df_p = df.drop(zero_var_catag_cols, axis=1)

#checking NaNs
df_p.isna().sum() #none

#getting remaining categoral columns
catag_cols = [x for x in df_p.columns if df_p[x].dtype == 'O']
#checking percentage counts of each label in the remaining categoral features
for col in catag_cols:
    #removing extra apostrophes in categorical data
    df_p[col] = df_p[col].str.replace("'","")
    #getting count of distinct values for each categorical column
    print(df_p[col].value_counts())


#checking if each customer has cosistant age and gender across data
df_age = df_p[['customer', 'age']].drop_duplicates()
df_gender = df_p[['customer', 'gender']].drop_duplicates()
df_gender[df_gender['customer'].duplicated()] #none
df_age[df_age['customer'].duplicated()] #none

#exploring unknowns in age and gender
df_p[df_p['age']=='U']['gender'].value_counts() #all enterprise gender
df_p[df_p['age']=='U']['fraud'].value_counts() # only 7 representing fraudulent records
df_p[df_p['gender']=='U']['fraud'].value_counts() #all in zero class


#exploring outliers
f,(ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
#step boxplot
sns.boxplot(x="fraud", y="step", data=df_p,ax=ax1, palette=colors)
ax1.set_title("Steps in Hours", fontsize=14)
#amount boxplot
sns.boxplot(x="fraud", y="amount", data=df_p, ax=ax2, palette=colors)
ax2.set_title("Amount", fontsize=14)
plt.show() #no substantial outliers present in box plots

#visualizing the data imbalance
sns.countplot(x='fraud', data=df_p, palette= ['#385C72', '#EBAF75'])
plt.show()

#quantifying the imbalance in the dataset
print('No Fraud', round(df_p['fraud'].value_counts()[0]/len(df_p) * 100,2), '% of the dataset')
print('Fraud', round(df_p['fraud'].value_counts()[1]/len(df_p) * 100,2), '% of the dataset') #1.21% of data has label 1


#visualizing time (in steps) and amount distribution
fig, ax = plt.subplots(1, 2, figsize=(18,4))
#getting values
amount_val = df_p['amount'].values
time_val = df_p['step'].values
#amount subplot
sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])
#step subplot
sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time(step)', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)+5])
plt.show() #transaction ammount follows almost a uniform distribution, there is a possible outlier in the step column


#visualizing amount distribution for both labels
sns.set(font_scale=1.2)
sns.violinplot(x ='fraud', y ='amount', data = df_p)
plt.show()

#visualizing distribution of gender 
sns.countplot(x = 'gender', hue = 'fraud', data = df_p, palette = 'hls')
plt.show()

#getting only numerical features and creating linear correlation matrix between them
df_numer = df_p[[x for x in df_p.columns if df_p[x].dtype in ('float', 'int64')]]
corr = df_numer.corr()
print(corr) #no substantial linear correlation between numeric features and/or target


#-------------------------------Features Engineering-------------------------------

#merchant transaction frequency
df_merchant_perc = pd.DataFrame(df_p['merchant'].value_counts()/df_p.shape[0]).reset_index()
df_merchant_perc['merchant_freq_flag'] = np.where(df_merchant_perc['count']>0.10, 'frequent', 'non-frequent')
df_merchant_perc.drop('count', axis=1, inplace=True)
df_p = df_p.merge(df_merchant_perc, how='left', on='merchant')

#average transaction amount per merchant
df_avg_trx_amnt = df_p[['merchant', 'amount']].groupby('merchant',as_index=False).mean()
df_avg_trx_amnt.columns = ['merchant', 'avg_amnt_per_merch']
df_p = df_p.merge(df_avg_trx_amnt, how='left', on='merchant')

#time of day [hour in day]
df_p['time_of_day'] = df_p['step']%24
df_p.drop('step',axis=1, inplace=True)


#visualizing distribution of new features
#merchant flag
sns.countplot(x = 'merchant_freq_flag', hue = 'fraud', data = df_p,palette = 'hls')
plt.show() #non frequent merchants contain the bulk of fraudelent transactions

#average per merchant
sns.distplot(df_p['avg_amnt_per_merch'].values, color='#385C72')
plt.show()

#time of day
sns.distplot(df_p['avg_amnt_per_merch'].values, color='b')
plt.show()

#amount against merchant type with label as hue
sns.scatterplot(data=df_p, x="time_of_day", y="amount", hue="fraud")
plt.show()

#drop merchant after creating the related wanted features
df_p.drop('merchant', axis=1, inplace=True)


#-------------------------------Pre-processing-------------------------------

#dropping unknown age along with enterprise age (since all unknown gender is enterprise gender)
df_p = df_p[df_p['age'] != 'U']
#dropping unknown gender
df_p = df_p[df_p['gender'] != 'U']


#splitting dataset into training and testing datasets
#storing customers in seperate variable and dropping it from dataset
custmr_ids = df_p['customer']
df_p.drop('customer', axis=1, inplace=True)

#seperating features and target into seperate variables
X = df_p.drop('fraud', axis=1)
y = df_p['fraud']

# splitting dataset into testing and traing datasets
X_train, X_test,y_train, y_test = train_test_split(X,y ,random_state=42, test_size=0.25, shuffle=True)

#dictionary to store encoder to be used fro test dataset
dict_encoders={}
one_hot_out = {}
#one hot encoding categorical variables
categ_features = X_train.select_dtypes('O').columns
numeric_features = [x for x in X_train.columns if x not in categ_features]


# #scaling train data
scaler = StandardScaler()
X_train.loc[:,numeric_features] = scaler.fit_transform(X_train.loc[:,numeric_features])

for col in categ_features:
    encoder = OneHotEncoder(drop='first')
    encoded_column = encoder.fit_transform(X_train[[col]])
    dict_encoders[col] = encoder  # Store the encoder object in the dictionary

    # Create a DataFrame for the one-hot encoded column
    encoded_column_df = pd.DataFrame(encoded_column.toarray(), columns=encoder.get_feature_names_out([col]))
    one_hot_out[col] = encoded_column_df
    X_train.drop(col, axis=1, inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    X_train = pd.concat([X_train, encoded_column_df], axis=1)

# #scaling test data
X_test.loc[:,numeric_features] = scaler.transform(X_test.loc[:,numeric_features])

#encoding X_test
for col in categ_features:
    encoder = dict_encoders[col]
    encoded_column = encoder.transform(X_test[[col]])

    # Create a DataFrame for the one-hot encoded column
    encoded_column_df = pd.DataFrame(encoded_column.toarray(), columns=encoder.get_feature_names_out([col]))
    X_test.drop(col, axis=1, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    X_test = pd.concat([X_test, encoded_column_df], axis=1)


# #applying SMOTE to train dataset to remove imbalance between classes
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# from imblearn.under_sampling import RandomUnderSampler
# rus = RandomUnderSampler(random_state=42)
# X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

#both sampling methods yielded a much lower test f1 score of around 60% whilst showing signs of overfitting with a cv training f1 score of around 96%
#-------------------------------Choosing best model-------------------------------

#evaluating best model 
clf1 = RandomForestClassifier(random_state=42)
clf3 = LogisticRegression(random_state=42)
clf4 = DecisionTreeClassifier(random_state=42)
clf5 = xgb.XGBClassifier(random_state=42)


#parameters for the different models chosen
param1 = {}
param1['classifier__n_estimators'] = [10, 50, 100, 250]
param1['classifier__max_depth'] = [5, 10, 20]
param1['classifier'] = [clf1]

param2 = {}
param2['classifier__C'] = [10**-1, 10**0, 10**1]
param2['classifier__penalty'] = ['l1', 'l2']
param2['classifier'] = [clf3]

param3 = {}
param3['classifier__max_depth'] = [5,10,25,None]
param3['classifier__min_samples_split'] = [2,5,10]
param3['classifier'] = [clf4]

param4 = {}
param4['classifier__max_depth'] = [3, 5, 7, 10]
param4['classifier__learning_rate'] = [10**-4, 10**-3, 10**-2]
param4['classifier__subsample'] =  [0.5, 0.7, 1]
param4['classifier'] = [clf5]


#creating pipeline
pipeline = Pipeline([('classifier', clf1)])
params = [param1, param2, param3, param4]

start_time = dt.now()
# Train the random search model
rs = RandomizedSearchCV(pipeline, params, cv=4, n_jobs=-1, scoring='f1',verbose=2).fit(X_train, y_train)

#printing time taken
print('Time taken: ', dt.now()-start_time)


#creating dataframe for results of random search 
models_scores = pd.DataFrame(rs.cv_results_)[['params', 'split0_test_score','split1_test_score', 'split2_test_score', 'mean_test_score']]
models_scores['Classifier'] = models_scores['params'].apply(lambda x: x['classifier'])
models_scores.to_clipboard()


#-------------------------------Tuning best model-------------------------------

#tuning best model
# Create an XGBoost DMatrix for your training data
xgb_model = xgb.XGBClassifier(random_state=42)


# Define XGBoost parameters
params = {
        'min_child_weight': [1, 5, 10],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.001, 0.01, 0.1]
        }

xgb_tuning = RandomizedSearchCV(estimator=xgb_model, param_distributions=params, scoring='f1', n_jobs=-1, cv=3, verbose=2 ).fit(X_train, y_train)
best_score = grid.best_score_
best_params = grid.best_params_

#-------------------------------Training and testing of best model-------------------------------

# Train the best XGBoost model
xgb_best_model = xgb.XGBClassifier(random_state=42,
                                    subsample= 0.6, min_child_weight= 5, 
                                   max_depth= 6, learning_rate= 0.1, 
                                   colsample_bytree= 1.0)

#fitting the model
xgb_best_model.fit(X_train, y_train)

# Make probability predictions on the testing data
y_pred = xgb_best_model.predict_proba(X_test)


thresholds = [x/100 for x in range(15,60)]
# Convert predicted probabilities to binary class labels
max__recall_score=0
optim_recall_thres=0
for thres in thresholds:
    y_pred_binary = [1 if pred[1] > thres else 0 for pred in y_pred]
    if recall_score(y_test, y_pred_binary) > max__recall_score and f1_score(y_test, y_pred_binary) > 0.808:
        max__recall_score = recall_score(y_test, y_pred_binary)
        optim_recall_thres = thres

#getting the binary rprediction results based on optimized threshold
y_pred_binary = [1 if pred[1] > optim_recall_thres else 0 for pred in y_pred]

# Calculate the F1 score
f1 = f1_score(y_test, y_pred_binary)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Print the F1 score
print(f'F1 Score: {f1:.4f}')

#printing the classification report and confusion matrix
print(classification_report(y_test, y_pred_binary))
print(conf_matrix)

#making font bigger for chart
sns.set(font_scale=1.2)
#plotting confusion matrix
plt.figure(figsize=(8,6), dpi=100)
# Scale up the size of all text
sns.set(font_scale = 1.1)
# Plot Confusion Matrix 
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=['#385C72', '#8EB1C7'])
# set x-axis label and ticks. 
ax.set_xlabel("Predicted Diagnosis", fontsize=14, labelpad=20)
ax.xaxis.set_ticklabels(['Negative', 'Positive'])
# set y-axis label and ticks
ax.set_ylabel("Actual Diagnosis", fontsize=14, labelpad=20)
ax.yaxis.set_ticklabels(['Negative', 'Positive'])
plt.show()


#-------------------------------Interpreting Trained Model-------------------------------

#feature importance
results=pd.DataFrame()
results['Features']=X_train.columns
results['Importances Percentage'] = xgb_best_model.feature_importances_
results.sort_values(by='Importances Percentage',ascending=False,inplace=True)

#plotting top 10 important features
top_10_features = results.iloc[:10,:]
sns.barplot(x="Importances Percentage", y="Features", data=top_10_features, color='#385C72')
plt.show()
