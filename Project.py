# -*- coding: utf-8 -*-
''' Data Mining project'''

#remove warnings
import warnings
warnings.filterwarnings("ignore")

#Imports
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.svm import LinearSVR
import seaborn as sns
from sklearn import preprocessing




#function for split the DataFrame in data complete and incomplete
def split(data_insurance, reset_index = False):
    data_insurance_complete = pd.DataFrame()
    data_insurance_incomplete = data_insurance[data_insurance.isna().any(axis=1)]
    if reset_index:
        data_insurance_incomplete.reset_index(inplace=True)
        data_insurance_incomplete.drop('index', axis=1, inplace=True)
    data_insurance_complete = data_insurance[~data_insurance.isna().any(axis=1)]
    return data_insurance_complete, data_insurance_incomplete


#Function that uses KNN to classify the missing values on CATEGORICAL columns
def classify_categorical_data(data_insurance, categorical_columns):
    data_insurance_complete, data_insurance_incomplete = split(data_insurance, reset_index=True)

    
    #Creating a classifier to fill the categorical Educational Degree Data
    for index, value in enumerate(categorical_columns):
        #index = 0
        #value = categorical_columns[0]
        clf = KNeighborsClassifier(n_neighbors=5)    
        
        incomplete = deepcopy(data_insurance_incomplete)
        complete = deepcopy(data_insurance_complete)
        
        X_train, X_test, y_train, y_test = train_test_split(complete.loc[:,complete.columns != value].values,
                                                            complete.loc[:,value].values, test_size = 0.2, random_state = 0)
        
        trained_model = clf.fit(X_train, 
                                 y_train)
        
        #fill the numerical columns with the column mean
        incomplete.loc[:, ~incomplete.columns.isin(categorical_columns) ] = incomplete.loc[:, 
                                ~incomplete.columns.isin(categorical_columns)].apply(lambda column: column.fillna(column.mean()), axis=0)
        
        #Round Age and First Policy's Year
        incomplete['Age'] = incomplete['Age'].apply(lambda x:round(x))
        incomplete['First Policy´s Year'] =  incomplete['First Policy´s Year'].apply(lambda x:round(x))
        
        
        #Categorical columns insted the one we want to predic
        cat_without_the_column = deepcopy(categorical_columns)
        cat_without_the_column.pop(index)
        
        #Fill the categorical columns insted the one we want to predict with the column mode
        incomplete.loc[:, incomplete.columns.isin(cat_without_the_column) ] = incomplete.loc[:, 
                        incomplete.columns.isin(cat_without_the_column)].apply(lambda column: column.fillna(int(column.mean())), axis=0)

        
        
        prediction = trained_model.predict(incomplete.loc[:,incomplete.columns != value])
        temp_df = pd.DataFrame(prediction.reshape(-1,1), columns = [value])
        
        
        #now we are filling data_arrivals_incomplete 
        for ind in range(len(temp_df)):
            if np.isnan(data_insurance_incomplete[value][ind]):
                data_insurance_incomplete[value][ind] = temp_df[value][ind]


    #and filling the nan's on arrivals_df
    dataset = pd.concat([data_insurance_complete, data_insurance_incomplete])
    dataset.set_index(dataset['Customer Identity'] - 1, inplace=True)
    
    return dataset




#funcion for checking which algorithm is the best for using on each column for NUMERICAL columns
def checking_choices(data_insurance, number_of_tests=10):
    data_insurance_complete, data_insurance_incomplete = split(data_insurance)

    choices = []
    better_for_each_column = []
    
    #testing
    for i in range(number_of_tests):
        choices.append(regressor_test(data_insurance))
    
    #chosing the best algorithm for each column
    for i in range(len(data_insurance.columns)):
        l = []
        for j in range(len(choices)):
            l.append(choices[j][i])
        better_for_each_column.append(max(set(l), key = l.count))
        
    return better_for_each_column




#function for test which regressor is best for each numerical column
#Return a list of lists with the best algorithm for each test (choose the number of tests on checking_choices function )
def regressor_test(data_insurance):
#variables to hold the Mean Squared Errors for each model
    kn_errors = []
    linear_errors = []
    svr_errors = []    
    
    complete,incomplete = split(data_insurance)
    
    for i in complete.columns:
            
        X_train, X_test, y_train, y_test = train_test_split(complete.loc[:,complete.columns != i].values,
                                                            complete.loc[:,i].values, test_size = 0.2, random_state = 0)
        
        regressor1 = KNeighborsRegressor(5, 
                                       weights ='distance', 
                                       metric = 'euclidean')
        regressor2= LinearRegression()
        regressor3=LinearSVR()
        
        
        KN_trained_model1 = regressor1.fit(X_train, 
                                 y_train)
        Linear_trained_model2 = regressor2.fit(X_train, 
                                 y_train)
        SVR_trained_model3 = regressor3.fit(X_train, 
                                 y_train)  
        
        incomplete_2 = deepcopy(incomplete)
        incomplete_2.loc[:, incomplete.columns != i] = incomplete_2.loc[:, 
                                incomplete.columns != i].apply(lambda row: row.fillna(row.mean()), axis=1)

        y_pred1 = regressor1.predict(X_test)
        y_pred2 = regressor2.predict(X_test)
        y_pred3 = regressor3.predict(X_test)
        
        
        kn_errors.append(math.sqrt(mean_squared_error(y_test, y_pred1)))
        linear_errors.append(math.sqrt(mean_squared_error(y_test, y_pred2)))
        svr_errors.append(math.sqrt(mean_squared_error(y_test, y_pred3)))
        
        
    #ROOT MEAN SQUARED ERROR 
    RMSE= []

    #Filling RMSE for each column
    for i in range(0, len(complete.columns)):
        l = []
        l.extend((kn_errors[i], linear_errors[i], svr_errors[i]))
        
        if min(l) == kn_errors[i]:
            RMSE.append("KNN")
        elif min(l) == linear_errors[i]:
            RMSE.append("Linear")
        elif min(l) == svr_errors[i]:
            RMSE.append("SVR")
    


    return RMSE



#function to apply the regressors
def apply_regressors(choices, data_insurance, numerical_columns):

    complete,incomplete = split(data_insurance)

    for i,v in enumerate(complete.columns):
        
        #Check if it is a numerical column
        if v in numerical_columns:
            
            #use the choosen algorithm 
            if choices[i] == 'KNN':
                regressor = KNeighborsRegressor(5, 
                                                weights ='distance', 
                                                metric = 'euclidean')
            elif choices[i] == 'SVR':
                regressor = LinearSVR()
                
            elif choices[i] == 'Linear':
                regressor = LinearRegression()
                
            #Split in train-test data    
            X_train, X_test, y_train, y_test = train_test_split(complete.loc[:,complete.columns != v].values,
                                                                complete.loc[:,v].values, test_size = 0.2, random_state = 0)
            #Train the model
            trained_model = regressor.fit(X_train, 
                                     y_train)
            
            #Make predictions
            incomplete_2 = deepcopy(incomplete)
            incomplete_2.loc[:, incomplete.columns != v] = incomplete_2.loc[:, 
                                    incomplete.columns != v].apply(lambda row: row.fillna(row.mean()), axis=1)
            
            prediction = trained_model.predict(incomplete_2.loc[:,incomplete_2.columns != v])
            temp_df = pd.DataFrame(prediction.reshape(-1,1), columns = [v])
            
            #fill NaN's on data_arrivals_incomplete 
            for index in range(len(temp_df)):
                if np.isnan(incomplete.iloc[index,i]):
                    incomplete.iloc[index,i] = temp_df[v][index]



    #and filling the nan's on arrivals_df
    dataset = pd.concat([complete, incomplete])
    dataset.set_index(dataset['Customer Identity'] - 1, inplace=True)
    
    
    return dataset



#_________________________Cleaning and Filling the Data with the algorithms___________________________________________

#Read the dataset
insurance_df = pd.read_csv('https://raw.githubusercontent.com/apanchot/data_mining/master/A2Z_Insurance.csv')

#Create Age column
insurance_df['Age'] = insurance_df.loc[:, 'Brithday Year'].apply(lambda x : 2019 - x )

#Drop Birthday Year and Customer Id
insurance_df.drop(['Brithday Year'], axis=1, inplace=True)

#Drop rows with more than 3 NaN's
insurance_df.dropna(thresh=(len(insurance_df.columns) - 3), inplace=True, axis=0)

categorical_columns = ['Educational Degree', 'Geographic Living Area','Has Children (Y=1)']

numerical_columns = ['Customer Identity','First Policy´s Year','Gross Monthly Salary',
                     'Customer Monetary Value','Claims Rate', 'Premiums in LOB: Motor',
                     'Premiums in LOB: Household','Premiums in LOB: Health','Premiums in LOB:  Life',
                     'Premiums in LOB: Work Compensations', 'Age']



data_insurance = deepcopy(insurance_df)

#Encoding Educational Degree and returning back the NaN's
data_insurance['Educational Degree'] = data_insurance['Educational Degree'].apply(str)

labelencoder_X = LabelEncoder()

data_insurance.loc[:,'Educational Degree'] = labelencoder_X.fit_transform(data_insurance.loc[:,'Educational Degree'])

data_insurance['Educational Degree'] = data_insurance['Educational Degree'].apply(lambda x : np.nan if x == 4 else x )

#Fill categorical data with the KNN predited Values
data_insurance = classify_categorical_data(data_insurance, categorical_columns)

#Fill numerical data with the best regressor algorithm
data_insurance = apply_regressors(checking_choices(data_insurance),data_insurance, numerical_columns)


#Full dataset
data_insurance.isnull().sum()




#_________________________Checking the distributions, correlations and outliers___________________________________________

#Age x Premiuns
# 7195 is an outlier or wrong filling for age.
sns.set_style("ticks")
sns.pairplot(data_insurance[['Age',
                             'Premiums in LOB: Motor',
                             'Premiums in LOB: Household',
                             'Premiums in LOB: Health',
                             'Premiums in LOB:  Life',
                             'Premiums in LOB: Work Compensations']],
            diag_kind='hist',
            kind='scatter',
            palette="husl",
           plot_kws = {'alpha': 0.6,
                      's': 20,
                      'edgecolor':'k'},
           height=2)

plt.show()


#Eduaction x Premiuns
sns.set_style("ticks")
sns.pairplot(data_insurance[['Educational Degree',
                             'Premiums in LOB: Motor',
                             'Premiums in LOB: Household',
                             'Premiums in LOB: Health',
                             'Premiums in LOB:  Life',
                             'Premiums in LOB: Work Compensations']],
            diag_kind='hist',
            kind='scatter',
            palette="husl",
           plot_kws = {'alpha': 0.6,
                      's': 20,
                      'edgecolor':'k'},
           height=2)

plt.show()

#Gross Monthly x Premiuns
#Here we can see 2 outliers on GMS index 5882 and 8261 
sns.set_style("ticks")
sns.pairplot(data_insurance[['Gross Monthly Salary',
                             'Premiums in LOB: Motor',
                             'Premiums in LOB: Household',
                             'Premiums in LOB: Health',
                             'Premiums in LOB:  Life',
                             'Premiums in LOB: Work Compensations']],
            diag_kind='hist',
            kind='scatter',
            palette="husl",
           plot_kws = {'alpha': 0.6,
                      's': 20,
                      'edgecolor':'k'},
           height=2)

plt.show()

#CMV x Premiuns
#Here we can see 1 outlier on GMS index 171
sns.set_style("ticks")
sns.pairplot(data_insurance[['Customer Monetary Value',
                             'Premiums in LOB: Motor',
                             'Premiums in LOB: Household',
                             'Premiums in LOB: Health',
                             'Premiums in LOB:  Life',
                             'Premiums in LOB: Work Compensations']],
            diag_kind='hist',
            kind='scatter',
            palette="husl",
           plot_kws = {'alpha': 0.6,
                      's': 20,
                      'edgecolor':'k'},
           height=2)

plt.show()

#Claims Rate x Premiuns
#Again the 171 is an outlier for Claim Rate, since its the opposite of CMV
sns.set_style("ticks")
sns.pairplot(data_insurance[['Claims Rate',
                             'Premiums in LOB: Motor',
                             'Premiums in LOB: Household',
                             'Premiums in LOB: Health',
                             'Premiums in LOB:  Life',
                             'Premiums in LOB: Work Compensations']],
            diag_kind='hist',
            kind='scatter',
            palette="husl",
           plot_kws = {'alpha': 0.6,
                      's': 20,
                      'edgecolor':'k'},
           height=2)

plt.show()


#Just Premiums
#We have to check each of this outliers
# 5293 for Motor
# 8866 for Household
# 9149 for Health
# 7961 and 7988 for Work Compensation
sns.set_style("ticks")
sns.pairplot(data_insurance[['Premiums in LOB: Motor',
                             'Premiums in LOB: Household',
                             'Premiums in LOB: Health',
                             'Premiums in LOB:  Life',
                             'Premiums in LOB: Work Compensations']],
            diag_kind='hist',
            kind='scatter',
            palette="husl",
           plot_kws = {'alpha': 0.6,
                      's': 20,
                      'edgecolor':'k'},
           height=2)

plt.show()


#Those are the outliers that i could identify, we should talk about what to do with them
outliers = [7195,5882,8261,171,5293,8866,9149,7961]








#_________________________Encoding the data ___________________________________________________________________
#Saving the column names
columns_list = ['Basic','High School', 'BSc/MSc','PhD','Area 1','Area 2','Area 3','Area 4','No Kids','Have Kids']
for i in insurance_df.columns:
    if i not in categorical_columns:
        columns_list.append(i)

#Should we use dummy variables on educational degree ? My opinion is Yes !
onehotencoder = OneHotEncoder(categorical_features = [2,4,5])
encoded_data = pd.DataFrame(onehotencoder.fit_transform(data_insurance).toarray())

#Give the column names back
encoded_data.columns = columns_list

#Drop identity not needed anymore
encoded_data.drop('Customer Identity', axis=1, inplace=True)
columns_list.remove('Customer Identity')





#_________________________Standadizing the data ___________________________________________________________________

#Should we standardize or normalize the data ? (depends on which ML algorithm we will use)
# Create the Scaler object
scaler = preprocessing.StandardScaler()

# Scaling 
scaled_df = scaler.fit_transform(encoded_data.loc[:,'First Policy´s Year':])
scaled_df = pd.DataFrame(scaled_df)

scaled_data = pd.concat([encoded_data.loc[:,:'Customer Identity'],scaled_df], axis=1)  

scaled_data.columns = columns_list












