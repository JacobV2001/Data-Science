# Import needed libraries
import pandas as pd
import numpy as np

# import sleep data
sleep_data = pd.read_csv('student_sleep_patterns.csv')

print(sleep_data.info())
print(sleep_data.describe(include='all'))
print(sleep_data.head(5))


# Dropping Student_ID feature as it would provide no value to our algorithm
sleep_data.drop(columns='Student_ID', inplace=True)
print(sleep_data.dtypes)

# Create list with columns for future reference

numerical_cols = ['Age', 'Sleep_Duration',
       'Study_Hours', 'Screen_Time', 'Caffeine_Intake', 'Physical_Activity',
       'Sleep_Quality', 'Weekday_Sleep_Start', 'Weekend_Sleep_Start',
       'Weekday_Sleep_End', 'Weekend_Sleep_End']

categorical_cols = ['Gender', 'University_Year']

# Iterate through each feature
for col in sleep_data.columns:
    print(col)                          # Print Name
    print(sleep_data[col].unique())     # Print Unique Values
    print(sleep_data[col].describe())   # Print Info On Feature
    print('\n')



# Save Old Categorical Data
old_cat = sleep_data[['Gender', 'University_Year']]

# Create a new col for each feature except the first which occurs when all other vars are false
sleep_data = pd.get_dummies(sleep_data, columns=categorical_cols, drop_first=True)

# Reset Categorical Columns
categorical_cols = ['Gender_Male', 'Gender_Other',
       'University_Year_2nd Year', 'University_Year_3rd Year',
       'University_Year_4th Year']



# Import necessary addons
import matplotlib.pyplot as plt
import seaborn as sns

# Create a feature for weekday & weekend sleep
sleep_data['Weekday_Sleep'] = sleep_data['Weekday_Sleep_Start'] - sleep_data['Weekday_Sleep_End']
sleep_data['Weekend_Sleep'] = sleep_data['Weekend_Sleep_Start'] - sleep_data['Weekend_Sleep_End']
numerical_cols.extend(['Weekday_Sleep', 'Weekend_Sleep']) # Add to numerical_cols




for col in numerical_cols:
    # Create Subplots plotting histogram & box & whiskers for each feature
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)

    # Histogram of Feature
    sns.histplot(sleep_data[col], bins=20, kde=True)
    plt.title('Histogram of {}'.format(col))
    plt.xlabel(col)
    plt.ylabel('Frequency')

    # Boxplot of Feature
    plt.subplot(1, 2, 2)
    sns.boxplot(x=sleep_data[col])
    plt.title('Boxplot of {}'.format(col))
    plt.xlabel(col)  # Adjust the label based on your data

    plt.tight_layout()
    plt.show()


# Loop through each column in old_cat
for col in old_cat:
    # Get value counts
    counts = old_cat[col].value_counts()
    
    # Plot
    counts.plot(kind='bar')
    
    # Add titles and labels
    plt.title('Distribution of {}'.format(col))
    plt.xlabel(col)
    plt.ylabel('Counts')
    
    # Show counts on the bars
    for index, value in enumerate(counts):
        plt.text(index, value, str(value), ha='center', va='bottom')

    plt.show()



# Calculate the correlation matrix for numerical features
correlation_matrix = sleep_data[numerical_cols].corr()

# Print the correlation with Sleep Quality
print("Numerical Features Correlation with Sleep Quality:")

# Create a heatmap for numerical features
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix for Numerical Features')
plt.show()


for col in numerical_cols:
    if col == 'Sleep_Quality': continue
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x='Sleep_Quality', y=col, data=sleep_data, alpha=0.7, size=3)  # Adjust size as needed
    plt.title('{} by Sleep Quality'.format(col))
    plt.xlabel('Sleep Quality (1-10)')
    plt.ylabel(col)
    plt.show()


from sklearn.model_selection import train_test_split

# Splitting data into x and y
x = sleep_data.drop(columns=['Sleep_Quality'])
y = sleep_data['Sleep_Quality']

# Setting the Split with .2 of values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)



# Import necessary modules
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


# Create models
Forest_Model = RandomForestRegressor(max_features= 3, n_estimators=100, random_state=12)
SVM_Model = SVR()
Linear_Model = LinearRegression()


# Define the hyperparameter grid for Random Forest
forest_param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_features': [1, 2, 3, 'sqrt'],  # Number of features 
    'max_depth': [None, 10, 20, 30],  # Maximum depth 
    'min_samples_split': [2, 5, 10]  # Minimum number of samples required
}


forest_grid_search = GridSearchCV(RandomForestRegressor(random_state=12), 
                                   forest_param_grid, # Grid defined above
                                   cv=5, # number of cross-validations
                                   scoring='neg_root_mean_squared_error') # Score with Neg. RSME

# Fit the grid search to the training data
forest_grid_search.fit(x_train, y_train)

# Retrieve the best estimator (model) found during the search
best_forest_model = forest_grid_search.best_estimator_


# Hyperparameter tuning for SVM
svm_param_grid = {
    'C': [0.1, 1, 10], # Regularization parameter
    'gamma': ['scale', 'auto', 0.01, 0.1], # Kernel 
    'kernel': ['linear', 'rbf'] # Kernel type
}

svm_grid_search = GridSearchCV(SVR(), 
                                svm_param_grid, 
                                cv=5, 
                                scoring='neg_root_mean_squared_error')

# Fit the grid search to the training data
svm_grid_search.fit(x_train, y_train)

# Retrieve the best estimator (model) found during the search
best_svm_model = svm_grid_search.best_estimator_


# Import necessary modules for tuning
from sklearn.linear_model import Ridge

# Hyperparameter tuning for Ridge Regression
ridge_param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0] # Regularization Strength
}

ridge_grid_search = GridSearchCV(Ridge(), 
                                  ridge_param_grid, 
                                  cv=5, 
                                  scoring='neg_root_mean_squared_error')

# Fit the grid search to the training data
ridge_grid_search.fit(x_train, y_train)

# Retrieve the best estimator (model) found during the search
best_ridge_model = ridge_grid_search.best_estimator_


Forest_y_pred_tuned = best_forest_model.predict(x_test)
SVM_y_pred_tuned = best_svm_model.predict(x_test)
Linear_y_pred_tuned = best_ridge_model.predict(x_test)

# Create Root Mean Squared Error Function
def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Evaluate the classifier
Forest_RMSE = calculate_rmse(y_test, Forest_y_pred_tuned)
SVM_RMSE = calculate_rmse(y_test, SVM_y_pred_tuned)
Linear_RMSE = calculate_rmse(y_test, Linear_y_pred_tuned)

# Print RMSE values
print("Random Forest Regression RMSE: {}".format(Forest_RMSE))
print("SVM Regression RMSE: {}".format(SVM_RMSE))
print("Linear Regression RMSE: {}".format(Linear_RMSE))


# Create a DataFrame for predictions
data = {
    'Actual': y_test,
    'Random Forest': Forest_y_pred_tuned,
    'SVM': SVM_y_pred_tuned,
    'Linear Regression': Linear_y_pred_tuned
}

predictions_df = pd.DataFrame(data)

# Print the DataFrames
print("Predictions DataFrame:")
print(predictions_df.head(10))

print(predictions_df.describe())
