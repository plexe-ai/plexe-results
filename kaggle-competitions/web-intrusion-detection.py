# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Set random seed for reproducibility
SEED = 42

# Paths to the data files
PATH_TO_DATA = '/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/'
TRAIN_DATA = os.path.join(PATH_TO_DATA, 'train_sessions.csv')
TEST_DATA = os.path.join(PATH_TO_DATA, 'test_sessions.csv')
SITE_DICT = os.path.join(PATH_TO_DATA, 'site_dic.pkl')

# Load the site dictionary
with open(SITE_DICT, 'rb') as f:
    site2id = pickle.load(f)

# Create an inverse mapping from site IDs to site names
id_to_site = {v: k for k, v in site2id.items()}
id_to_site[0] = 'unknown'

# Load the training and test data
times = ['time{}'.format(i) for i in range(1, 11)]
sites = ['site{}'.format(i) for i in range(1, 11)]

train_df = pd.read_csv(TRAIN_DATA, index_col='session_id', parse_dates=times)
test_df = pd.read_csv(TEST_DATA, index_col='session_id', parse_dates=times)

# Sort the training data by the first timestamp
train_df = train_df.sort_values(by='time1')

# Fill NaN values and convert site columns to integers
train_df[sites] = train_df[sites].fillna(0).astype(int)
test_df[sites] = test_df[sites].fillna(0).astype(int)

# Prepare the sessions as strings of site names
def prepare_sessions(df):
    return df[sites].apply(lambda row: ' '.join([id_to_site.get(site_id, 'unknown') for site_id in row]), axis=1)

train_sessions = prepare_sessions(train_df)
test_sessions = prepare_sessions(test_df)

# Create TF-IDF features with parameters from your original code
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=29000, tokenizer=lambda x: x.split())
X_train_sites = vectorizer.fit_transform(train_sessions)
X_test_sites = vectorizer.transform(test_sessions)

# Extract the target variable
y_train = train_df['target'].astype(int).values

# Initialize list to collect feature names
features_name = []

# Add the detailed time features
def add_time_features(times_df, X_sparse):
    hour = times_df['time1'].apply(lambda t: 100 * t.hour + t.minute) / 1000
    morning_1 = (((hour >= 0.901) & (hour <= 0.904) | (hour >= 0.922) & (hour <= 1.209)).astype('int') * hour).values.reshape(-1, 1)
    morning_2 = (((hour >= 0.905) & (hour <= 0.921)).astype('int') * hour).values.reshape(-1, 1)
    day_1 = (((hour >= 1.210) & (hour <= 1.239)).astype('int') * hour).values.reshape(-1, 1)
    day_2 = (((hour >= 1.240) & (hour <= 1.335)).astype('int') * hour).values.reshape(-1, 1)
    day_3 = (((hour >= 1.336) & (hour <= 1.358)).astype('int') * hour).values.reshape(-1, 1)
    day_4 = (((hour >= 1.359) & (hour <= 1.517)).astype('int') * hour).values.reshape(-1, 1)
    day_5 = (((hour >= 1.518) & (hour <= 1.553)).astype('int') * hour).values.reshape(-1, 1)
    evening_1 = (((hour >= 1.554) & (hour <= 1.629) | (hour >= 1.705) & (hour <= 1.755)) * hour).values.reshape(-1, 1)
    evening_2 = ((hour >= 1.653) & (hour <= 1.704)).astype('int').values.reshape(-1, 1)
    evening_3 = (((hour >= 1.756) & (hour <= 1.828) | (hour >= 1.626) & (hour <= 1.656)) * hour).values.reshape(-1, 1)
    night = (((hour >= 1.829) & (hour <= 2.359) | (hour >= 0) & (hour <= 0.900)) * hour).values.reshape(-1, 1)
    
    objects_to_hstack = [X_sparse, morning_1, morning_2, day_1, day_2, day_3, day_4, day_5,
                         evening_1, evening_2, evening_3, night]
    feature_names = ['morning_1', 'morning_2', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5',
                     'evening_1', 'evening_2', 'evening_3', 'night']
    
    X = hstack(objects_to_hstack)
    return X, feature_names

# Add day of week features
def add_day_month(times_df, X_sparse):
    day_of_week = times_df['time1'].apply(lambda t: t.weekday())
    day_of_week_df = pd.get_dummies(day_of_week)
    day_of_week_df['5_6'] = day_of_week_df[5] + day_of_week_df[6]
    day_of_week_df['2_3'] = day_of_week_df[2] + day_of_week_df[3]
    
    for d in (2,3,5,6):
        del day_of_week_df[d]
    
    day_of_week_df = day_of_week_df.rename({i: 'weekday_' + str(i) for i in day_of_week_df.columns}, axis=1)
    
    objects_to_hstack = [X_sparse, day_of_week_df]
    feature_names = list(day_of_week_df.columns)
    
    X = hstack(objects_to_hstack)
    return X, feature_names

# Add day of month features
def add_dom(times_df, X_sparse):
    dom = times_df['time1'].apply(lambda ts: ts.day)
    dom_1 = (dom.isin([3,5,6,7,8,10,11,12,21,23,27,28,30])).astype(int).values.reshape(-1, 1)
    dom_2 = (dom.isin([9,24])).astype(int).values.reshape(-1, 1)
    dom_3 = (dom.isin([17,18,19,20,21,22,24,25,26,31])).astype(int).values.reshape(-1, 1)
    
    objects_to_hstack = [X_sparse, dom_1, dom_2, dom_3]
    feature_names = ['dom_1', 'dom_2', 'dom_3']
    
    X = hstack(objects_to_hstack)
    return X, feature_names

# Add the detailed time features
X_train_final, time_feat_names = add_time_features(train_df, X_train_sites)
X_test_final, _ = add_time_features(test_df, X_test_sites)
features_name += time_feat_names

# Add day of week features
X_train_final, dow_feat_names = add_day_month(train_df, X_train_final)
X_test_final, _ = add_day_month(test_df, X_test_final)
features_name += dow_feat_names

# Add day of month features
X_train_final, dom_feat_names = add_dom(train_df, X_train_final)
X_test_final, _ = add_dom(test_df, X_test_final)
features_name += dom_feat_names

# Define all feature names
feature_names = vectorizer.get_feature_names_out()
all_features = np.concatenate([feature_names, features_name])

# Initialize TimeSeriesSplit
time_split = TimeSeriesSplit(n_splits=10)

# Initialize Logistic Regression model with parameters from your code
final_model = LogisticRegression(C=20, random_state=SEED, solver='liblinear')

# Cross-validation with TimeSeriesSplit
cv_scores = cross_val_score(final_model, X_train_final, y_train, cv=time_split, scoring='roc_auc', n_jobs=-1)
print('CV scores:', cv_scores)
print('Mean CV score:', cv_scores.mean())

# Fit the model on the entire training data
final_model.fit(X_train_final, y_train)

# Make predictions on the test set
y_test_pred = final_model.predict_proba(X_test_final)[:, 1]

# Prepare submission
submission = pd.DataFrame({'session_id': test_df.index, 'target': y_test_pred})
submission.to_csv('submission.csv', index=False)
