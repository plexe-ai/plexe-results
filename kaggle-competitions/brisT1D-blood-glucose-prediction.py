import polars as pl
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
import random
warnings.filterwarnings('ignore')

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

seed_everything(seed=2024)

def load_data():
    train = pl.read_csv("/kaggle/input/brist1d/train.csv").to_pandas()
    test = pl.read_csv("/kaggle/input/brist1d/test.csv").to_pandas()
    return train, test

def get_more_data(df, mode='train', ts=None, feats=None):
    full_data = []
    if mode == 'train':
        for start in range(0, len(ts)-60):
            end = start + 48
            target = end + 12
            hours_data = df[['p_num'] + 
                          [f+ts[i] for i in range(start,end) for f in feats] + 
                          ['bg'+ts[target]]]
            
            cols = ['p_num'] + [f+str(i-start) for i in range(start,end) for f in feats] + ['bg+1:00']
            hours_data.columns = cols
            hours_data = hours_data[~hours_data['bg+1:00'].isna()]
            full_data.append(hours_data)
    else:  # test mode
        # For test data, only take the last window that leads to the prediction
        start = len(ts) - 48
        hours_data = df[['p_num'] + [f+ts[i] for i in range(start, len(ts)) for f in feats]]
        hours_data.columns = ['p_num'] + [f+str(i-start) for i in range(start, len(ts)) for f in feats]
        full_data.append(hours_data)
    
    return pd.concat(full_data).drop_duplicates()

def enhanced_FE(df, mode='train'):
    
    # Extract p_num from id first
    df['p_num'] = df['id'].str.split('_').str[0]
    
    # Base features and timestamps
    feats = ['bg', 'insulin', 'hr', 'steps', 'cals']
    ts = ['-5:55', '-5:50', '-5:45', '-5:40', '-5:35', '-5:30', '-5:25', '-5:20', '-5:15', 
          '-5:10', '-5:05', '-5:00', '-4:55', '-4:50', '-4:45', '-4:40', '-4:35', '-4:30', 
          '-4:25', '-4:20', '-4:15', '-4:10', '-4:05', '-4:00', '-3:55', '-3:50', '-3:45', 
          '-3:40', '-3:35', '-3:30', '-3:25', '-3:20', '-3:15', '-3:10', '-3:05', '-3:00', 
          '-2:55', '-2:50', '-2:45', '-2:40', '-2:35', '-2:30', '-2:25', '-2:20', '-2:15', 
          '-2:10', '-2:05', '-2:00', '-1:55', '-1:50', '-1:45', '-1:40', '-1:35', '-1:30', 
          '-1:25', '-1:20', '-1:15', '-1:10', '-1:05', '-1:00', '-0:55', '-0:50', '-0:45', 
          '-0:40', '-0:35', '-0:30', '-0:25', '-0:20', '-0:15', '-0:10', '-0:05', '-0:00']
    
    # Convert 'None' to np.nan
    for col in df.columns:
        if df[col].dtype == object and col != 'p_num' and col != 'id':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Step transform
    step_cols = [col for col in df.columns if 'step' in col]
    for c in step_cols:
        df[c] = df[c].fillna(50).astype(np.float32)
        df[c] = np.log1p(df[c])
    
    # Drop high NaN columns
    nan_cols = [col for col in df.columns if df[col].isna().mean() > 0.95]
    df.drop(['id', 'time'] + nan_cols, axis=1, inplace=True)
    
    # Fill BG values
    bg_cols = [col for col in df.columns if (col != 'bg+1:00') and ('bg' in col)]
    df[bg_cols] = df[bg_cols].fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
    
    df = get_more_data(df, mode, ts, feats)
    df['p_num'] = df['p_num'].apply(lambda x: int(x[1:])).astype(np.int8)
    
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    
    # 1. Time-based glucose variability
    for window in [3, 6, 12]:
        recent_bg = [f'bg{i}' for i in range(max(0, 48-window), 48)]
        if all(col in df.columns for col in recent_bg):
            df[f'recent_bg_mean_{window}'] = df[recent_bg].astype(float).mean(axis=1)
            df[f'recent_bg_std_{window}'] = df[recent_bg].astype(float).std(axis=1)
            df[f'recent_bg_max_{window}'] = df[recent_bg].astype(float).max(axis=1)
            df[f'recent_bg_min_{window}'] = df[recent_bg].astype(float).min(axis=1)
            df[f'recent_bg_range_{window}'] = df[f'recent_bg_max_{window}'] - df[f'recent_bg_min_{window}']
    
    # 2. Advanced glucose rate of change
    for i in range(42, 48):
        if f'bg{i}' in df.columns:
            # First order derivative (velocity)
            if i > 42:  # Changed from i > 0
                df[f'bg_velocity_{i}'] = df[f'bg{i}'].astype(float) - df[f'bg{i-1}'].astype(float)
            # Second order derivative (acceleration)
            if i > 43:  # Changed from i > 1
                df[f'bg_acceleration_{i}'] = (df[f'bg{i}'].astype(float) - 2*df[f'bg{i-1}'].astype(float) + df[f'bg{i-2}'].astype(float))
            # Percentage change
            if i > 42:  # Changed from i > 0
                df[f'bg_pct_change_{i}'] = (df[f'bg{i}'].astype(float) - df[f'bg{i-1}'].astype(float)) / (df[f'bg{i-1}'].astype(float) + 1e-6)
    
    # 3. Insulin impact features
    for i in range(42, 48):
        if all(f'{feat}{i}' in df.columns for feat in ['insulin', 'bg']):
            df[f'insulin_bg_interaction_{i}'] = df[f'insulin{i}'].astype(float) * df[f'bg{i}'].astype(float)
            if i > 42:  # Changed from i > 0
                df[f'insulin_sensitivity_{i}'] = (df[f'bg{i}'].astype(float) - df[f'bg{i-1}'].astype(float)) / (df[f'insulin{i-1}'].astype(float) + 1e-6)
            # Calculate insulin sum for last 6 time points
            lookback = min(6, i-41)  # Ensure we don't look back before index 42
            df[f'insulin_sum_{i}'] = df[[f'insulin{j}' for j in range(i-lookback+1, i+1)]].astype(float).sum(axis=1)
    
    # 4. Activity impact features
    for i in range(42, 48):
        if all(f'{feat}{i}' in df.columns for feat in ['steps', 'hr']):
            df[f'activity_intensity_{i}'] = df[f'steps{i}'].astype(float) * df[f'hr{i}'].astype(float)
            if i > 42:  # Changed from i > 0
                df[f'activity_change_{i}'] = df[f'activity_intensity_{i}'] - df[f'activity_intensity_{i-1}']
                
    # 5. Enhanced patient-specific features
    recent_cols = [col for col in df.columns if any(f'{feat}47' in col for feat in feats)]
    for col in recent_cols:
        df[f'{col}_patient_mean'] = df.groupby('p_num')[col].transform('mean')
        df[f'{col}_patient_std'] = df.groupby('p_num')[col].transform('std')
        df[f'{col}_patient_max'] = df.groupby('p_num')[col].transform('max')
        df[f'{col}_patient_min'] = df.groupby('p_num')[col].transform('min')
        df[f'{col}_patient_range'] = df[f'{col}_patient_max'] - df[f'{col}_patient_min']
    
    return df.astype(np.float32)

class EnhancedEnsemble:
    def __init__(self, num_folds=7):
        self.num_folds = num_folds
        self.models = self._create_models()
        self.kf = KFold(n_splits=num_folds, shuffle=True, random_state=2024)
        
    def _create_models(self):
        return [
            LGBMRegressor(
                n_estimators=2000,
                learning_rate=0.005,
                num_leaves=63,
                feature_fraction=0.8,
                subsample=0.8,
                reg_alpha=0.3,
                reg_lambda=0.3,
                min_child_samples=20,
                random_state=2024,
                n_jobs=-1,
                force_col_wise=True
            ),
            XGBRegressor(
                n_estimators=2000,
                learning_rate=0.005,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.3,
                reg_lambda=0.3,
                min_child_weight=3,
                random_state=2024,
                n_jobs=-1,
                tree_method='hist'
            ),
            CatBoostRegressor(
                iterations=2000,
                learning_rate=0.005,
                depth=8,
                l2_leaf_reg=5,
                random_seed=2024,
                verbose=False,
                thread_count=-1
            )
        ]
    
    def train_and_predict(self, X_train, y_train, X_test):
        oof_predictions = np.zeros((X_train.shape[0], len(self.models)))
        test_predictions = np.zeros((X_test.shape[0], len(self.models)))
        
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            for i, model in enumerate(self.models):
                model.fit(X_tr, y_tr)
                fold_pred = model.predict(X_val)
                oof_predictions[val_idx, i] = fold_pred
                test_predictions[:, i] += model.predict(X_test) / self.num_folds
                
                fold_rmse = np.sqrt(np.mean((y_val - fold_pred) ** 2))
        
        return oof_predictions, test_predictions

def main():
    train, test = load_data()
    train = enhanced_FE(train, mode='train')
    test = enhanced_FE(test, mode='test')
    
    target_col = 'bg+1:00'
    important_features = [col for col in train.columns 
                         if col not in [target_col, 'p_num']
                         and (('bg' in col and '4' in col) or
                             ('insulin' in col and '4' in col) or
                             'velocity' in col or
                             'acceleration' in col or
                             'pct_change' in col or
                             'interaction' in col or
                             'sensitivity' in col or
                             'activity' in col or
                             'intensity' in col or
                             'patient_' in col)]
    
    X_train = train[important_features]
    y_train = train[target_col]
    X_test = test[important_features]
    
    del train, test
    import gc
    gc.collect()
    
    model = EnhancedEnsemble()
    oof_preds, test_preds = model.train_and_predict(X_train, y_train, X_test)
    
    # Using weights that favor XGBoost based on validation performance
    final_preds = np.average(test_preds, axis=1, weights=[0.3, 0.4, 0.3])
    
    submission = pd.read_csv('/kaggle/input/brist1d/sample_submission.csv')
    submission['bg+1:00'] = final_preds
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
