import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# ===================== Configuration =====================
CONFIG = {
    'BASE_PATH': './data/',                       # Update if your data directory differs
    'CV_FOLDS': 3,
    'RANDOM_STATE': 42,
    'ENSEMBLE_WEIGHTS': {'lgb': 0.75, 'xgb': 0.25},
    'USE_CLASS_BALANCING': True,
    'USE_FEATURE_ENGINEERING': False,
    'HYPERPARAMETER_TUNING': False,
}

# ===================== Data Loading =====================
def load_data(config):
    paths = {
        "train": config['BASE_PATH'] + "train_data.parquet",
        "test": config['BASE_PATH'] + "test_data.parquet",
        "data_dict": config['BASE_PATH'] + "data_dictionary.csv"
    }
    print("Loading data files...")
    data = {
        'train': pd.read_parquet(paths['train']),
        'test': pd.read_parquet(paths['test']),
        'data_dict': pd.read_csv(paths['data_dict'])
    }
    print(f"Train shape: {data['train'].shape}, Test shape: {data['test'].shape}")
    return data

# ===================== Feature Engineering =====================
def get_feature_types(data_dict):
    data_dict.columns = [c.strip() for c in data_dict.columns]
    num_features = data_dict.loc[
        data_dict['Type'].str.contains('Numerical', case=False, na=False),
        'masked_column'
    ].tolist()
    cat_features = data_dict.loc[
        data_dict['Type'].str.contains('Categorical|One hot encoded', case=False, na=False),
        'masked_column'
    ].tolist()
    drop_cols = ['y', 'id1', 'id2', 'id3', 'id4', 'id5']
    num_features = [c for c in num_features if c not in drop_cols]
    cat_features = [c for c in cat_features if c not in drop_cols]
    return num_features, cat_features

def apply_class_balancing(X_train, y_train):
    pos_idx = np.where(y_train == 1)
    neg_idx = np.where(y_train == 0)
    if len(neg_idx) > len(pos_idx):
        neg_sample_size = min(len(pos_idx) * 2, len(neg_idx))
        neg_sample_idx = np.random.choice(neg_idx, neg_sample_size, replace=False)
        balanced_idx = np.concatenate([pos_idx, neg_sample_idx])
        np.random.shuffle(balanced_idx)
        return X_train[balanced_idx], y_train[balanced_idx]
    return X_train, y_train

# ===================== Preprocessing Pipeline =====================
def build_preprocessor(num_features, cat_features):
    from sklearn.preprocessing import FunctionTransformer
    def convert_to_str(X):
        return X.astype(str)
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('str_converter', FunctionTransformer(convert_to_str)),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False, max_categories=20))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_features),
        ('cat', cat_pipe, cat_features)
    ], remainder='drop')
    return preprocessor

# ===================== Evaluation Metrics =====================
def apk(actual, predicted, k=7):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(predicted), k)

def mapk(actual, predicted, k=7):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def evaluate_mapk(y_true, y_pred, groups, k=7):
    df = pd.DataFrame({'id2': groups, 'pred': y_pred, 'y': y_true})
    actuals, preds = [], []
    for _, group in df.groupby('id2'):
        group_sorted = group.sort_values('pred', ascending=False)
        actuals.append(group_sorted['y'].iloc)
        preds.append(group_sorted['y'].iloc[:k].tolist())
    return mapk(actuals, preds, k)

# ===================== Model Training =====================
def train_ensemble_model(X, y, X_test, groups, config):
    print("Training ensemble model...")
    skf = GroupKFold(n_splits=config['CV_FOLDS'])
    test_preds = {model: np.zeros(len(X_test)) for model in ['lgb', 'xgb']}
    oof_preds = np.zeros(len(X))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y, groups)):
        print(f"\n=== Fold {fold + 1}/{config['CV_FOLDS']} ===")
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        if config['USE_CLASS_BALANCING'] and len(np.unique(y_train_fold)) > 1:
            X_train_fold, y_train_fold = apply_class_balancing(X_train_fold, y_train_fold)
            print(f"Applied class balancing: {len(y_train_fold)} samples")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=config['RANDOM_STATE'],
            class_weight='balanced',
            verbosity=-1,
            n_jobs=-1
        )
        lgb_model.fit(X_train_fold, y_train_fold)
        scale_pos_weight = (y_train_fold == 0).sum() / max((y_train_fold == 1).sum(), 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=config['RANDOM_STATE'],
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
            n_jobs=-1
        )
        xgb_model.fit(X_train_fold, y_train_fold)
        val_pred_lgb = lgb_model.predict_proba(X_val_fold)[:, 1]
        val_pred_xgb = xgb_model.predict_proba(X_val_fold)[:, 1]
        val_pred = (config['ENSEMBLE_WEIGHTS']['lgb'] * val_pred_lgb +
                    config['ENSEMBLE_WEIGHTS']['xgb'] * val_pred_xgb)
        oof_preds[val_idx] = val_pred
        test_preds['lgb'] += lgb_model.predict_proba(X_test)[:, 1] / config['CV_FOLDS']
        test_preds['xgb'] += xgb_model.predict_proba(X_test)[:, 1] / config['CV_FOLDS']
        fold_auc = roc_auc_score(y_val_fold, val_pred)
        print(f"Fold {fold + 1} AUC: {fold_auc:.4f}")
    final_test_preds = (config['ENSEMBLE_WEIGHTS']['lgb'] * test_preds['lgb'] +
                        config['ENSEMBLE_WEIGHTS']['xgb'] * test_preds['xgb'])
    overall_auc = roc_auc_score(y, oof_preds)
    overall_mapk = evaluate_mapk(y, oof_preds, groups)
    print(f"\n=== Overall Performance ===")
    print(f"OOF AUC: {overall_auc:.4f}")
    print(f"OOF MAP@7: {overall_mapk:.4f}")
    return final_test_preds

# ===================== Submission Generation =====================
def generate_submission(test_df, predictions, filename="submission1.csv"):
    submission = test_df[['id1', 'id2', 'id3', 'id5']].copy()
    submission['pred'] = predictions
    submission = submission.sort_values(['id2', 'pred'], ascending=[True, False])
    submission.to_csv(filename, index=False)
    print(f"Submission saved as {filename}")

# ===================== Main Pipeline =====================
def main():
    print("Starting AMEX Campus Challenge Solution...")
    data = load_data(CONFIG)
    train_df = data['train'].copy()
    test_df = data['test'].copy()
    num_features, cat_features = get_feature_types(data['data_dict'])
    common_cols = set(train_df.columns) & set(test_df.columns)
    num_features = [f for f in num_features if f in common_cols]
    cat_features = [f for f in cat_features if f in common_cols]
    print(f"Using {len(num_features)} numeric and {len(cat_features)} categorical features")
    train_df['id2'] = train_df['id2'].astype(str)
    test_df['id2'] = test_df['id2'].astype(str)
    y = train_df['y'].astype(int).values
    groups = train_df['id2'].values
    preprocessor = build_preprocessor(num_features, cat_features)
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    print(f"Preprocessed data shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    predictions = train_ensemble_model(X_train, y, X_test, groups, CONFIG)
    generate_submission(test_df, predictions)
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
