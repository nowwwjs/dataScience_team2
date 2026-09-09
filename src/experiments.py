import pandas as pd
import numpy as np
import time

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def add_brand_frequency(train, evaluation):
    """Fit manufacturer frequency on train only, then apply it to evaluation."""
    train = train.copy()
    evaluation = evaluation.copy()
    if "manufacturer" not in train.columns:
        return train, evaluation

    frequency = train["manufacturer"].value_counts(normalize=True)
    train["brand_frequency"] = train["manufacturer"].map(frequency).fillna(0.0)
    evaluation["brand_frequency"] = evaluation["manufacturer"].map(frequency).fillna(0.0)
    return train, evaluation


def _build_preprocessor(X_train, scaler_name, encoder_name):
    """Create a preprocessor fitted only after a CV train fold is selected."""
    numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    scaler_dict = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }

    if encoder_name == "onehot":
        return ColumnTransformer(
            transformers=[
                ("num", scaler_dict[scaler_name], numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ]
        )
    if encoder_name == "frequency":
        return ColumnTransformer(
            transformers=[("num", scaler_dict[scaler_name], numeric_features)],
            remainder="drop",
        )
    raise ValueError(f"Unsupported encoder: {encoder_name}")


def _frequency_encode(train, evaluation):
    """Fit category frequencies on the train fold only."""
    train = train.copy()
    evaluation = evaluation.copy()
    categorical_features = train.select_dtypes(include=["object", "category"]).columns
    for column in categorical_features:
        frequency = train[column].value_counts(normalize=True)
        train[column] = train[column].map(frequency).fillna(0.0)
        evaluation[column] = evaluation[column].map(frequency).fillna(0.0)
    return train, evaluation


def _make_model(model_name, params):
    if model_name == "linear":
        return LinearRegression()
    if model_name == "ridge":
        return Ridge(**params["ridge"])
    if model_name == "lasso":
        return Lasso(**params["lasso"])
    if model_name == "random_forest":
        return RandomForestRegressor(**params["random_forest"])
    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(**params["gradient_boosting"])
    raise ValueError(f"Unsupported model: {model_name}")

def run_used_car_regression_experiments(
    data,
    target="price",
    scalers=("standard", "minmax", "robust"),
    encoders=("onehot", "frequency"),
    models=("linear", "ridge", "lasso", "random_forest", "gradient_boosting", "voting"),
    model_params=None,  # 🌟 에러 원인 해결: 이 매개변수가 확실히 들어가 있어야 합니다!
    cv=5,
    random_state=42
):
    """
    유저가 외부에서 던진 하이퍼파라미터를 동적으로 반영하여 
    전처리 조합별 5-Fold 교차검증을 수행하는 자동화 함수입니다.
    """
    X = data.drop(columns=[target])
    y = data[target]
    
    # 기본 하이퍼파라미터 세팅
    default_params = {
        "ridge": {"alpha": 1.0},
        "lasso": {"alpha": 1.0, "max_iter": 1000},
        "random_forest": {"n_estimators": 50, "max_depth": 12, "n_jobs": -1, "random_state": random_state},
        "gradient_boosting": {"n_estimators": 50, "max_depth": 5, "learning_rate": 0.1, "random_state": random_state}
    }
    
    # 외부 커스텀 파라미터가 들어왔다면 병합
    if model_params is not None:
        for m in model_params:
            if m in default_params:
                default_params[m].update(model_params[m])
                
    results_list = []
    
    for s_name in scalers:
        if s_name not in {"standard", "minmax", "robust"}:
            continue
        
        for e_name in encoders:
            if e_name not in {"onehot", "frequency"}:
                continue
            for m_name in models:
                if m_name not in {"linear", "ridge", "lasso", "random_forest", "gradient_boosting", "voting"}:
                    continue
                
                kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
                cv_maes, cv_rmses, cv_r2s, cv_mapes = [], [], [], []
                start_time = time.time()
                
                for train_idx, val_idx in kf.split(X):
                    X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    X_train_raw, X_val_raw = add_brand_frequency(X_train_raw, X_val_raw)
                    if e_name == "frequency":
                        X_train_raw, X_val_raw = _frequency_encode(X_train_raw, X_val_raw)
                    preprocessor = _build_preprocessor(X_train_raw, s_name, e_name)
                    X_train = preprocessor.fit_transform(X_train_raw)
                    X_val = preprocessor.transform(X_val_raw)
                    
                    if m_name == "voting":
                        rf_sub = RandomForestRegressor(**default_params["random_forest"])
                        gb_sub = GradientBoostingRegressor(**default_params["gradient_boosting"])
                        rf_sub.fit(X_train, y_train)
                        gb_sub.fit(X_train, y_train)
                        preds = (rf_sub.predict(X_val) + gb_sub.predict(X_val)) / 2.0
                    else:
                        model = _make_model(m_name, default_params)
                        model.fit(X_train, y_train)
                        preds = model.predict(X_val)
                    
                    cv_maes.append(mean_absolute_error(y_val, preds))
                    cv_rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
                    cv_r2s.append(r2_score(y_val, preds))
                    cv_mapes.append(np.mean(np.abs((y_val - preds) / np.where(y_val == 0, 1, y_val))) * 100)
                
                results_list.append({
                    "Scaler": s_name, "Encoder": e_name, "Model": m_name,
                    "MAE": np.mean(cv_maes), "RMSE": np.mean(cv_rmses), "R2": np.mean(cv_r2s),
                    "MAPE(%)": np.mean(cv_mapes), "Runtime(s)": round(time.time() - start_time, 2)
                })
                
    return pd.DataFrame(results_list).sort_values(by="MAE", ascending=True).reset_index(drop=True)
