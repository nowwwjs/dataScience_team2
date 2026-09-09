import os
import pandas as pd
import numpy as np

# 사이킷런의 격리 분할 및 최종 평가 지표 임포트
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# src 패키지에서 가공 및 실험 함수 수입
from src import add_brand_frequency, clean_and_prepare_data, run_used_car_regression_experiments

def load_or_create_preprocessed_data():
    """
    이미 전처리되어 저장된 파일이 있으면 바로 로드하고,
    없으면 원본에서 샘플링 및 전처리를 수행하여 가공된 데이터를 반환합니다.
    """
    PROCESSED_DATA_PATH = "data/processed_vehicles.csv"
    RAW_DATA_PATH = "data/vehicles.csv"
    SAMPLED_DATA_PATH = "data/sampled_vehicles.csv"
    
    if os.path.exists(PROCESSED_DATA_PATH):
        print(f"📦 전처리 완료 데이터 파일이 이미 존재합니다: {PROCESSED_DATA_PATH}")
        print("🔄 가공된 최신 데이터를 즉시 로드합니다.")
        return pd.read_csv(PROCESSED_DATA_PATH)
    
    print("🔄 전처리 파일이 없어 새롭게 파이프라인을 구동합니다.")
    if os.path.exists(SAMPLED_DATA_PATH):
        raw_sample = pd.read_csv(SAMPLED_DATA_PATH)
    else:
        if not os.path.exists(RAW_DATA_PATH):
            raise FileNotFoundError(f"'{RAW_DATA_PATH}' 원본 파일이 필요합니다. 'data/' 폴더에 넣어주세요.")
        df = pd.read_csv(RAW_DATA_PATH)
        raw_sample = df.sample(n=20000, random_state=42)
        raw_sample.to_csv(SAMPLED_DATA_PATH, index=False)
        
    processed_data = clean_and_prepare_data(raw_sample)
    processed_data.to_csv(PROCESSED_DATA_PATH, index=False)
    return processed_data

if __name__ == "__main__":
    try:
        # [1단계] 완전히 정제된 데이터셋 준비 (18,215행)
        df_cleaned = load_or_create_preprocessed_data()
        
        print("\n==================================================================")
        print("🔒 [데이터 격리] 방법 A: Train/Val(80%) vs 최종 실전 Test(20%) 분할")
        print("==================================================================")
        # 데이터 누수를 완벽히 차단하기 위해, 전체 데이터의 20%를 기말고사용으로 숨겨둡니다.
        df_train_val, df_test = train_test_split(df_cleaned, test_size=0.2, random_state=42)
        
        print(f"📋 학습 및 교차검증용(Train/Val) 데이터 크기: {df_train_val.shape[0]}행")
        print(f"🔒 최종 실전 테스트용(Test) 데이터 크기    : {df_test.shape[0]}행")
        
        print("\n==================================================================")
        print("🎯 [교차 검증 단계] 80% 데이터 내에서 격자 루프 실험 시작")
        print("==================================================================")
        # 재서님이 고르고 튜닝한 하이퍼파라미터 조합 세팅
        
        # main.py 내부의 하이퍼파라미터 세팅 구역 변경
        custom_tuning_params = {
            # 나무를 120개로 대폭 늘리고 깊이를 18로 줘서 더 깐깐하게 마스터하게 만듭니다.
            "random_forest": {
                "n_estimators": 120, 
                "max_depth": 18
            },
            # 부스팅 릴레이 단계를 150번으로 늘리고, 보폭을 0.03으로 아주 미세하게 쪼갭니다.
            "gradient_boosting": {
                "n_estimators": 150, 
                "max_depth": 6, 
                "learning_rate": 0.03
            }
        }
        
        # 80% 데이터인 df_train_val만 주입하여 5-Fold 교차검증 리더보드를 뽑아냅니다.
        leaderboard = run_used_car_regression_experiments(
            data=df_train_val,
            target="price",
            scalers=("standard", "robust"),
            encoders=("onehot", "frequency"),
            models=("random_forest", "gradient_boosting", "voting"),
            model_params=custom_tuning_params,
            cv=5
        )
        
        print("\n🏆 [Validation 리더보드 결과] 🏆")
        print("==========================================================================================")
        print(leaderboard.to_string(index=False))
        print("==========================================================================================")
        
        # [3단계] 리더보드에서 평균 MAE 성능이 가장 뛰어난 1등 최적의 조합을 자동으로 도출합니다.
        best_row = leaderboard.iloc[0]
        best_scaler = best_row['Scaler']
        best_encoder = best_row['Encoder']
        best_model_name = best_row['Model']
        
        print(f"\n🥇 최고의 성적을 거둔 앙상블 조합 결정!")
        print(f"➡️ [Scaler: {best_scaler}] | [Encoder: {best_encoder}] | [Model: {best_model_name}] (Validation MAE: {best_row['MAE']:.2f})")
        
        print("\n==================================================================")
        print("🏁 [최종 실전 TEST 단계] 때 묻지 않은 20% 격리 데이터로 최종 평가 수행")
        print("==================================================================")
        print("🔄 1등 전처리 조합으로 최종 학습 및 실전 테스트지 예측 진행 중...")
        
        # 1등한 조합의 전처리 스펙대로 Train 데이터 전체를 피팅(fit)시킵니다.
        X_train = df_train_val.drop(columns=['price'])
        y_train = df_train_val['price']
        X_test = df_test.drop(columns=['price'])
        y_test = df_test['price']
        # 제조사 빈도는 train 데이터로만 학습해 test에 적용한다.
        X_train, X_test = add_brand_frequency(X_train, X_test)
        
        numeric_features = X_train.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 1등 스케일러 세팅
        scaler_obj = RobustScaler() if best_scaler == "robust" else StandardScaler()
        
        # 1등 인코더 세팅 및 데이터 변환
        if best_encoder == "onehot":
            preprocessor = ColumnTransformer(transformers=[
                ('num', scaler_obj, numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])
            X_train_tensor = preprocessor.fit_transform(X_train)
            X_test_tensor = preprocessor.transform(X_test)
            
        else:  # frequency인 경우
            for col in categorical_features:
                freq_map = X_train[col].value_counts(normalize=True)
                X_train[col] = X_train[col].map(freq_map)
                X_test[col] = X_test[col].map(freq_map).fillna(0)
            
            preprocessor = ColumnTransformer(transformers=[('num_all', scaler_obj, numeric_features + categorical_features)])
            X_train_tensor = preprocessor.fit_transform(X_train)
            X_test_tensor = preprocessor.transform(X_test)
            
        # 1등 모델 객체를 생성하여 '처음 보는' Test 데이터를 대상으로 최종 평가지표 도출
        rf_param = custom_tuning_params["random_forest"]
        gb_param = custom_tuning_params["gradient_boosting"]
        
        if best_model_name == "voting":
            model_rf = RandomForestRegressor(n_estimators=rf_param["n_estimators"], max_depth=rf_param["max_depth"], n_jobs=-1, random_state=42)
            model_gb = GradientBoostingRegressor(n_estimators=gb_param["n_estimators"], max_depth=gb_param["max_depth"], learning_rate=gb_param["learning_rate"], random_state=42)
            model_rf.fit(X_train_tensor, y_train)
            model_gb.fit(X_train_tensor, y_train)
            final_preds = (model_rf.predict(X_test_tensor) + model_gb.predict(X_test_tensor)) / 2.0
        elif best_model_name == "random_forest":
            model = RandomForestRegressor(n_estimators=rf_param["n_estimators"], max_depth=rf_param["max_depth"], n_jobs=-1, random_state=42)
            model.fit(X_train_tensor, y_train)
            final_preds = model.predict(X_test_tensor)
        else:
            model = GradientBoostingRegressor(n_estimators=gb_param["n_estimators"], max_depth=gb_param["max_depth"], learning_rate=gb_param["learning_rate"], random_state=42)
            model.fit(X_train_tensor, y_train)
            final_preds = model.predict(X_test_tensor)
            
        # 🏁 최종 모의고사 평가지표 계산
        test_mae = mean_absolute_error(y_test, final_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
        test_r2 = r2_score(y_test, final_preds)
        test_mape = np.mean(np.abs((y_test - final_preds) / np.where(y_test == 0, 1, y_test))) * 100
        
        print("\n📢 [최종 기말고사] 실전 Test 데이터셋 평가 지표 결과")
        print("==========================================================================================")
        print(f"📊 최종 실전 MAE (평균 절대 오차)      : {test_mae:.2f} 달러")
        print(f"📊 최종 실전 RMSE (오차 제곱근 평균)   : {test_rmse:.2f} 달러")
        print(f"📊 최종 실전 R2 Score (모델 설명력)    : {test_r2:.4f} (1에 가까울수록 완벽)")
        print(f"📊 최종 실전 MAPE (평균 백분율 오차율) : {test_mape:.2f} %")
        print("==========================================================================================")
        print("🚀 검증(Validation) 리더보드 탐색과 실전 격리 테스트(Test) 지표 산출까지 모두 완벽하게 완수되었습니다!")
        
    except Exception as e:
        print(f"❌ 프로세스 실행 중 에러 발생: {e}")
