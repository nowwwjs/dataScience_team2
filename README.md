# UsedCar: 중고차 가격 예측

Craigslist Vehicles Dataset 기반 20,000건 고정 시드 샘플로 차량 판매 가격을 예측하는 머신러닝 파이프라인입니다. 전처리와 모델을 임의로 하나 고르지 않고, 교차검증 리더보드에서 가장 낮은 평균 MAE 조합을 선택한 뒤 격리된 hold-out test로 한 번 더 평가합니다.

## 검증 설계

```text
Raw data
  -> 정제와 파생 피처
  -> Train/Validation 80%와 Hold-out Test 20% 분리
  -> Train/Validation 내부 5-Fold CV
  -> 최적 조합 선택
  -> Hold-out Test 최종 평가
```

- Scaler: StandardScaler, RobustScaler
- Encoder: One-Hot Encoding, Frequency Encoding
- Model: Random Forest, Gradient Boosting, 평균 앙상블
- 총 12개 조합을 동일한 5-Fold로 비교

`brand_frequency`, 범주 빈도, Scaler, Encoder는 매 Fold의 train 데이터에서만 `fit`하고 validation에는 `transform`만 적용합니다. 최종 test 평가에서도 Train/Validation 데이터만으로 변환기를 학습합니다.

## 피처

`car_age`, `odometer_per_year`, `is_high_mileage`, `posting_month`, `description_length`, `brand_frequency`를 추가했습니다. 이 중 `brand_frequency`는 전체 데이터 분포를 미리 보지 않도록 학습 데이터에서만 계산합니다.

## 이전 실행 기록

이전 실행에서는 Robust Scaler, One-Hot Encoder, Random Forest 조합이 5-Fold 평균 MAE **$4,704.28**로 선택됐고, hold-out test MAE는 **$4,755.79**, RMSE는 **$9,374.53**, R²는 **0.6673**이었습니다.

현재 코드는 Fold 내부에서 변환기를 `fit`하도록 보완했습니다. 따라서 위 수치는 참고용 기록이며, 최종 제출용 수치는 동일 데이터를 다시 실행해 갱신해야 합니다.

## 실행

1. [Craigslist Vehicles Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data/data)의 `vehicles.csv`를 `data/vehicles.csv`에 둡니다.
2. `python -m pip install -r requirements.txt`
3. `python main.py`

원본 데이터와 실행 중 생성되는 샘플·전처리 파일은 저장소에 포함하지 않습니다.
