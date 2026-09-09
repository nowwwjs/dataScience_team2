import pandas as pd
import numpy as np

def clean_and_prepare_data(df):
    """
    제안서(Proposal)에 명시된 규칙대로 20,000건의 중고차 데이터를 정제하고 
    행 단위로 계산 가능한 5가지 파생 변수를 생성합니다.
    제조사 빈도(brand_frequency)는 데이터 누수를 막기 위해 실험 단계에서 train 기준으로 생성합니다.
    """
    data = df.copy()
    
    # ----------------------------------------------------
    # [1] 불필요한 식별자 컬럼 삭제 (제안서 Section 5.1)
    # ----------------------------------------------------
    drop_cols = ['id', 'url', 'region_url', 'VIN', 'image_url', 'county']
    data = data.drop(columns=[col for col in drop_cols if col in data.columns], errors='ignore')
    
    # ----------------------------------------------------
    # [2] 비현실적인 이상치 제거 (제안서 Section 5.1)
    # ----------------------------------------------------
    data = data[(data['price'] > 0) & (data['price'] < 300000)]
    data = data[(data['year'] >= 1900) & (data['year'] <= 2026)]
    data = data[(data['odometer'] > 0) & (data['odometer'] < 500000)]
    
    # ----------------------------------------------------
    # [3] 결측치 처리 (제안서 Section 5.1)
    # ----------------------------------------------------
    numeric_cols = ['year', 'odometer', 'lat', 'long']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
            
    categorical_cols = [
        'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 
        'title_status', 'transmission', 'drive', 'size', 'type', 'paint_color', 'state', 'region'
    ]
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].fillna("Unknown")
            
    if 'description' in data.columns:
        data['description'] = data['description'].fillna("")
        
    # ----------------------------------------------------
    # [4] 6대 피처 엔지니어링 구현 (제안서 Section 5.2)
    # ----------------------------------------------------
    
    # ① car_age
    data['car_age'] = 2026 - data['year']
    data['car_age'] = data['car_age'].apply(lambda x: 1 if x <= 0 else x)
    
    # ② odometer_per_year
    data['odometer_per_year'] = data['odometer'] / data['car_age']
    
    # ③ is_high_mileage
    data['is_high_mileage'] = (data['odometer_per_year'] > 15000).astype(int)
    
    # ④ posting_month (🚨 에러 해결 및 경고 방지 수정 파트)
    if 'posting_date' in data.columns:
        # mixed 타임존 경고를 없애기 위해 utc=True를 주고, 에러 방지를 위해 확실하게 datetime형으로 강제 변환합니다.
        datetime_series = pd.to_datetime(data['posting_date'], errors='coerce', utc=True)
        # 이제 안전하게 .dt 접근자를 사용해 월을 추출합니다.
        data['posting_month'] = datetime_series.dt.month
        # 혹시 변환에 실패한 결측치는 중고차 데이터 수집기 대다수가 몰려있는 4월(기본값)로 채웁니다.
        data['posting_month'] = data['posting_month'].fillna(4).astype(int)
    else:
        data['posting_month'] = 4
        
    # ⑤ description_length
    if 'description' in data.columns:
        data['description_length'] = data['description'].str.len()
    else:
        data['description_length'] = 0
        
    # ⑥ brand_frequency
    # 제조사 빈도는 학습 데이터에서만 계산해야 한다. 교차 검증 또는 최종 평가 단계에서
    # train fold 기준으로 만들어 붙인다. 여기서 전체 데이터 기준으로 계산하면 validation/test
    # 분포가 학습 피처에 섞일 수 있기 때문이다.
    
    # 모델 학습에 불필요한 대용량 텍스트 및 날짜 스트링 컬럼 드롭
    data = data.drop(columns=['description', 'posting_date'], errors='ignore')
    
    print("✨ 기본 파생 변수 5개 생성 완료! (brand_frequency는 학습 단계에서 생성)")
    print(f"📊 정제 후 최종 데이터 크기: {data.shape[0]}행, {data.shape[1]}열")
    
    return data
