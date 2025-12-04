import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ----------------------------------------------------------
# 설정
# ----------------------------------------------------------
INPUT_PATH = "original.csv"   # 보간된 데이터
TRAIN_OUT = "./360/train_original_70.csv"
VAL_OUT   = "./360/val_original_30.csv"

# ----------------------------------------------------------
# 1. CSV 불러오기
# ----------------------------------------------------------
df = pd.read_csv(INPUT_PATH)

# Alpha, Cl, Cd 컬럼 체크
required_cols = ["aoa", "cl", "cd"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"CSV에 '{c}' 컬럼이 없습니다.")

# ----------------------------------------------------------
# 2. 70% / 30% 분리
# ----------------------------------------------------------
train_df, val_df = train_test_split(
    df,
    test_size=0.30,     # 30% 검증
    shuffle=True,       # 셔플하여 랜덤 분할
    random_state=42     # 재현성 확보
)

# ----------------------------------------------------------
# 3. 파일 저장
# ----------------------------------------------------------
train_df.to_csv(TRAIN_OUT, index=False)
val_df.to_csv(VAL_OUT, index=False)

print(f"전체 데이터 개수: {len(df)}")
print(f"Train 70%: {len(train_df)} → {os.path.abspath(TRAIN_OUT)}")
print(f"Validation 30%: {len(val_df)} → {os.path.abspath(VAL_OUT)}")
