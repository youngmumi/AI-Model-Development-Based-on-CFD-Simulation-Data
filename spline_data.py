import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import os

# ----------------------------------------------------------
# 설정
# ----------------------------------------------------------
INPUT_PATH = "./original.csv"       # 원본 데이터 360개
OUTPUT_PATH = "./interpolated_36000.csv"

# ----------------------------------------------------------
# 1. CSV 불러오기
# ----------------------------------------------------------
df = pd.read_csv(INPUT_PATH)

# 필수 컬럼 확인
required_cols = ["aoa", "cl", "cd"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"CSV에 '{c}' 컬럼이 없습니다.")

# 정렬
df = df.sort_values(by="aoa").reset_index(drop=True)

alpha = df["aoa"].values
cl = df["cl"].values
cd = df["cd"].values

# ----------------------------------------------------------
# 2. 보간용 새로운 Alpha 생성 (3600개)
# ----------------------------------------------------------
alpha_new = np.linspace(alpha.min(), alpha.max(), 36000)

# ----------------------------------------------------------
# 3. 스플라인 보간 모델 생성
# ----------------------------------------------------------
cl_spline = CubicSpline(alpha, cl)
cd_spline = CubicSpline(alpha, cd)

# ----------------------------------------------------------
# 4. 보간 수행
# ----------------------------------------------------------
cl_new = cl_spline(alpha_new)
cd_new = cd_spline(alpha_new)

# ----------------------------------------------------------
# 5. 결과 저장
# ----------------------------------------------------------
df_new = pd.DataFrame({
    "Alpha": alpha_new,
    "Cl": cl_new,
    "Cd": cd_new
})

df_new.to_csv(OUTPUT_PATH, index=False)

print("보간 완료!")
print(f"저장 위치: {os.path.abspath(OUTPUT_PATH)}")
print(f"원본: {len(alpha)}개 → 보간 결과: {len(alpha_new)}개")
