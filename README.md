# AI Model Development Based on CFD Simulation Data

본 연구는 단일 에어포일 형상(Eppler 423)의 공력 특성 분석과 공력 계수 예측 모델 개발을 목적으로 수행되었다.
CFD 해석을 통해 받음각(AoA)에 따른 공력 데이터를 구축하고, k-w GEKO 난류 모델과 격자 및 도메인 조건
을 적용하여 유동장을 해석하였다. 격자 민감도 분석을 통해 공력 계수의 수치적 신뢰성을 확보하였으며, 풍동 실
험을 통해 CFD 해석 결과의 유동 구조를 정성적으로 비교하였다. 확보된 데이터를 기반으로 Cl에는 MLP 회귀 모
델을, Cd에는 XGBoost 회귀 모델을 적용하였다. 그 결과 전체 받음각 범위에서 공력 계수 예측이 안정적으로 수
행되었으며, 제안한 모델이 공력 설계 초기 단계에서 계산 효율 향상에 활용 가능함을 확인하였다.

This study analyzes the aerodynamic characteristics of a single airfoil (Eppler 423) and develops an
aerodynamic coefficient prediction model. Aerodynamic data were obtained through CFD simulations
using the k-w GEKO turbulence model, with numerical reliability ensured via mesh sensitivity analysis
and qualitative validation through wind tunnel experiments. An MLP regression model was applied for Cl
prediction and an XGBoost model for Cd prediction. The results confirm stable prediction performance
across the full angle-of-attack range, demonstrating the model’s potential to enhance computational
efficiency in early-stage aerodynamic design.
