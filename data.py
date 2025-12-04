import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ==========================================================
# 경로 설정 (필요하면 여기만 수정)
# ==========================================================
TRAIN_CSV_PATH = "./3600/train_70.csv"
VAL_CSV_PATH   = "./3600/val_30.csv"
OUTDIR         = "./mini_mlp_3600_output"
os.makedirs(OUTDIR, exist_ok=True)


# ----------------------------------------------------------
# 1. 데이터 로드
# ----------------------------------------------------------
def load_data(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)

    required = ["Alpha", "Cl", "Cd"]
    for c in required:
        if c not in train_df.columns:
            raise ValueError(f"train CSV에 '{c}' 컬럼이 없습니다.")
        if c not in val_df.columns:
            raise ValueError(f"val CSV에 '{c}' 컬럼이 없습니다.")

    return train_df, val_df


# ----------------------------------------------------------
# 2. 특성/타깃 분리 + 스케일러
# ----------------------------------------------------------
def prepare_data(train_df, val_df):
    X_train = train_df[["Alpha"]].values
    X_val   = val_df[["Alpha"]].values

    y_cl_train = train_df["Cl"].values
    y_cd_train = train_df["Cd"].values
    y_cl_val   = val_df["Cl"].values
    y_cd_val   = val_df["Cd"].values

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_val_scaled   = x_scaler.transform(X_val)

    return X_train_scaled, X_val_scaled, y_cl_train, y_cd_train, y_cl_val, y_cd_val, x_scaler


# ----------------------------------------------------------
# 3. 모델 학습 (Cl: MLP, Cd: XGB)
# ----------------------------------------------------------
def train_models(X_train, y_cl_train, y_cd_train):
    cl_mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42,
    )
    cl_mlp.fit(X_train, y_cl_train)

    cd_xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    cd_xgb.fit(X_train, y_cd_train)

    return cl_mlp, cd_xgb


# ----------------------------------------------------------
# 4. 성능 지표
# ----------------------------------------------------------
def eval_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return {"MSE": float(mse), "MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


# ----------------------------------------------------------
# 5-1. aoa vs cd/cl (true vs pred, line)
# ----------------------------------------------------------
def plot_aoa_vs_target(alpha, y_true, y_pred, ylabel, filename, title):
    # alpha 기준 정렬해서 부드러운 곡선
    idx = np.argsort(alpha)
    a   = alpha[idx]
    t   = y_true[idx]
    p   = y_pred[idx]

    plt.figure(figsize=(12, 4))
    plt.plot(a, t, label=f"true {ylabel}", linewidth=2)
    plt.plot(a, p, "--", label=f"pred {ylabel}", linewidth=2)

    plt.xlabel("aoa [deg]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    save_path = os.path.join(OUTDIR, filename)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("✔ Saved:", save_path)


# ----------------------------------------------------------
# 5-2. Actual vs Predicted scatter (대각선 포함)
# ----------------------------------------------------------
def plot_actual_vs_pred(y_true, y_pred, ylabel, filename, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)

    # y=x 기준선
    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_v, max_v], [min_v, max_v], "r--")

    plt.xlabel(f"Actual {ylabel}")
    plt.ylabel(f"Predicted {ylabel}")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)

    save_path = os.path.join(OUTDIR, filename)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("✔ Saved:", save_path)


# ----------------------------------------------------------
# 5-3. Index vs True/Pred scatter
# ----------------------------------------------------------
def plot_index_vs_true_pred(y_true, y_pred, ylabel, filename, title):
    idx = np.arange(len(y_true))

    plt.figure(figsize=(7, 5))
    plt.scatter(idx, y_true, s=20, label="True", alpha=0.8)
    plt.scatter(idx, y_pred, s=20, label="Pred", alpha=0.8)

    plt.xlabel("Sample index (val set)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    save_path = os.path.join(OUTDIR, filename)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("✔ Saved:", save_path)


# ----------------------------------------------------------
# 6. 메인
# ----------------------------------------------------------
def main():
    # 1) 데이터 로드
    train_df, val_df = load_data(TRAIN_CSV_PATH, VAL_CSV_PATH)

    # 2) 전처리
    (X_train, X_val,
     y_cl_train, y_cd_train,
     y_cl_val,   y_cd_val,
     x_scaler) = prepare_data(train_df, val_df)

    # 3) 모델 학습
    cl_mlp, cd_xgb = train_models(X_train, y_cl_train, y_cd_train)

    # 4) 검증셋 예측
    cl_pred_val = cl_mlp.predict(X_val)
    cd_pred_val = cd_xgb.predict(X_val)
    alpha_val   = val_df["Alpha"].values

    # 5) 성능 지표
    cl_metrics = eval_metrics(y_cl_val, cl_pred_val)
    cd_metrics = eval_metrics(y_cd_val, cd_pred_val)

    metrics_all = {
        "info": {
            "train_samples": int(len(train_df)),
            "val_samples":   int(len(val_df)),
            "features":      ["Alpha"],
            "models": {"Cl": "MLPRegressor", "Cd": "XGBRegressor"},
        },
        "Cl": cl_metrics,
        "Cd": cd_metrics,
    }

    # 6) 그래프 생성 6개 (요청 스타일)
    # --- aoa vs target (true vs pred, line) ---
    plot_aoa_vs_target(
        alpha_val, y_cd_val, cd_pred_val,
        ylabel="cd",
        filename="val_cd_aoa_true_vs_pred.png",
        title="Validation: aoa vs cd (true vs pred)",
    )

    plot_aoa_vs_target(
        alpha_val, y_cl_val, cl_pred_val,
        ylabel="cl",
        filename="val_cl_aoa_true_vs_pred.png",
        title="Validation: aoa vs cl (true vs pred)",
    )

    # --- Actual vs Predicted scatter ---
    plot_actual_vs_pred(
        y_cd_val, cd_pred_val,
        ylabel="Cd",
        filename="val_cd_actual_vs_pred.png",
        title="Cd: Actual vs Predicted",
    )

    plot_actual_vs_pred(
        y_cl_val, cl_pred_val,
        ylabel="Cl",
        filename="val_cl_actual_vs_pred.png",
        title="Cl: Actual vs Predicted",
    )

    # --- Index vs True/Pred scatter ---
    plot_index_vs_true_pred(
        y_cd_val, cd_pred_val,
        ylabel="Cd",
        filename="val_cd_index_true_vs_pred.png",
        title="Cd: True vs Pred (scatter by index)",
    )

    plot_index_vs_true_pred(
        y_cl_val, cl_pred_val,
        ylabel="Cl",
        filename="val_cl_index_true_vs_pred.png",
        title="Cl: True vs Pred (scatter by index)",
    )

    # 7) 모델 & 스케일러 저장
    joblib.dump(x_scaler, os.path.join(OUTDIR, "x_scaler.joblib"))
    joblib.dump(cl_mlp,  os.path.join(OUTDIR, "cl_mlp_model.joblib"))
    joblib.dump(cd_xgb,  os.path.join(OUTDIR, "cd_xgb_model.joblib"))

    # 8) 지표 JSON 저장
    with open(os.path.join(OUTDIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=4)

    # 9) 콘솔 출력
    print("\n==== Validation metrics ====")
    print("[Cl]")
    for k, v in cl_metrics.items():
        print(f"  {k}: {v:.6g}")
    print("\n[Cd]")
    for k, v in cd_metrics.items():
        print(f"  {k}: {v:.6g}")
    print(f"\n모든 결과 저장 폴더: {OUTDIR}")


if __name__ == "__main__":
    main()
