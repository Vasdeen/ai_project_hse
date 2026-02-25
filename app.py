"""
Сервис на Streamlit для детекции аномалий подшипников опор ГТД
с использованием IsolationForest + feature engineering.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# -----------------------------------------------------------------------------
# Конфигурация признаков (как в analysis.ipynb)
# -----------------------------------------------------------------------------

DIAG_COLUMNS = [
    "V1", "VoГГ", "V2",
    "Fтк4", "Fc2", "Fc3", "Fc4",
    "F1", "2F1", "3F1", "F2", "2F2", "3F2", "Fкпа", "Fцс",
    "Pm", "P615", "dPfgo", "dPf1", "dPf2",
    "Рс1", "Рс2",
    "T607", "T606", "T600", "T638", "Tк.з.", "Lm1",
]

REGIME_COLUMNS = ["N1", "N2", "N3", "Qtg", "P2", "T1", "T4ср"]

ALL_FEATURE_COLUMNS = DIAG_COLUMNS + REGIME_COLUMNS

# Колонки, которые должны быть в загружаемом файле (минимальный набор)
REQUIRED_COLUMNS = [
    "Дата и время",
    "N1", "N2", "N3", "Qtg", "P2", "T1", "T4ср",
    "V1", "VoГГ", "V2",
    "F1", "2F1", "3F1", "F2", "2F2", "3F2",
    "Fтк4", "Fc2", "Fc3", "Fc4",
    "Pm", "P615", "dPfgo", "dPf1", "dPf2",
    "Рс1", "Рс2",
    "T607", "T606", "T600", "T638", "Tк.з.", "Lm1",
]


def load_csv(file_or_path, is_upload=True) -> pd.DataFrame:
    """Загрузка CSV с ожидаемым форматом (sep=';', decimal=',')."""
    try:
        if is_upload:
            df = pd.read_csv(
                file_or_path,
                sep=";",
                decimal=",",
                encoding="utf-8",
                dayfirst=True,
            )
        else:
            df = pd.read_csv(
                file_or_path,
                sep=";",
                decimal=",",
                encoding="utf-8",
                parse_dates=["Дата и время"],
                dayfirst=True,
            )
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        return None

    # Приводим "Дата и время" к datetime, если ещё не
    if "Дата и время" not in df.columns:
        st.error('В файле отсутствует колонка "Дата и время".')
        return None

    if not pd.api.types.is_datetime64_any_dtype(df["Дата и время"]):
        df["Дата и время"] = pd.to_datetime(
            df["Дата и время"],
            dayfirst=True,
            errors="coerce",
        )

    return df


def make_feature_table(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Построение таблицы признаков для модели аномалий."""
    df = df.sort_values("Дата и время").reset_index(drop=True).copy()
    df = df.set_index("Дата и время")

    cols = [c for c in ALL_FEATURE_COLUMNS if c in df.columns]
    if len(cols) < 10:
        return None
    feat = df[cols].astype(float)

    vib_keys = [
        c
        for c in [
            "VoГГ", "V1", "V2",
            "F1", "2F1", "3F1", "F2", "2F2", "3F2",
            "Fтк4", "Fc2", "Fc3", "Fc4",
        ]
        if c in feat.columns
    ]

    roll = feat[vib_keys].rolling(window=window, min_periods=window // 3)
    roll_mean = roll.mean().add_suffix("_roll_mean")
    roll_std = roll.std().add_suffix("_roll_std")

    ratio_features = {}
    if "F1" in feat.columns and "2F1" in feat.columns:
        ratio_features["F2F1_over_F1"] = feat["2F1"] / (feat["F1"] + 1e-6)
    if "F1" in feat.columns and "3F1" in feat.columns:
        ratio_features["F3F1_over_F1"] = feat["3F1"] / (feat["F1"] + 1e-6)
    if "F2" in feat.columns and "2F2" in feat.columns:
        ratio_features["F2F2_over_F2"] = feat["2F2"] / (feat["F2"] + 1e-6)
    if "F2" in feat.columns and "3F2" in feat.columns:
        ratio_features["F3F2_over_F2"] = feat["3F2"] / (feat["F2"] + 1e-6)

    ratio_df = pd.DataFrame(ratio_features, index=feat.index)

    norm_features = {}
    if "N1" in feat.columns:
        for c in vib_keys:
            norm_features[f"{c}_per_N1"] = feat[c] / (feat["N1"] + 1e-3)
    if "N2" in feat.columns:
        for c in vib_keys:
            norm_features[f"{c}_per_N2"] = feat[c] / (feat["N2"] + 1e-3)
    norm_df = pd.DataFrame(norm_features, index=feat.index)

    full = pd.concat([feat, roll_mean, roll_std, ratio_df, norm_df], axis=1)
    full = full.dropna().reset_index()
    return full


def check_columns(df: pd.DataFrame):
    """Проверка наличия необходимых колонок."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return len(missing) == 0, missing


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Детекция аномалий подшипников ГТД",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 Детекция аномалий подшипников опор ГТД")
st.markdown(
    "Модель: **IsolationForest** на engineered-признаках (скользящие статистики, "
    "отношения гармоник, нормировка по оборотам)."
)

# -----------------------------------------------------------------------------
# Обучение модели на обучающей выборке
# -----------------------------------------------------------------------------

@st.cache_resource
def fit_model(train_path: str):
    """Обучение IsolationForest и StandardScaler на обучающей выборке."""
    df = load_csv(train_path, is_upload=False)
    if df is None or df.empty:
        return None, None, None, None, "Ошибка загрузки обучающей выборки."

    ok, missing = check_columns(df)
    if not ok:
        return None, None, None, None, f"Не хватает колонок: {missing}"

    train_feat = make_feature_table(df, window=60)
    if train_feat is None:
        return None, None, None, None, "Не удалось построить признаки (мало колонок)."

    feature_cols = [c for c in train_feat.columns if c != "Дата и время"]
    X_train = train_feat[feature_cols].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train_scaled)

    iso_scores_train = -iso.decision_function(X_train_scaled)
    thr = float(np.quantile(iso_scores_train, 0.995))

    return iso, scaler, feature_cols, thr, None


# Путь к обучающей выборке
TRAIN_PATH = os.path.join(os.path.dirname(__file__), "Обучающая выборка.csv")
train_exists = os.path.isfile(TRAIN_PATH)

iso, scaler, feature_cols, thr, err = None, None, None, None, None

if train_exists:
    iso, scaler, feature_cols, thr, err = fit_model(TRAIN_PATH)
else:
    with st.expander("Обучающая выборка не найдена. Загрузите свой файл.", expanded=True):
        train_upload = st.file_uploader(
            "Загрузите обучающую выборку (CSV, sep=';', decimal=',')",
            type=["csv"],
            key="train_upload",
        )
        if train_upload:
            df_train = load_csv(train_upload, is_upload=True)
            if df_train is not None:
                ok, missing = check_columns(df_train)
                if not ok:
                    st.error(f"Не хватает колонок: {missing}")
                else:
                    train_feat = make_feature_table(df_train, window=60)
                    if train_feat is None:
                        st.error("Не удалось построить признаки.")
                    else:
                        _cols = [c for c in train_feat.columns if c != "Дата и время"]
                        X_tr = train_feat[_cols].values
                        _scaler = StandardScaler()
                        X_tr_scaled = _scaler.fit_transform(X_tr)
                        _iso = IsolationForest(
                            n_estimators=200,
                            contamination=0.01,
                            random_state=42,
                            n_jobs=-1,
                        )
                        _iso.fit(X_tr_scaled)
                        _scores = -_iso.decision_function(X_tr_scaled)
                        _thr = float(np.quantile(_scores, 0.995))
                        iso, scaler, feature_cols, thr = _iso, _scaler, _cols, _thr
                        st.session_state["iso"] = _iso
                        st.session_state["scaler"] = _scaler
                        st.session_state["feature_cols"] = _cols
                        st.session_state["thr"] = _thr
                        st.success("Модель обучена на загруженной обучающей выборке.")
        if "iso" in st.session_state and iso is None:
            iso = st.session_state["iso"]
            scaler = st.session_state["scaler"]
            feature_cols = st.session_state["feature_cols"]
            thr = st.session_state["thr"]

if train_exists and err:
    st.warning(f"**Модель не обучена:** {err}")

# -----------------------------------------------------------------------------
# Загрузка датасета для анализа
# -----------------------------------------------------------------------------

st.subheader("Загрузка датасета для анализа")
uploaded_file = st.file_uploader(
    "Загрузите CSV с теми же признаками, что в обучающей выборке "
    "(разделитель ;, десятичная запятая)",
    type=["csv"],
)

if uploaded_file and iso is not None and scaler is not None:
    df = load_csv(uploaded_file, is_upload=True)
    if df is not None:
        ok, missing = check_columns(df)
        if not ok:
            st.error(f"В загруженном файле отсутствуют колонки: {missing}")
        else:
            feat = make_feature_table(df, window=60)
            if feat is None:
                st.error("Не удалось построить признаки. Проверьте данные.")
            else:
                X = feat[feature_cols].reindex(columns=feature_cols, fill_value=0.0).values
                X_scaled = scaler.transform(X)
                scores = -iso.decision_function(X_scaled)

                feat["iso_score"] = scores
                feat["аномалия"] = scores > thr

                n_total = len(feat)
                n_anom = int(feat["аномалия"].sum())
                frac_anom = n_anom / n_total if n_total else 0

                st.success(f"Обработано {n_total} строк. Аномалий: {n_anom} ({frac_anom:.2%})")
                st.metric("Порог (99.5% train)", f"{thr:.4f}")
                st.metric("Доля аномалий", f"{frac_anom:.2%}")

                tab1, tab2, tab3 = st.tabs(["График аномальности", "Топ аномалий", "Исходные данные"])

                with tab1:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(feat["Дата и время"], feat["iso_score"], label="iso_score")
                    ax.axhline(thr, color="r", linestyle="--", label=f"порог {thr:.4f}")
                    ax.set_title("Аномальный балл IsolationForest по времени")
                    ax.set_ylabel("iso_score")
                    ax.set_xlabel("Дата и время")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()

                with tab2:
                    top = feat.nlargest(100, "iso_score")[["Дата и время", "iso_score", "аномалия"]]
                    st.dataframe(top, use_container_width=True)

                with tab3:
                    st.dataframe(df.head(500), use_container_width=True)

elif uploaded_file and iso is None:
    st.error("Сначала необходимо обучить модель. Добавьте 'Обучающая выборка.csv' в папку приложения.")
