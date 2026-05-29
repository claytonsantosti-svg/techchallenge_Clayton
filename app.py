"""
TECH CHALLENGE - FASE 4 | POSTECH FIAP - Data Analytics
Autor: Clayton Dias Santos
Aplicação Streamlit — Sistema Preditivo de Obesidade
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ObesityPredict — POSTECH FIAP",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado
st.markdown("""
<style>
    /* Paleta e tipografia */
    :root {
        --azul:    #185FA5;
        --verde:   #1D9E75;
        --laranja: #D85A30;
        --roxo:    #533AB7;
        --amarelo: #EF9F27;
        --cinza:   #5F5E5A;
    }

    /* ── SIDEBAR ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1e 0%, #0e1525 60%, #0a1020 100%);
        min-width: 280px !important;
        max-width: 280px !important;
        padding: 0 !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Radio buttons da sidebar — viram cards de navegação */
    [data-testid="stSidebar"] .stRadio > label {
        display: none;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        gap: 4px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
        border: 1px solid transparent !important;
    }
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
        background: rgba(55,138,221,0.15) !important;
        border-color: rgba(55,138,221,0.3) !important;
    }
    [data-testid="stSidebar"] .stRadio label[aria-checked="true"][data-baseweb="radio"] {
        background: rgba(55,138,221,0.2) !important;
        border-color: #378ADD !important;
    }
    [data-testid="stSidebar"] .stRadio span[data-testid="stMarkdownContainer"] p {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] div:first-child {
        display: none !important;
    }

    /* Separador da sidebar */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
        margin: 12px 0 !important;
    }

    .metric-card {
        background: #f7f8fa;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        border: 1px solid #e8eaed;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: #6b7280;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 11px;
        color: #9ca3af;
        margin-top: 2px;
    }

    .result-card {
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        border-left: 6px solid;
    }
    .result-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .result-sub {
        font-size: 14px;
        opacity: 0.8;
    }

    /* Badges de severidade */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .stButton > button {
        background: #185FA5;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-size: 15px;
        font-weight: 600;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: #0c447c;
        color: white;
    }

    h1, h2, h3 { font-weight: 700; }
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #111827;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 8px;
        margin: 1.5rem 0 1rem;
    }

    /* Ocultar menu padrão do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO E TREINAMENTO DO MODELO (cache para não reprocessar)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Treinando modelo de ML...")
def load_and_train():
    df = pd.read_csv("Obesity.csv")

    # Feature Engineering
    df["BMI"]    = df["Weight"] / (df["Height"] ** 2)
    df["FCVC_r"] = df["FCVC"].round().astype(int)
    df["NCP_r"]  = df["NCP"].round().astype(int)
    df["CH2O_r"] = df["CH2O"].round().astype(int)
    df["FAF_r"]  = df["FAF"].round().astype(int)
    df["TUE_r"]  = df["TUE"].round().astype(int)

    binary_cols = ["Gender", "family_history", "FAVC", "SMOKE", "SCC"]
    le = LabelEncoder()
    for col in binary_cols:
        df[col + "_enc"] = le.fit_transform(df[col])

    df["CAEC_enc"]   = df["CAEC"].map({"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3})
    df["CALC_enc"]   = df["CALC"].map({"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3})
    df["MTRANS_enc"] = df["MTRANS"].map({"Walking": 0, "Bike": 1, "Public_Transportation": 2, "Motorbike": 3, "Automobile": 4})

    obesity_order = [
        "Insufficient_Weight", "Normal_Weight",
        "Overweight_Level_I", "Overweight_Level_II",
        "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
    ]
    obesity_map = {v: i for i, v in enumerate(obesity_order)}
    df["target"] = df["Obesity"].map(obesity_map)

    FEATURES = [
        "Gender_enc", "Age", "Height", "Weight", "BMI",
        "family_history_enc", "FAVC_enc", "FCVC_r", "NCP_r",
        "CAEC_enc", "SMOKE_enc", "CH2O_r", "SCC_enc",
        "FAF_r", "TUE_r", "CALC_enc", "MTRANS_enc",
    ]

    X = df[FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=obesity_order, output_dict=True)

    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

    fi = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_})
    fi = fi.sort_values("Importance", ascending=False).reset_index(drop=True)

    return model, df, FEATURES, obesity_order, accuracy, cm, report, cv_scores, fi


# Rótulos em português
LABELS_PT = {
    "Insufficient_Weight": "Abaixo do Peso",
    "Normal_Weight":       "Peso Normal",
    "Overweight_Level_I":  "Sobrepeso Grau I",
    "Overweight_Level_II": "Sobrepeso Grau II",
    "Obesity_Type_I":      "Obesidade Grau I",
    "Obesity_Type_II":     "Obesidade Grau II",
    "Obesity_Type_III":    "Obesidade Grau III",
}

CORES_CLASSE = [
    "#185FA5", "#378ADD", "#FAC775", "#EF9F27",
    "#F0997B", "#D85A30", "#993C1D",
]

SEVERITY_CONFIG = {
    "Insufficient_Weight": {"color": "#185FA5", "bg": "#E6F1FB", "icon": "⬇️",  "risco": "Baixo peso"},
    "Normal_Weight":       {"color": "#1D9E75", "bg": "#E1F5EE", "icon": "✅",  "risco": "Saudável"},
    "Overweight_Level_I":  {"color": "#EF9F27", "bg": "#FAEEDA", "icon": "⚠️", "risco": "Atenção"},
    "Overweight_Level_II": {"color": "#D85A30", "bg": "#FAECE7", "icon": "⚠️", "risco": "Atenção elevada"},
    "Obesity_Type_I":      {"color": "#C0392B", "bg": "#FCEBEB", "icon": "🔴",  "risco": "Alto risco"},
    "Obesity_Type_II":     {"color": "#A93226", "bg": "#FCEBEB", "icon": "🔴",  "risco": "Risco muito alto"},
    "Obesity_Type_III":    {"color": "#7B241C", "bg": "#FCEBEB", "icon": "🔴",  "risco": "Risco crítico"},
}


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — NAVEGAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # Logo / cabeçalho
    st.markdown("""
    <div style='padding: 28px 20px 16px; border-bottom: 1px solid rgba(255,255,255,0.07);'>
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:6px;'>
            <span style='font-size:28px;'>🩺</span>
            <span style='font-size:20px; font-weight:700; color:#ffffff; letter-spacing:-0.3px;'>ObesityPredict</span>
        </div>
        <div style='font-size:12px; color:#6b7280; padding-left:38px;'>Clayton Dias Santos · POSTECH FIAP · Fase 4</div>
    </div>
    """, unsafe_allow_html=True)

    # Label seção navegação
    st.markdown("""
    <div style='padding: 16px 16px 4px; font-size:10px; font-weight:600;
                color:#4b5563; letter-spacing:0.1em; text-transform:uppercase;'>
        Navegação
    </div>
    """, unsafe_allow_html=True)

    pagina = st.radio(
        "Navegação",
        ["🏠  Home", "🔮  Preditor Clínico", "📊  Painel Analítico", "🤖  Métricas do Modelo"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # KPIs do modelo
    st.markdown("""
    <div style='padding: 4px 20px 12px;'>
        <div style='font-size:10px; font-weight:600; color:#4b5563;
                    letter-spacing:0.1em; text-transform:uppercase; margin-bottom:14px;'>
            Sobre o modelo
        </div>
        <div style='display:flex; flex-direction:column; gap:10px;'>
            <div style='background:rgba(29,158,117,0.12); border:1px solid rgba(29,158,117,0.25);
                        border-radius:10px; padding:10px 14px;'>
                <div style='font-size:10px; color:#6b7280; margin-bottom:2px;'>Acurácia</div>
                <div style='font-size:22px; font-weight:700; color:#1D9E75;'>98,35%</div>
                <div style='font-size:10px; color:#4b5563;'>CV 5-fold: 97,8% ± 0,3%</div>
            </div>
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px;'>
                <div style='background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                            border-radius:10px; padding:10px 12px;'>
                    <div style='font-size:10px; color:#6b7280;'>Pacientes</div>
                    <div style='font-size:18px; font-weight:700; color:#e0e0e0;'>2.111</div>
                </div>
                <div style='background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                            border-radius:10px; padding:10px 12px;'>
                    <div style='font-size:10px; color:#6b7280;'>Classes</div>
                    <div style='font-size:18px; font-weight:700; color:#e0e0e0;'>7</div>
                </div>
                <div style='background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                            border-radius:10px; padding:10px 12px;'>
                    <div style='font-size:10px; color:#6b7280;'>Features</div>
                    <div style='font-size:18px; font-weight:700; color:#e0e0e0;'>17</div>
                </div>
                <div style='background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                            border-radius:10px; padding:10px 12px;'>
                    <div style='font-size:10px; color:#6b7280;'>Árvores</div>
                    <div style='font-size:18px; font-weight:700; color:#e0e0e0;'>200</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Rodapé
    st.markdown("""
    <div style='padding: 8px 20px 20px; font-size:11px; color:#374151; line-height:1.8;'>
        <div style='margin-bottom:4px;'>🌲 Random Forest Classifier</div>
        <div style='margin-bottom:4px;'>📁 Dataset: Obesity.csv</div>
        <div>⚕️ Uso clínico assistido</div>
    </div>
    """, unsafe_allow_html=True)

# Carregar modelo
model, df, FEATURES, obesity_order, accuracy, cm, report, cv_scores, fi = load_and_train()


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA: HOME
# ─────────────────────────────────────────────────────────────────────────────
if pagina == "🏠  Home":
    st.markdown("# 🩺 Sistema Preditivo de Obesidade")
    st.markdown(
        "Ferramenta de apoio à decisão clínica desenvolvida para o **Tech Challenge Fase 4 — POSTECH FIAP**. "
        "O modelo classifica pacientes em 7 níveis de peso com base em variáveis comportamentais, físicas e genéticas."
    )

    st.markdown("---")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class='metric-card'><div class='metric-label'>Acurácia do modelo</div>
        <div class='metric-value' style='color:#1D9E75;'>98,35%</div>
        <div class='metric-sub'>Random Forest · 200 árvores</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='metric-card'><div class='metric-label'>Pacientes no dataset</div>
        <div class='metric-value'>2.111</div>
        <div class='metric-sub'>Sem valores nulos</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='metric-card'><div class='metric-label'>Com algum grau de obesidade</div>
        <div class='metric-value' style='color:#D85A30;'>46,0%</div>
        <div class='metric-sub'>972 pacientes</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class='metric-card'><div class='metric-label'>IMC médio geral</div>
        <div class='metric-value'>29,7</div>
        <div class='metric-sub'>Limiar de sobrepeso: ≥ 25</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Como usar")
        st.markdown("""
- **🔮 Preditor Clínico** — Insira os dados do paciente e obtenha a predição com probabilidades por classe.
- **📊 Painel Analítico** — Explore insights do dataset: distribuição, fatores de risco, comportamentos.
- **🤖 Métricas do Modelo** — Veja a performance detalhada: matriz de confusão, importância das features e relatório por classe.
        """)

    with col2:
        st.markdown("### Variáveis utilizadas")
        st.markdown("""
| Tipo | Variáveis |
|---|---|
| Físicas | Gênero, Idade, Altura, Peso, **IMC** |
| Genética | Histórico familiar de sobrepeso |
| Alimentação | FAVC, FCVC, NCP, CAEC, CALC |
| Saúde | Tabagismo, Água (CH2O), SCC |
| Atividade | FAF (exercício), TUE (telas) |
| Mobilidade | Meio de transporte (MTRANS) |
        """)

    st.info("💡 **Aviso**: Este sistema é uma ferramenta de **apoio à decisão** e não substitui avaliação médica profissional.")


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA: PREDITOR CLÍNICO
# ─────────────────────────────────────────────────────────────────────────────
elif pagina == "🔮  Preditor Clínico":
    st.markdown("# 🔮 Preditor Clínico de Obesidade")
    st.markdown("Preencha os dados do paciente nos campos abaixo e clique em **Calcular Predição**.")

    with st.form("prediction_form"):
        st.markdown("### 📋 Dados físicos e demográficos")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = st.selectbox("Gênero", ["Male", "Female"],
                                  format_func=lambda x: "Masculino" if x == "Male" else "Feminino")
        with c2:
            age = st.number_input("Idade (anos)", min_value=14, max_value=61, value=30)
        with c3:
            height = st.number_input("Altura (m)", min_value=1.45, max_value=1.98, value=1.70, step=0.01, format="%.2f")
        with c4:
            weight = st.number_input("Peso (kg)", min_value=39.0, max_value=173.0, value=75.0, step=0.5)

        bmi_preview = weight / (height ** 2)
        st.markdown(f"**IMC calculado:** `{bmi_preview:.2f}`")

        st.markdown("### 🧬 Histórico e hábitos alimentares")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            family_history = st.selectbox("Histórico familiar de sobrepeso",
                                          ["yes", "no"],
                                          format_func=lambda x: "Sim" if x == "yes" else "Não")
        with c2:
            favc = st.selectbox("Consome alimentos calóricos frequentemente (FAVC)",
                                ["yes", "no"],
                                format_func=lambda x: "Sim" if x == "yes" else "Não")
        with c3:
            fcvc = st.selectbox("Frequência de vegetais nas refeições (FCVC)",
                                [1, 2, 3],
                                format_func=lambda x: {1: "1 — Raramente", 2: "2 — Às vezes", 3: "3 — Sempre"}[x],
                                index=1)
        with c4:
            ncp = st.selectbox("Número de refeições principais por dia (NCP)",
                               [1, 2, 3, 4],
                               index=2)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            caec = st.selectbox("Come entre refeições (CAEC)",
                                ["no", "Sometimes", "Frequently", "Always"],
                                format_func=lambda x: {"no": "Não", "Sometimes": "Às vezes",
                                                       "Frequently": "Frequentemente", "Always": "Sempre"}[x],
                                index=1)
        with c2:
            calc = st.selectbox("Consumo de álcool (CALC)",
                                ["no", "Sometimes", "Frequently", "Always"],
                                format_func=lambda x: {"no": "Não bebe", "Sometimes": "Às vezes",
                                                       "Frequently": "Frequentemente", "Always": "Sempre"}[x],
                                index=1)
        with c3:
            ch2o = st.selectbox("Consumo de água/dia (CH2O)",
                                [1, 2, 3],
                                format_func=lambda x: {1: "1 — Menos de 1L", 2: "2 — 1 a 2L", 3: "3 — Mais de 2L"}[x],
                                index=1)
        with c4:
            scc = st.selectbox("Monitora calorias ingeridas (SCC)",
                               ["no", "yes"],
                               format_func=lambda x: "Não" if x == "no" else "Sim")

        st.markdown("### 🏃 Atividade física e comportamento")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            faf = st.selectbox("Frequência de atividade física/semana (FAF)",
                               [0, 1, 2, 3],
                               format_func=lambda x: {0: "0 — Nenhuma", 1: "1 — 1 a 2×/sem",
                                                      2: "2 — 3 a 4×/sem", 3: "3 — 5×/sem ou mais"}[x],
                               index=1)
        with c2:
            tue = st.selectbox("Tempo usando dispositivos eletrônicos/dia (TUE)",
                               [0, 1, 2],
                               format_func=lambda x: {0: "0 — Até 2h", 1: "1 — 3 a 5h", 2: "2 — Mais de 5h"}[x],
                               index=1)
        with c3:
            smoke = st.selectbox("Fuma (SMOKE)",
                                 ["no", "yes"],
                                 format_func=lambda x: "Não" if x == "no" else "Sim")
        with c4:
            mtrans = st.selectbox("Meio de transporte habitual (MTRANS)",
                                  ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
                                  format_func=lambda x: {
                                      "Public_Transportation": "Transporte Público",
                                      "Walking": "A Pé",
                                      "Automobile": "Automóvel",
                                      "Motorbike": "Moto",
                                      "Bike": "Bicicleta",
                                  }[x])

        submitted = st.form_submit_button("🔍 Calcular Predição")

    # ── Resultado da predição ──
    if submitted:
        bmi = weight / (height ** 2)

        # Encoding
        gender_enc   = 1 if gender == "Male" else 0
        fam_enc      = 1 if family_history == "yes" else 0
        favc_enc     = 1 if favc == "yes" else 0
        smoke_enc    = 1 if smoke == "yes" else 0
        scc_enc      = 1 if scc == "yes" else 0
        caec_enc     = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[caec]
        calc_enc     = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[calc]
        mtrans_enc   = {"Walking": 0, "Bike": 1, "Public_Transportation": 2, "Motorbike": 3, "Automobile": 4}[mtrans]

        X_input = pd.DataFrame([[
            gender_enc, age, height, weight, bmi,
            fam_enc, favc_enc, fcvc, ncp,
            caec_enc, smoke_enc, ch2o, scc_enc,
            faf, tue, calc_enc, mtrans_enc,
        ]], columns=FEATURES)

        pred_idx   = model.predict(X_input)[0]
        pred_proba = model.predict_proba(X_input)[0]
        pred_label = obesity_order[pred_idx]
        conf       = pred_proba[pred_idx] * 100
        cfg        = SEVERITY_CONFIG[pred_label]

        st.markdown("---")
        st.markdown("## Resultado da Predição")

        # Card de resultado principal
        st.markdown(
            f"""
            <div class='result-card' style='background:{cfg["bg"]}; border-color:{cfg["color"]};'>
                <div style='font-size:14px; color:{cfg["color"]}; font-weight:600; margin-bottom:6px;'>
                    {cfg["icon"]} Diagnóstico Predito — Confiança: {conf:.1f}%
                </div>
                <div class='result-title' style='color:{cfg["color"]};'>{LABELS_PT[pred_label]}</div>
                <div class='result-sub' style='color:{cfg["color"]};'>
                    Nível de risco: <b>{cfg["risco"]}</b> &nbsp;·&nbsp; IMC calculado: <b>{bmi:.2f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Métricas rápidas
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("IMC", f"{bmi:.2f}", help="Índice de Massa Corporal = peso / altura²")
        with c2:
            st.metric("Confiança da predição", f"{conf:.1f}%")
        with c3:
            bmi_cat = ("Abaixo do peso" if bmi < 18.5 else
                       "Normal" if bmi < 25 else
                       "Sobrepeso" if bmi < 30 else
                       "Obesidade Grau I" if bmi < 35 else
                       "Obesidade Grau II" if bmi < 40 else "Obesidade Grau III")
            st.metric("Categoria OMS pelo IMC", bmi_cat)

        # Probabilidades por classe
        st.markdown("### Probabilidades por classe")
        prob_df = pd.DataFrame({
            "Classe": [LABELS_PT[c] for c in obesity_order],
            "Probabilidade (%)": [round(p * 100, 2) for p in pred_proba],
            "Cor": CORES_CLASSE,
        })

        fig, ax = plt.subplots(figsize=(10, 3.5))
        bars = ax.barh(
            prob_df["Classe"],
            prob_df["Probabilidade (%)"],
            color=prob_df["Cor"],
            height=0.6,
            edgecolor="white",
        )
        ax.set_xlabel("Probabilidade (%)", fontsize=11)
        ax.set_xlim(0, 105)
        for bar, val in zip(bars, prob_df["Probabilidade (%)"]):
            if val > 0.5:
                ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=10, color="#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.set_facecolor("#f9fafb")
        fig.patch.set_facecolor("#f9fafb")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Recomendações
        st.markdown("### 💬 Recomendações clínicas")
        recs = {
            "Insufficient_Weight": [
                "Avaliação nutricional urgente para identificar causa do baixo peso.",
                "Aumentar ingesta calórica com orientação de nutricionista.",
                "Investigar possíveis distúrbios alimentares ou condições médicas.",
            ],
            "Normal_Weight": [
                "Manter os hábitos saudáveis atuais.",
                "Continuar com atividade física regular (3–5× por semana).",
                "Monitorar peso semestralmente.",
            ],
            "Overweight_Level_I": [
                "Reduzir consumo de alimentos ultraprocessados e calóricos.",
                "Aumentar frequência de atividade física para pelo menos 3× por semana.",
                "Consulta com nutricionista para ajuste alimentar.",
            ],
            "Overweight_Level_II": [
                "Acompanhamento médico e nutricional regular.",
                "Atividade física orientada — meta de 150 min/semana.",
                "Reduzir consumo de álcool e lanches entre refeições.",
                "Monitorar pressão arterial e glicemia.",
            ],
            "Obesity_Type_I": [
                "Encaminhamento para equipe multidisciplinar (médico, nutricionista, psicólogo).",
                "Investigar comorbidades: hipertensão, diabetes tipo 2, apneia.",
                "Programa estruturado de emagrecimento com meta de 5–10% do peso atual.",
                "Aumentar consumo de água e reduzir alimentos calóricos.",
            ],
            "Obesity_Type_II": [
                "Avaliação para tratamento farmacológico com endocrinologista.",
                "Monitoramento de comorbidades graves (cardiovascular, metabólico).",
                "Programa intensivo de mudança de estilo de vida.",
                "Considerar encaminhamento para cirurgia bariátrica conforme critérios.",
            ],
            "Obesity_Type_III": [
                "Avaliação cirúrgica bariátrica urgente conforme diretrizes.",
                "Monitoramento intensivo de comorbidades (cardiopatia, DM2, artropatia).",
                "Suporte psicológico contínuo.",
                "Acompanhamento em centro especializado em obesidade grave.",
            ],
        }
        for rec in recs[pred_label]:
            st.markdown(f"- {rec}")


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA: PAINEL ANALÍTICO
# ─────────────────────────────────────────────────────────────────────────────
elif pagina == "📊  Painel Analítico":
    st.markdown("# 📊 Painel Analítico — Obesidade")
    st.markdown("Exploração do dataset com foco em insights clínicos e de saúde pública.")

    obesity_order_local = [
        "Insufficient_Weight", "Normal_Weight",
        "Overweight_Level_I", "Overweight_Level_II",
        "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
    ]
    labels_pt_list = [LABELS_PT[c] for c in obesity_order_local]

    # ── KPIs ──
    st.markdown("### Visão geral")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class='metric-card'><div class='metric-label'>Total de pacientes</div>
        <div class='metric-value'>2.111</div></div>""", unsafe_allow_html=True)
    with c2:
        obese_pct = round(100 * df["Obesity"].str.contains("Obesity").mean(), 1)
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Com algum grau de obesidade</div>
        <div class='metric-value' style='color:#D85A30;'>{obese_pct}%</div></div>""", unsafe_allow_html=True)
    with c3:
        ow_pct = round(100 * df["Obesity"].str.contains("Overweight|Obesity").mean(), 1)
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Sobrepeso ou obesidade</div>
        <div class='metric-value' style='color:#EF9F27;'>{ow_pct}%</div></div>""", unsafe_allow_html=True)
    with c4:
        avg_bmi = round(df["BMI"].mean(), 1)
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>IMC médio geral</div>
        <div class='metric-value'>{avg_bmi}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Gráfico 1: Distribuição ──
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Distribuição das classes de peso")
        counts = df["Obesity"].value_counts()[obesity_order_local].values
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels_pt_list, counts, color=CORES_CLASSE, edgecolor="white", width=0.7)
        ax.set_ylabel("Nº de pacientes", fontsize=10)
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(val), ha="center", fontsize=9, color="#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f9fafb")
        fig.patch.set_facecolor("#f9fafb")
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### IMC médio por categoria")
        bmi_means = [df[df["Obesity"] == c]["BMI"].mean() for c in obesity_order_local]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels_pt_list, bmi_means, color=CORES_CLASSE, edgecolor="white", width=0.7)
        ax.axhline(25, color="#E24B4A", linestyle="--", linewidth=1.2, label="Limiar sobrepeso (25)")
        ax.axhline(30, color="#A32D2D", linestyle="--", linewidth=1.2, label="Limiar obesidade (30)")
        ax.legend(fontsize=9)
        for bar, val in zip(bars, bmi_means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", fontsize=9)
        ax.set_ylabel("IMC médio", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f9fafb")
        fig.patch.set_facecolor("#f9fafb")
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Gráfico 2: Fatores de risco ──
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Histórico familiar por classe (%)")
        fam_pct = [
            100 * (df[df["Obesity"] == c]["family_history"] == "yes").mean()
            for c in obesity_order_local
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(labels_pt_list, fam_pct, color="#7F77DD", edgecolor="white", height=0.6)
        for i, val in enumerate(fam_pct):
            ax.text(val + 1, i, f"{val:.1f}%", va="center", fontsize=9)
        ax.set_xlim(0, 115)
        ax.set_xlabel("% com histórico familiar", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f9fafb")
        fig.patch.set_facecolor("#f9fafb")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Frequência de atividade física por classe")
        faf_means = [df[df["Obesity"] == c]["FAF"].mean() for c in obesity_order_local]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(labels_pt_list, faf_means, marker="o", color="#1D9E75",
                linewidth=2.5, markersize=8)
        ax.fill_between(labels_pt_list, faf_means, alpha=0.15, color="#1D9E75")
        ax.set_ylabel("FAF médio (0–3)", fontsize=10)
        ax.set_ylim(0, 1.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f9fafb")
        fig.patch.set_facecolor("#f9fafb")
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Gráfico 3: Gênero e Transporte ──
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Distribuição por gênero e classe")
        male_counts   = [int((df[df["Obesity"] == c]["Gender"] == "Male").sum())   for c in obesity_order_local]
        female_counts = [int((df[df["Obesity"] == c]["Gender"] == "Female").sum()) for c in obesity_order_local]
        x = np.arange(len(labels_pt_list))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - 0.2, male_counts,   0.38, label="Masculino", color="#378ADD", edgecolor="white")
        ax.bar(x + 0.2, female_counts, 0.38, label="Feminino",  color="#D4537E", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_pt_list, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f9fafb")
        fig.patch.set_facecolor("#f9fafb")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Transporte habitual por classe")
        mtrans_groups = {
            "Automóvel": [46,45,66,94,110,95,1],
            "Transp. Público": [220,200,212,189,236,200,323],
            "A pé": [6,32,9,6,2,1,0],
        }
        mtrans_colors = ["#888780", "#1D9E75", "#378ADD"]
        x = np.arange(len(labels_pt_list))
        fig, ax = plt.subplots(figsize=(6, 4))
        bottom = np.zeros(len(labels_pt_list))
        for (name, vals), color in zip(mtrans_groups.items(), mtrans_colors):
            ax.bar(x, vals, label=name, bottom=bottom, color=color, edgecolor="white")
            bottom += np.array(vals)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_pt_list, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f9fafb")
        fig.patch.set_facecolor("#f9fafb")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Insights ──
    st.markdown("---")
    st.markdown("### 💡 Principais insights clínicos")
    ins1, ins2 = st.columns(2)
    with ins1:
        st.info("**Genética determina risco:** 100% dos pacientes com Obesidade III têm familiar com excesso de peso — a herança genética e ambiental é o fator mais associado aos casos graves.")
        st.warning("**Sedentarismo é marcador:** A atividade física cai de 1,25 (Normal) para 0,66 (Obesidade III). Pacientes obesos praticam menos da metade do exercício dos saudáveis.")
    with ins2:
        st.error("**Padrão de gênero surpreendente:** Obesidade II é quase exclusivamente masculina (295 H × 2 M), enquanto Obesidade III é quase exclusivamente feminina (1 H × 323 M) — padrão raro que merece investigação.")
        st.success("**Dieta calórica escala com obesidade:** O consumo frequente de alimentos calóricos sobe de 72,5% (Normal) para 99,7% (Obesidade III) — intervenção alimentar precoce é essencial.")


# ─────────────────────────────────────────────────────────────────────────────
# PÁGINA: MÉTRICAS DO MODELO
# ─────────────────────────────────────────────────────────────────────────────
elif pagina == "🤖  Métricas do Modelo":
    st.markdown("# 🤖 Métricas e Performance do Modelo")
    st.markdown("Avaliação completa do **Random Forest Classifier** treinado para predição de obesidade.")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Acurácia (test set)</div>
        <div class='metric-value' style='color:#1D9E75;'>{accuracy*100:.2f}%</div>
        <div class='metric-sub'>20% dos dados (423 amostras)</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Média CV (5-fold)</div>
        <div class='metric-value'>{cv_scores.mean()*100:.2f}%</div>
        <div class='metric-sub'>± {cv_scores.std()*100:.2f}%</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='metric-card'><div class='metric-label'>Algoritmo</div>
        <div class='metric-value' style='font-size:16px; padding-top:6px;'>Random Forest</div>
        <div class='metric-sub'>200 árvores · sem profundidade máxima</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class='metric-card'><div class='metric-label'>Features utilizadas</div>
        <div class='metric-value'>17</div>
        <div class='metric-sub'>Incluindo BMI derivado</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Matriz de confusão")
        fig, ax = plt.subplots(figsize=(7, 5.5))
        labels_short = ["Insuf.", "Normal", "OW I", "OW II", "Ob. I", "Ob. II", "Ob. III"]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_short)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Predito vs. Real", fontsize=11)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Importância das features")
        fi_top = fi.head(12).copy()
        fi_top["Importance_pct"] = fi_top["Importance"] * 100
        feat_names_pt = {
            "BMI": "IMC (BMI) ★",
            "Weight": "Peso (Weight)",
            "Age": "Idade",
            "Height": "Altura",
            "Gender_enc": "Gênero",
            "CAEC_enc": "Lanches (CAEC)",
            "FCVC_r": "Vegetais (FCVC)",
            "CALC_enc": "Álcool (CALC)",
            "family_history_enc": "Histórico Familiar",
            "NCP_r": "Refeições (NCP)",
            "FAF_r": "Atividade Física (FAF)",
            "MTRANS_enc": "Transporte (MTRANS)",
        }
        fi_top["Feature_PT"] = fi_top["Feature"].map(feat_names_pt).fillna(fi_top["Feature"])
        colors_fi = ["#1D9E75" if i == 0 else "#378ADD" if i == 1 else "#7F77DD"
                     for i in range(len(fi_top))]
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.barh(fi_top["Feature_PT"][::-1], fi_top["Importance_pct"][::-1],
                color=colors_fi[::-1], edgecolor="white", height=0.65)
        for i, (_, row) in enumerate(fi_top[::-1].iterrows()):
            ax.text(row["Importance_pct"] + 0.3, i, f"{row['Importance_pct']:.2f}%",
                    va="center", fontsize=9)
        ax.set_xlabel("Importância (%)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#f9fafb")
        fig.patch.set_facecolor("#f9fafb")
        patches = [
            mpatches.Patch(color="#1D9E75", label="IMC — feature mais relevante"),
            mpatches.Patch(color="#378ADD", label="Peso — complemento físico"),
            mpatches.Patch(color="#7F77DD", label="Demais features"),
        ]
        ax.legend(handles=patches, fontsize=8, loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Relatório por classe
    st.markdown("#### Relatório de classificação por classe")
    report_df = pd.DataFrame(report).T
    report_df = report_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
    report_df.index = [LABELS_PT.get(i, i) for i in report_df.index]
    report_df = report_df[["precision", "recall", "f1-score", "support"]].astype({
        "precision": float, "recall": float, "f1-score": float, "support": int
    })
    report_df.columns = ["Precisão", "Recall", "F1-Score", "Suporte"]
    st.dataframe(
        report_df.style.format({"Precisão": "{:.2%}", "Recall": "{:.2%}", "F1-Score": "{:.2%}"}),
        use_container_width=True,
    )

    st.markdown("#### Cross-validation (5-fold estratificado)")
    cv_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(5)],
        "Acurácia": [f"{s*100:.2f}%" for s in cv_scores],
    })
    st.dataframe(cv_df, use_container_width=True, hide_index=True)
    st.markdown(
        f"**Média:** `{cv_scores.mean()*100:.2f}%` &nbsp;·&nbsp; "
        f"**Desvio padrão:** `{cv_scores.std()*100:.2f}%` &nbsp;·&nbsp; "
        f"**Variação mínima-máxima:** `{cv_scores.min()*100:.2f}% — {cv_scores.max()*100:.2f}%`"
    )
