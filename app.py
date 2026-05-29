"""
TECH CHALLENGE - FASE 4 | POSTECH FIAP - Data Analytics
Autor: Clayton Dias Santos
Pipeline completo de Machine Learning para Predição de Obesidade
Modelo: Random Forest Classifier | Acurácia: 98.35%
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CARREGAMENTO DOS DADOS
# ─────────────────────────────────────────────
print("=" * 60)
print("TECH CHALLENGE - FASE 4 | POSTECH FIAP")
print("Predição de Nível de Obesidade - Pipeline de ML")
print("=" * 60)

df = pd.read_csv("Obesity.csv")
print(f"\n✔ Dataset carregado: {df.shape[0]} registros, {df.shape[1]} colunas")
print(f"\nDistribuição do Target (Obesity):")
print(df["Obesity"].value_counts())


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[STEP 1] Feature Engineering...")

# Calcular BMI (Índice de Massa Corporal) — feature mais importante
df["BMI"] = df["Weight"] / (df["Height"] ** 2)

# Arredondar variáveis ordinais com ruído decimal
df["FCVC_r"] = df["FCVC"].round().astype(int)  # Frequência vegetais (1-3)
df["NCP_r"]  = df["NCP"].round().astype(int)   # Nº refeições (1-4)
df["CH2O_r"] = df["CH2O"].round().astype(int)  # Consumo água (1-3)
df["FAF_r"]  = df["FAF"].round().astype(int)   # Atividade física (0-3)
df["TUE_r"]  = df["TUE"].round().astype(int)   # Tempo tela (0-2)

# Encoding de variáveis binárias
binary_cols = ["Gender", "family_history", "FAVC", "SMOKE", "SCC"]
le = LabelEncoder()
for col in binary_cols:
    df[col + "_enc"] = le.fit_transform(df[col])

# Encoding ordinal de variáveis categóricas
caec_map   = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
calc_map   = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
mtrans_map = {
    "Walking": 0, "Bike": 1, "Public_Transportation": 2,
    "Motorbike": 3, "Automobile": 4
}

df["CAEC_enc"]   = df["CAEC"].map(caec_map)
df["CALC_enc"]   = df["CALC"].map(calc_map)
df["MTRANS_enc"] = df["MTRANS"].map(mtrans_map)

print("✔ Features criadas: BMI, variáveis ordinais arredondadas, encodings categóricos")


# ─────────────────────────────────────────────
# 3. PREPARAÇÃO DO TARGET
# ─────────────────────────────────────────────
obesity_order = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I", "Overweight_Level_II",
    "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"
]
obesity_map = {v: i for i, v in enumerate(obesity_order)}
df["target"] = df["Obesity"].map(obesity_map)


# ─────────────────────────────────────────────
# 4. SELEÇÃO DE FEATURES
# ─────────────────────────────────────────────
FEATURES = [
    "Gender_enc", "Age", "Height", "Weight", "BMI",
    "family_history_enc", "FAVC_enc", "FCVC_r", "NCP_r",
    "CAEC_enc", "SMOKE_enc", "CH2O_r", "SCC_enc",
    "FAF_r", "TUE_r", "CALC_enc", "MTRANS_enc"
]

X = df[FEATURES]
y = df["target"]

print(f"\n✔ Total de features selecionadas: {len(FEATURES)}")


# ─────────────────────────────────────────────
# 5. SPLIT TREINO / TESTE
# ─────────────────────────────────────────────
print("\n[STEP 2] Divisão Treino/Teste (80/20, estratificada)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✔ Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")


# ─────────────────────────────────────────────
# 6. TREINAMENTO DO MODELO
# ─────────────────────────────────────────────
print("\n[STEP 3] Treinando Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✔ Modelo treinado com sucesso!")


# ─────────────────────────────────────────────
# 7. AVALIAÇÃO DO MODELO
# ─────────────────────────────────────────────
print("\n[STEP 4] Avaliação do Modelo...")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n{'='*60}")
print(f"  ACURÁCIA GERAL: {acc:.4f} ({acc*100:.2f}%)")
print(f"{'='*60}")

print("\nRelatório de Classificação por Classe:")
print(classification_report(y_test, y_pred, target_names=obesity_order))

# Cross-validation
print("[STEP 5] Cross-Validation (5-fold estratificado)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
print(f"✔ CV Scores: {cv_scores.round(4)}")
print(f"✔ Média: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ─────────────────────────────────────────────
# 8. IMPORTÂNCIA DAS FEATURES
# ─────────────────────────────────────────────
print("\n[STEP 6] Importância das Features:")
fi_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

fi_df["Importance_pct"] = (fi_df["Importance"] * 100).round(2)
print(fi_df.to_string(index=False))


# ─────────────────────────────────────────────
# 9. FUNÇÃO DE PREDIÇÃO (para deploy/Streamlit)
# ─────────────────────────────────────────────
def predict_obesity(
    gender: str,          # "Male" / "Female"
    age: float,
    height: float,        # em metros
    weight: float,        # em kg
    family_history: str,  # "yes" / "no"
    favc: str,            # "yes" / "no"
    fcvc: int,            # 1, 2 ou 3
    ncp: int,             # 1 a 4
    caec: str,            # "no","Sometimes","Frequently","Always"
    smoke: str,           # "yes" / "no"
    ch2o: int,            # 1, 2 ou 3
    scc: str,             # "yes" / "no"
    faf: int,             # 0 a 3
    tue: int,             # 0 a 2
    calc: str,            # "no","Sometimes","Frequently","Always"
    mtrans: str           # "Walking","Bike","Public_Transportation","Motorbike","Automobile"
) -> dict:
    """
    Prediz o nível de obesidade dado o perfil do paciente.
    Retorna dict com label predita, probabilidades e BMI calculado.
    """
    bmi = weight / (height ** 2)

    gender_enc         = 1 if gender == "Male" else 0
    family_history_enc = 1 if family_history == "yes" else 0
    favc_enc           = 1 if favc == "yes" else 0
    smoke_enc          = 1 if smoke == "yes" else 0
    scc_enc            = 1 if scc == "yes" else 0
    caec_enc           = {"no":0,"Sometimes":1,"Frequently":2,"Always":3}[caec]
    calc_enc           = {"no":0,"Sometimes":1,"Frequently":2,"Always":3}[calc]
    mtrans_enc         = {"Walking":0,"Bike":1,"Public_Transportation":2,"Motorbike":3,"Automobile":4}[mtrans]

    X_input = pd.DataFrame([[
        gender_enc, age, height, weight, bmi,
        family_history_enc, favc_enc, fcvc, ncp,
        caec_enc, smoke_enc, ch2o, scc_enc,
        faf, tue, calc_enc, mtrans_enc
    ]], columns=FEATURES)

    pred_idx  = model.predict(X_input)[0]
    pred_prob = model.predict_proba(X_input)[0]

    label_pt = {
        "Insufficient_Weight": "Abaixo do Peso",
        "Normal_Weight":        "Peso Normal",
        "Overweight_Level_I":   "Sobrepeso Grau I",
        "Overweight_Level_II":  "Sobrepeso Grau II",
        "Obesity_Type_I":       "Obesidade Grau I",
        "Obesity_Type_II":      "Obesidade Grau II",
        "Obesity_Type_III":     "Obesidade Grau III",
    }

    result = {
        "prediction":       obesity_order[pred_idx],
        "prediction_pt":    label_pt[obesity_order[pred_idx]],
        "confidence":       round(float(pred_prob[pred_idx]) * 100, 2),
        "bmi":              round(bmi, 2),
        "probabilities":    {obesity_order[i]: round(float(p)*100, 2)
                             for i, p in enumerate(pred_prob)}
    }
    return result


# ─────────────────────────────────────────────
# 10. TESTE DA FUNÇÃO DE PREDIÇÃO
# ─────────────────────────────────────────────
print("\n[STEP 7] Teste da Função de Predição:")
print("-" * 60)

# Caso 1: Perfil de risco elevado
resultado1 = predict_obesity(
    gender="Male", age=35, height=1.70, weight=110,
    family_history="yes", favc="yes", fcvc=1, ncp=3,
    caec="Frequently", smoke="no", ch2o=1, scc="no",
    faf=0, tue=2, calc="Sometimes", mtrans="Automobile"
)
print("Paciente 1 (alto risco):")
print(f"  → Predição: {resultado1['prediction_pt']}")
print(f"  → Confiança: {resultado1['confidence']}%")
print(f"  → BMI calculado: {resultado1['bmi']}")

# Caso 2: Perfil saudável
resultado2 = predict_obesity(
    gender="Female", age=25, height=1.65, weight=60,
    family_history="no", favc="no", fcvc=3, ncp=3,
    caec="Sometimes", smoke="no", ch2o=3, scc="yes",
    faf=3, tue=1, calc="no", mtrans="Walking"
)
print("\nPaciente 2 (perfil saudável):")
print(f"  → Predição: {resultado2['prediction_pt']}")
print(f"  → Confiança: {resultado2['confidence']}%")
print(f"  → BMI calculado: {resultado2['bmi']}")

print("\n" + "=" * 60)
print("Pipeline concluída com sucesso!")
print("Modelo pronto para deploy via Streamlit.")
print("=" * 60)


# ─────────────────────────────────────────────
# SUMÁRIO DE RESULTADOS
# ─────────────────────────────────────────────
print(f"""
RESUMO DO MODELO
─────────────────────────────────────────────
Algoritmo:          Random Forest Classifier
Nº de Árvores:      200
Acurácia (test):    {acc*100:.2f}%
CV Média (5-fold):  {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%
Feature + relevante: BMI ({fi_df.iloc[0]['Importance_pct']}%)
─────────────────────────────────────────────
""")