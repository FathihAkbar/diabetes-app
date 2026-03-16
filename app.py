# ============================================================
#  APLIKASI PREDIKSI DIABETES — Streamlit
#  Jalankan: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #f8f9fb; }

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #eaecf0;
    text-align: center;
}
.metric-value {
    font-size: 32px;
    font-weight: 600;
    margin: 0;
    line-height: 1.2;
}
.metric-label {
    font-size: 12px;
    color: #6b7280;
    margin: 4px 0 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Result card */
.result-box {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 16px 0;
}
.result-diabetes {
    background: #fff5f5;
    border: 2px solid #fca5a5;
}
.result-sehat {
    background: #f0fdf4;
    border: 2px solid #86efac;
}
.result-title {
    font-size: 28px;
    font-weight: 600;
    margin: 0 0 8px;
}
.result-prob {
    font-size: 48px;
    font-weight: 700;
    margin: 12px 0;
    line-height: 1;
}
.result-sub {
    font-size: 14px;
    color: #6b7280;
    margin: 0;
}

/* Risk badge */
.risk-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 600;
    margin-top: 12px;
}
.risk-tinggi  { background: #fee2e2; color: #b91c1c; }
.risk-sedang  { background: #fef9c3; color: #92400e; }
.risk-rendah  { background: #dcfce7; color: #166534; }

/* Section header */
.section-title {
    font-size: 18px;
    font-weight: 600;
    color: #111827;
    margin: 0 0 4px;
    padding-bottom: 10px;
    border-bottom: 2px solid #e5e7eb;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1e293b;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stSlider > label,
section[data-testid="stSidebar"] .stSelectbox > label,
section[data-testid="stSidebar"] .stRadio > label {
    color: #94a3b8 !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stButton > button {
    width: 100%;
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 14px;
    font-size: 15px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #1d4ed8;
    color: white;
    border: none;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #f1f5f9;
    padding: 4px;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# FUNGSI TRAINING MODEL
# ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def train_model(df):
    df_clean = df.copy()
    df_clean = df_clean[df_clean['gender'] != 'Other']
    df_clean = df_clean.drop_duplicates()

    le_gender  = LabelEncoder()
    le_smoking = LabelEncoder()
    df_clean['gender']          = le_gender.fit_transform(df_clean['gender'])
    df_clean['smoking_history'] = le_smoking.fit_transform(df_clean['smoking_history'])

    X = df_clean.drop('diabetes', axis=1)
    y = df_clean['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree'       : DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    hasil = {}
    for nama, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]
        hasil[nama] = {
            'model' : model,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'acc'   : accuracy_score(y_test, y_pred),
            'auc'   : roc_auc_score(y_test, y_prob),
        }

    model_terbaik = max(hasil, key=lambda n: hasil[n]['auc'])

    return hasil, scaler, le_gender, le_smoking, X_test, y_test, model_terbaik, X.columns.tolist()


# ────────────────────────────────────────────────────────────
# SIDEBAR — INPUT DATA PASIEN
# ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🩺 Data Pasien")
    st.markdown("---")

    gender = st.selectbox("Gender", ["Male", "Female"])
    usia   = st.slider("Usia (tahun)", 1, 80, 40)

    st.markdown("---")
    st.markdown("**Kondisi Medis**")
    hipertensi        = st.radio("Hipertensi", [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)
    penyakit_jantung  = st.radio("Penyakit Jantung", [0, 1], format_func=lambda x: "Ya" if x else "Tidak", horizontal=True)

    st.markdown("---")
    st.markdown("**Gaya Hidup**")
    riwayat_merokok = st.selectbox("Riwayat Merokok", ["never", "current", "former", "ever", "not current", "No Info"])

    st.markdown("---")
    st.markdown("**Hasil Lab**")
    bmi     = st.slider("BMI (kg/m²)", 10.0, 60.0, 25.0, 0.1)
    hba1c   = st.slider("HbA1c (%)", 3.5, 9.0, 5.5, 0.1)
    glukosa = st.slider("Glukosa Darah (mg/dL)", 80, 300, 100)

    st.markdown("---")
    prediksi_btn = st.button("🔍 Prediksi Sekarang")


# ────────────────────────────────────────────────────────────
# HEADER
# ────────────────────────────────────────────────────────────

st.markdown("## 🩺 Sistem Prediksi Diabetes")
st.markdown("Prediksi risiko diabetes berbasis Machine Learning menggunakan dataset 100.000 pasien.")
st.markdown("---")


# ────────────────────────────────────────────────────────────
# UPLOAD DATASET
# ────────────────────────────────────────────────────────────

col_up, col_info = st.columns([2, 3])

with col_up:
    uploaded = st.file_uploader(
        "Upload dataset CSV",
        type=['csv'],
        help="Upload file: diabetes_prediction_dataset.csv dari Kaggle"
    )

with col_info:
    st.info("📂 Upload file **diabetes_prediction_dataset.csv** dari Kaggle untuk memulai.\n\n"
            "Download di: `kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset`")

if uploaded is None:
    st.warning("⬆️ Silakan upload dataset terlebih dahulu untuk melatih model.")
    st.stop()

# Load data
df = pd.read_csv(uploaded)
st.success(f"✅ Dataset dimuat: **{df.shape[0]:,} baris**, **{df.shape[1]} kolom**")

# Training
with st.spinner("⚙️ Melatih model... (30–60 detik pertama kali)"):
    hasil_model, scaler, le_gender, le_smoking, X_test, y_test, model_terbaik, fitur_cols = train_model(df)

st.success(f"🏆 Model terbaik: **{model_terbaik}** (AUC: {hasil_model[model_terbaik]['auc']:.4f})")
st.markdown("---")


# ────────────────────────────────────────────────────────────
# TABS
# ────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Performa Model", "📈 Eksplorasi Data"])


# ══════════════════════════════════════════
# TAB 1: PREDIKSI
# ══════════════════════════════════════════

with tab1:
    if prediksi_btn:
        gender_enc  = le_gender.transform([gender])[0]
        smoking_enc = le_smoking.transform([riwayat_merokok])[0]

        data = np.array([[gender_enc, usia, hipertensi, penyakit_jantung,
                          smoking_enc, bmi, hba1c, glukosa]])
        data_scaled = scaler.transform(data)

        model = hasil_model[model_terbaik]['model']
        prediksi = model.predict(data_scaled)[0]
        prob     = model.predict_proba(data_scaled)[0][1]

        # Hasil utama
        col_res, col_detail = st.columns([1, 1], gap="large")

        with col_res:
            if prediksi == 1:
                risk_class = "risk-tinggi" if prob >= 0.7 else "risk-sedang"
                risk_text  = "Risiko Tinggi" if prob >= 0.7 else "Risiko Sedang"
                st.markdown(f"""
                <div class="result-box result-diabetes">
                    <p class="result-title" style="color:#dc2626;">⚠️ Terdeteksi Diabetes</p>
                    <p class="result-prob" style="color:#dc2626;">{prob:.0%}</p>
                    <p class="result-sub">probabilitas diabetes</p>
                    <span class="risk-badge {risk_class}">{risk_text}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                risk_class = "risk-rendah" if prob < 0.3 else "risk-sedang"
                risk_text  = "Risiko Rendah" if prob < 0.3 else "Risiko Sedang"
                st.markdown(f"""
                <div class="result-box result-sehat">
                    <p class="result-title" style="color:#16a34a;">✅ Tidak Terdeteksi Diabetes</p>
                    <p class="result-prob" style="color:#16a34a;">{prob:.0%}</p>
                    <p class="result-sub">probabilitas diabetes</p>
                    <span class="risk-badge {risk_class}">{risk_text}</span>
                </div>
                """, unsafe_allow_html=True)

            # Gauge bar
            fig_g, ax_g = plt.subplots(figsize=(5, 0.6))
            ax_g.barh(0, 1, color='#e5e7eb', height=0.5)
            color_bar = '#dc2626' if prob >= 0.7 else '#f59e0b' if prob >= 0.4 else '#16a34a'
            ax_g.barh(0, prob, color=color_bar, height=0.5)
            ax_g.set_xlim(0, 1)
            ax_g.axis('off')
            fig_g.patch.set_alpha(0)
            plt.tight_layout(pad=0)
            st.pyplot(fig_g, use_container_width=True)
            plt.close()

        with col_detail:
            st.markdown('<p class="section-title">Data yang Diinput</p>', unsafe_allow_html=True)
            data_dict = {
                "Parameter": ["Gender", "Usia", "BMI", "HbA1c", "Glukosa Darah",
                               "Hipertensi", "Penyakit Jantung", "Riwayat Merokok"],
                "Nilai": [gender, f"{usia} tahun", f"{bmi}", f"{hba1c}%",
                          f"{glukosa} mg/dL",
                          "Ya" if hipertensi else "Tidak",
                          "Ya" if penyakit_jantung else "Tidak",
                          riwayat_merokok]
            }
            st.dataframe(pd.DataFrame(data_dict), hide_index=True, use_container_width=True)

            # Nilai rujukan HbA1c & glukosa
            st.markdown('<p class="section-title" style="margin-top:16px">Nilai Rujukan Medis</p>', unsafe_allow_html=True)
            rujukan = {
                "Parameter": ["HbA1c", "HbA1c", "Glukosa Darah", "Glukosa Darah"],
                "Kategori":  ["Normal", "Diabetes", "Normal (puasa)", "Diabetes (puasa)"],
                "Rentang":   ["< 5.7%", "≥ 6.5%", "< 100 mg/dL", "≥ 126 mg/dL"]
            }
            st.dataframe(pd.DataFrame(rujukan), hide_index=True, use_container_width=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color: #9ca3af;">
            <div style="font-size: 56px;">🩺</div>
            <p style="font-size: 18px; font-weight: 500; margin: 16px 0 8px; color: #374151;">Siap untuk memprediksi</p>
            <p style="font-size: 14px;">Isi data pasien di sidebar kiri, lalu klik <strong>Prediksi Sekarang</strong></p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB 2: PERFORMA MODEL
# ══════════════════════════════════════════

with tab2:
    st.markdown('<p class="section-title">Perbandingan Semua Model</p>', unsafe_allow_html=True)

    # Metric cards
    col_m = st.columns(4)
    colors_m = ['#2563eb', '#f59e0b', '#16a34a', '#dc2626']
    for i, (nama, hasil) in enumerate(hasil_model.items()):
        with col_m[i]:
            is_best = nama == model_terbaik
            border  = "2px solid #2563eb" if is_best else "1px solid #eaecf0"
            badge   = "<br><span style='font-size:10px;background:#dbeafe;color:#1e40af;padding:2px 8px;border-radius:99px;'>TERBAIK</span>" if is_best else ""
            st.markdown(f"""
            <div class="metric-card" style="border:{border}">
                <p class="metric-value" style="color:{colors_m[i]}">{hasil['auc']:.3f}</p>
                <p class="metric-label">AUC-ROC{badge}</p>
                <p style="font-size:11px;color:#9ca3af;margin:6px 0 0">{nama}</p>
                <p style="font-size:13px;color:#374151;margin:2px 0 0">Akurasi: {hasil['acc']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_roc, col_cm = st.columns(2, gap="large")

    with col_roc:
        st.markdown('<p class="section-title">ROC Curve</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f9fb')
        colors_roc = ['#2563eb', '#f59e0b', '#16a34a', '#dc2626']
        for (nama, hasil), c in zip(hasil_model.items(), colors_roc):
            fpr, tpr, _ = roc_curve(y_test, hasil['y_prob'])
            lw = 2.5 if nama == model_terbaik else 1.5
            ax.plot(fpr, tpr, color=c, lw=lw, label=f"{nama} ({hasil['auc']:.3f})",
                    alpha=1.0 if nama == model_terbaik else 0.6)
        ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.3)
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(alpha=0.3)
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_cm:
        st.markdown(f'<p class="section-title">Confusion Matrix — {model_terbaik}</p>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        fig2.patch.set_facecolor('white')
        cm = confusion_matrix(y_test, hasil_model[model_terbaik]['y_pred'])
        sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax2,
                    xticklabels=['Tidak\nDiabetes','Diabetes'],
                    yticklabels=['Tidak\nDiabetes','Diabetes'],
                    linewidths=0.5, linecolor='#e5e7eb',
                    annot_kws={'size': 14, 'weight': 'bold'})
        ax2.set_xlabel('Prediksi', fontsize=11)
        ax2.set_ylabel('Aktual', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # Feature importance
    st.markdown('<p class="section-title">Fitur Paling Berpengaruh (Random Forest)</p>', unsafe_allow_html=True)
    rf = hasil_model['Random Forest']['model']
    imp = pd.Series(rf.feature_importances_, index=fitur_cols).sort_values(ascending=True)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    fig3.patch.set_facecolor('white')
    ax3.set_facecolor('#f8f9fb')
    colors_bar = ['#2563eb' if v == imp.max() else '#93c5fd' for v in imp.values]
    bars = ax3.barh(imp.index, imp.values, color=colors_bar, edgecolor='white', height=0.6)
    for bar, val in zip(bars, imp.values):
        ax3.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=10, color='#374151')
    ax3.set_xlim(0, imp.max() * 1.2)
    ax3.spines[['top','right','left']].set_visible(False)
    ax3.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════
# TAB 3: EKSPLORASI DATA
# ══════════════════════════════════════════

with tab3:
    st.markdown('<p class="section-title">Ringkasan Dataset</p>', unsafe_allow_html=True)

    col_s = st.columns(4)
    total       = len(df)
    n_diabetes  = df['diabetes'].sum()
    n_sehat     = total - n_diabetes
    persen_dm   = n_diabetes / total

    for col, val, label, color in zip(
        col_s,
        [f"{total:,}", f"{n_diabetes:,}", f"{n_sehat:,}", f"{persen_dm:.1%}"],
        ["Total Pasien", "Diabetes", "Tidak Diabetes", "Prevalensi DM"],
        ["#2563eb", "#dc2626", "#16a34a", "#f59e0b"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value" style="color:{color}">{val}</p>
                <p class="metric-label">{label}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_v1, col_v2 = st.columns(2, gap="large")

    with col_v1:
        st.markdown('<p class="section-title">Distribusi HbA1c</p>', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        fig4.patch.set_facecolor('white')
        ax4.set_facecolor('#f8f9fb')
        ax4.hist(df[df['diabetes']==0]['HbA1c_level'], bins=30, alpha=0.7, color='#3b82f6', label='Tidak Diabetes')
        ax4.hist(df[df['diabetes']==1]['HbA1c_level'], bins=30, alpha=0.7, color='#ef4444', label='Diabetes')
        ax4.axvline(6.5, color='#111827', linestyle='--', lw=1.5, label='Batas DM (6.5%)')
        ax4.set_xlabel('HbA1c (%)', fontsize=11)
        ax4.legend(fontsize=9)
        ax4.spines[['top','right']].set_visible(False)
        ax4.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

    with col_v2:
        st.markdown('<p class="section-title">Distribusi Glukosa Darah</p>', unsafe_allow_html=True)
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        fig5.patch.set_facecolor('white')
        ax5.set_facecolor('#f8f9fb')
        ax5.hist(df[df['diabetes']==0]['blood_glucose_level'], bins=30, alpha=0.7, color='#3b82f6', label='Tidak Diabetes')
        ax5.hist(df[df['diabetes']==1]['blood_glucose_level'], bins=30, alpha=0.7, color='#ef4444', label='Diabetes')
        ax5.axvline(126, color='#111827', linestyle='--', lw=1.5, label='Batas DM (126 mg/dL)')
        ax5.set_xlabel('Glukosa Darah (mg/dL)', fontsize=11)
        ax5.legend(fontsize=9)
        ax5.spines[['top','right']].set_visible(False)
        ax5.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close()

    # Preview data
    st.markdown('<p class="section-title">Preview Dataset (10 baris pertama)</p>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
