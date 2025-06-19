import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Prediksi Peluang Kerja", page_icon="🎓", layout="centered")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset_Siswa.csv")
    # Mapping nilai kategorikal
    df['ExtracurricularActivities'] = df['ExtracurricularActivities'].map({'Yes': 1, 'No': 0})
    df['PlacementTraining'] = df['PlacementTraining'].map({'Yes': 1, 'No': 0})
    df['PlacementStatus'] = df['PlacementStatus'].map({'Placed': 1, 'NotPlaced': 0})
    return df

df = load_data()

feature_cols = [
    'IPK_4', 'Internships', 'Projects', 'Workshops/Certifications',
    'AptitudeTestScore', 'SoftSkillsRating', 'ExtracurricularActivities',
    'PlacementTraining'
]
X = df[feature_cols]
y = df['PlacementStatus']

# --- CACHE TRAINING MODEL ---
@st.cache_resource
def train_model(X, y):
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return model

model = train_model(X, y)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://cdn-icons-png.flaticon.com/512/3135/3135755.png" width="90" style="margin-bottom: 10px;" />
            <h2 style="color:#4F8BF9; margin-bottom:0; margin-top:0;">Career Predictor</h2>
            <span style="background-color:#28C76F; color:white; border-radius:12px; padding:4px 12px; font-size:14px;">
                Data Mining & ML
            </span>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("**🔍 Prediksi peluang kamu diterima kerja setelah lulus berdasarkan data akademik dan pengalaman.**")
    st.markdown(
        """
        <ul style="padding-left: 20px;">
        <li>🎓 <b>IPK, Magang, Proyek</b></li>
        <li>📜 <b>Sertifikasi & Workshop</b></li>
        <li>🤝 <b>Soft Skills & Ekstrakurikuler</b></li>
        <li>🧑‍💻 <b>Pelatihan Penempatan</b></li>
        </ul>
        """, unsafe_allow_html=True
    )
    st.markdown("---")
    st.info("Tips: Ubah nilai input di form utama untuk simulasi peluang terbaikmu!")
    st.markdown(
        """
        <hr>
        <center>
        <small style="color:gray;">© 2025 | Universitas Pelita Bangsa | Chaerul Hidayat & Reza Maulana | Powered by Streamlit</small>
        </center>
        """, unsafe_allow_html=True
    )

# --- PENJELASAN MODEL DAN TABEL KONTRIBUSI FAKTOR ---
st.title("📈 Prediksi Peluang Mendapatkan Pekerjaan Setelah Lulus Kuliah")
st.markdown("""
### 🧠 Tentang Model Prediksi

Model yang digunakan untuk memprediksi peluang mahasiswa **Mendapatkan Kerja Setelah Lulus** adalah **Regresi Logistik (Logistic Regression)**.
Model ini menganalisis seberapa besar pengaruh setiap faktor (misal: IPK, magang, proyek, sertifikasi, soft skills, dsb.) terhadap peluang mahasiswa untuk berhasil mendapatkan penempatan kerja berdasarkan data riil mahasiswa.

Setiap fitur memiliki bobot (koefisien) yang menunjukkan **seberapa besar kontribusinya** terhadap peluang ditempatkan kerja. Semakin besar nilai koefisien (positif), semakin besar pula pengaruh faktor tersebut dalam meningkatkan peluang penempatan kerja.
""")

rata2 = X.mean()
coef = pd.Series(model.coef_[0], index=feature_cols)
kontribusi = rata2 * coef

nama_fitur = {
    'IPK_4': 'IPK (skala 4)',
    'Internships': 'Magang',
    'Projects': 'Proyek',
    'Workshops/Certifications': 'Sertifikasi/Workshop',
    'AptitudeTestScore': 'Tes Aptitude',
    'SoftSkillsRating': 'Soft Skills',
    'ExtracurricularActivities': 'Ekstrakurikuler',
    'PlacementTraining': 'Pelatihan Penempatan'
}
tabel = pd.DataFrame({
    "Faktor": [nama_fitur[x] for x in X.columns],
    "Rata-rata Data": rata2.round(2).values,
    "Koefisien Model": coef.round(2).values,
    "Kontribusi": kontribusi.round(2).values
}).sort_values("Kontribusi", ascending=False).reset_index(drop=True)

st.markdown("### 🔬 Faktor-Faktor Penentu Peluang Mendapatkan Kerja")
st.dataframe(tabel, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("## 🎯 Input Data Untuk Prediksi")

def reset_prediksi():
    st.session_state['prediksi'] = None

col1, col2 = st.columns(2)
with col1:
    ipk = st.slider('IPK (skala 4)', 0.00, 4.00, float(df['IPK_4'].mean()), 0.01, format="%.2f", key="ipk", on_change=reset_prediksi)
    internships = st.slider('Jumlah Magang', 0, 10, int(df['Internships'].mean()), 1, key="internships", on_change=reset_prediksi)
    projects = st.slider('Jumlah Proyek', 0, 10, int(df['Projects'].mean()), 1, key="projects", on_change=reset_prediksi)
    workshops = st.slider('Workshop/Sertifikasi', 0, 10, int(df['Workshops/Certifications'].mean()), 1, key="workshops", on_change=reset_prediksi)
with col2:
    aptitude = st.slider('Skor Tes Aptitude (Bakat)', int(df['AptitudeTestScore'].min()), int(df['AptitudeTestScore'].max()), int(df['AptitudeTestScore'].mean()), 1, key="aptitude", on_change=reset_prediksi)
    softskills = st.slider('Rating Soft Skills', 0.0, 5.0, float(df['SoftSkillsRating'].mean()), 0.1, format="%.1f", key="softskills", on_change=reset_prediksi)
    extracurricular = st.selectbox('Aktif Ekstrakurikuler?', ['Tidak', 'Ya'], key="extracurricular", on_change=reset_prediksi)
    placement_training = st.selectbox('Ikut Pelatihan Penempatan?', ['Tidak', 'Ya'], key="placement_training", on_change=reset_prediksi)

# Mapping input ke format model
extracurricular_val = 1 if extracurricular == 'Ya' else 0
placement_training_val = 1 if placement_training == 'Ya' else 0

# Tombol prediksi
if st.button("🔮 Prediksi Peluang"):
    input_data = np.array([[ipk, internships, projects, workshops, aptitude, softskills,
                            extracurricular_val, placement_training_val]])
    prob = model.predict_proba(input_data)[0][1]
    st.session_state['prediksi'] = prob

# Tampilkan hasil hanya jika sudah prediksi & input belum berubah
if 'prediksi' in st.session_state and st.session_state['prediksi'] is not None:
    prob = st.session_state['prediksi']
    st.markdown("## Hasil Prediksi")
    st.progress(min(int(prob*100), 100), text="Peluang ditempatkan kerja")
    st.markdown(f"<h2 style='color:#28C76F'>{prob*100:.1f}%</h2>", unsafe_allow_html=True)
    if prob >= 0.8:
        st.success("Peluang sangat tinggi untuk Mendapatkan kerja! 🚀")
    elif prob >= 0.6:
        st.info("Peluang cukup baik, tingkatkan pengalaman & sertifikasi! 💡")
    elif prob >= 0.4:
        st.warning("Peluang sedang, perbanyak magang/proyek dan skill. 🔧")
    else:
        st.error("Peluang rendah, perlu pengembangan diri lebih lanjut. 📚")
    st.caption("Simulasi ini hanya prediksi berbasis data. Hasil aktual bisa berbeda tergantung banyak faktor lain.")

# --- FAQ INTERAKTIF DI BAWAHNYA ---
st.markdown("---")
st.markdown("## ❓ Tanya Jawab Seputar Peluang Kerja Setelah Lulus")
faq_list = [
    {
        "q": "Apakah IPK tinggi pasti cepat dapat kerja?",
        "a": "IPK tinggi memang membantu, tapi pengalaman magang, proyek, sertifikasi, dan soft skills juga sangat penting untuk meningkatkan peluang kerja."
    },
    {
        "q": "Seberapa penting magang untuk fresh graduate?",
        "a": "Magang sangat penting karena memberikan pengalaman kerja nyata yang dicari perusahaan. Magang juga membantu membangun jaringan dan memahami dunia kerja."
    },
    {
        "q": "Apakah aktif organisasi kampus berpengaruh pada peluang kerja?",
        "a": "Ya, aktif organisasi menunjukkan kemampuan komunikasi dan leadership yang menjadi nilai tambah di mata perekrut."
    },
    {
        "q": "Bagaimana jika IPK saya sedang tapi pengalaman magang dan proyek banyak?",
        "a": "Peluang tetap besar! Data menunjukkan pengalaman magang dan proyek bisa mengimbangi IPK yang sedang."
    },
    {
        "q": "Apakah pelatihan penempatan kerja benar-benar membantu?",
        "a": "Iya, pelatihan seperti simulasi interview dan pembuatan CV terbukti meningkatkan peluang diterima kerja."
    },
    {
        "q": "Bagaimana cara meningkatkan soft skills?",
        "a": "Ikut organisasi, aktif di kelas, ikut pelatihan komunikasi, dan sering latihan presentasi atau kerja kelompok adalah cara efektif meningkatkan soft skills."
    },
    {
        "q": "Apa yang harus dilakukan jika hasil prediksi peluang kerja saya rendah?",
        "a": "Fokuslah menambah pengalaman magang, proyek, sertifikasi, aktif organisasi, dan ikuti pelatihan penempatan untuk meningkatkan peluang kerja."
    }
]

for item in faq_list:
    with st.expander(f"❓ {item['q']}"):
        st.write(item['a'])

st.markdown(
    """
    <hr>
    <center>
    <small>© 2025 | Data Mining & Machine Learning | Chaerul Hidayat & Reza Maulana</small>
    </center>
    """, unsafe_allow_html=True
)
