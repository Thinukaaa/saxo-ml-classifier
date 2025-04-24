import streamlit as st
import base64
import numpy as np
import pandas as pd
import joblib
import pyodbc
import bcrypt
import zipfile, io
from datetime import datetime
import librosa
import plotly.express as px
# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="SaxoFy ML", layout="wide")
# ─── Background helper ────────────────────────────────────────────────────────
def set_background(image_path: str):
    """Reads a local image and injects it as full‐page CSS background."""
    with open(image_path, "rb") as img:
        b64 = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
        }}
        /* dim sidebar so text remains legible */
        .css-1d391kg {{
            background: rgba(0,0,0,0.6) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ─── Helpers ─────────────────────────────────────────────────────────────────
def get_connection():
    return pyodbc.connect(
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=MSISWORDTHINUKA;"
        "Database=saxo_db;"
        "Trusted_Connection=yes;"
    )

def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

def verify_password(pw: str, hsh: str) -> bool:
    return bcrypt.checkpw(pw.encode(), hsh.encode())

@st.cache_resource
def load_models():
    sax   = joblib.load("models/svm_saxophone_model.pkl")
    style = joblib.load("models/svm_style_model.pkl")
    return sax, style

sax_model, style_model = load_models()

def extract_features(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    feat  = np.mean(mfcc, axis=1).reshape(1, -1)
    return feat, (y, sr)

def footer():
    st.markdown(
        "<hr><center>© SaxoFy Saxophone Classifier 2025</center>",
        unsafe_allow_html=True
    )

# ─── Session defaults ─────────────────────────────────────────────────────────
for key, default in {
    "authenticated": False,
    "user_id": None,
    "user_name": None,
    "page": "Login"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Sidebar navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    if not st.session_state.authenticated:
        options = ["Login", "Register"]
    else:
        options = ["Home", "Visualize", "Upload Audio", "Batch Upload",
                   "View History", "Export Data", "Logout"]
    choice = st.radio("Menu", options, index=options.index(st.session_state.page))
    st.session_state.page = choice

# ─── PAGES ────────────────────────────────────────────────────────────────────

## LOGIN ##
if st.session_state.page == "Login":
    # set the login/register background
    set_background("assets/sax_bg01.png")

    # center‑aligned header
    st.markdown(
        "<div style='text-align:center; margin-top:1rem;'>"
        "<h2>🔐 Login</h2>"
        "</div>",
        unsafe_allow_html=True
    )

    # create three columns: left & right are empty, middle holds the form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Narrower text inputs - they will fill the col2 width
        user_input = st.text_input("Username or Email")
        pwd        = st.text_input("Password", type="password")

        # center the button as well
        if st.button("Login", key="login_btn"):
            conn = get_connection(); cur = conn.cursor()
            cur.execute(
                "SELECT user_id,password,name FROM dbo.Users WHERE email=? OR name=?",
                user_input, user_input
            )
            row = cur.fetchone(); conn.close()
            if row and verify_password(pwd, row.password):
                st.session_state.authenticated = True
                st.session_state.user_id       = row.user_id
                st.session_state.user_name     = row.name
                st.session_state.page          = "Home"
                st.rerun()
            else:
                st.error("❌ Invalid username/email or password")

    footer()


## REGISTER ##
elif st.session_state.page == "Register":
    set_background("assets/sax_bg01.png")

    st.header("🆕 Register")
    name    = st.text_input("Full Name")
    email   = st.text_input("Email")
    pwd     = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")
    if st.button("Register"):
        if not (name and email and pwd):
            st.error("All fields required")
        elif pwd != confirm:
            st.error("Passwords do not match")
        else:
            try:
                conn = get_connection(); cur = conn.cursor()
                cur.execute(
                    "INSERT INTO dbo.Users(name,email,password,role,created_at,updated_at) "
                    "VALUES(?,?,?,?,GETDATE(),GETDATE())",
                    name, email, hash_password(pwd), "student"
                )
                conn.commit()
                st.success("✅ Registered! Now switch to Login.")
            except Exception as e:
                st.error(f"❌ {e}")
            finally:
                conn.close()
    footer()

## LOGOUT ##
elif st.session_state.page == "Logout":
    st.session_state.authenticated = False
    st.session_state.user_id       = None
    st.session_state.user_name     = None
    st.session_state.page          = "Login"
    st.rerun()

## HOME ##
elif st.session_state.page == "Home":
    set_background("assets/sax_bg.png")
    st.title(f"👋 Welcome, {st.session_state.user_name}!")
    # fetch metrics
    conn = get_connection()
    stats = pd.read_sql(
        """
        SELECT COUNT(*) AS total_preds, AVG(confidence_score) AS avg_conf
        FROM dbo.Predictions
        WHERE user_id = ?
        """,
        conn, params=[st.session_state.user_id]
    )
    conn.close()
    total    = int(stats.at[0, "total_preds"])
    avg_conf = stats.at[0, "avg_conf"] or 0.0

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Predictions", f"{total}")
    with col2:
        st.metric("Average Confidence", f"{avg_conf*100:.1f}%")
    st.markdown("---")
    a, b, c = st.columns(3, gap="large")
    with a:
        if st.button("🔍 Visualize Audio"):
            st.session_state.page = "Visualize"; st.rerun()
    with b:
        if st.button("📤 Upload & Predict"):
            st.session_state.page = "Upload Audio"; st.rerun()
    with c:
        if st.button("🗂️ Batch Upload"):
            st.session_state.page = "Batch Upload"; st.rerun()

    d, e, _ = st.columns(3, gap="large")
    with d:
        if st.button("📜 View History"):
            st.session_state.page = "View History"; st.rerun()
    with e:
        if st.button("📥 Export Data"):
            st.session_state.page = "Export Data"; st.rerun()
    st.markdown("")  # spacing
    footer()

## VISUALIZE ##
elif st.session_state.page == "Visualize":
    st.header("🎛️ Audio Visualizer")
    uploaded = st.file_uploader("Upload a .wav/.mp3", type=["wav","mp3"])
    if uploaded:
        feats,(y,sr) = extract_features(uploaded.read())
        t    = np.linspace(0, len(y)/sr, num=len(y))
        df_w = pd.DataFrame({"time":t,"amplitude":y})
        st.plotly_chart(px.line(df_w, x="time", y="amplitude", title="Waveform"), use_container_width=True)
        S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        st.plotly_chart(px.imshow(S_db, origin="lower",
                                  labels={'x':'Frame','y':'Freq','color':'dB'},
                                  title="Spectrogram"),
                        use_container_width=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        st.plotly_chart(px.imshow(mfcc, origin="lower",
                                  labels={'x':'Frame','y':'MFCC','color':'dB'},
                                  title="MFCC"),
                        use_container_width=True)
    footer()

## UPLOAD & PREDICT ##
elif st.session_state.page == "Upload Audio":
    st.header("🎷 Upload & Predict")
    uploaded = st.file_uploader("Choose audio", type=["wav","mp3"])
    if uploaded:
        feats,(y,sr) = extract_features(uploaded.read())
        st.audio(uploaded)
        if st.button("Run Prediction"):
            sax_pred   = sax_model.predict(feats)[0]
            style_pred = style_model.predict(feats)[0]
            sax_probs  = sax_model.predict_proba(feats)[0]
            style_probs= style_model.predict_proba(feats)[0]
            conf       = float(np.round((sax_probs.max()+style_probs.max())/2,3))

            c1,c2,c3 = st.columns(3)
            with c1:
                st.subheader("🎷 Sax Type");    st.metric("", sax_pred)
            with c2:
                st.subheader("🎼 Style");       st.metric("", style_pred)
            with c3:
                st.subheader("🔒 Confidence")
                st.progress(int(conf*100)); st.caption(f"{conf*100:.1f}% sure")

            t    = np.linspace(0, len(y)/sr, len(y))
            df_w = pd.DataFrame({"time":t,"amplitude":y})
            st.plotly_chart(px.line(df_w, x="time", y="amplitude", 
                                    title="Waveform Snippet", height=150),
                            use_container_width=True)

            df_sax   = pd.DataFrame({"Type": sax_model.classes_,   "Prob": sax_probs})
            df_style = pd.DataFrame({"Style": style_model.classes_, "Prob": style_probs})
            st.table(df_sax.nlargest(3,"Prob").style.format({"Prob":"{:.1%}"}))
            st.table(df_style.nlargest(3,"Prob").style.format({"Prob":"{:.1%}"}))

            st.markdown(f"""
            > **Prediction:** A _{sax_pred}_ sax in _{style_pred}_ style  
            > **Confidence:** {conf*100:.1f}%  
            > _“Top MFCC bands match classic {style_pred} inflections.”_
            """)

            conn = get_connection(); cur = conn.cursor()
            cur.execute(
                "INSERT INTO dbo.Predictions"
                "(user_id,audio_id,saxophone_type,performance_style,confidence_score,predicted_at) "
                "VALUES(?,?,?,?,?,?);",
                st.session_state.user_id, None,
                sax_pred, style_pred, conf, datetime.now()
            )
            conn.commit(); cur.close(); conn.close()

            fb = st.radio("Was this correct?", ["Yes","No"])
            if st.button("Submit Feedback"):
                conn = get_connection(); cur = conn.cursor()
                last_id = cur.execute("SELECT @@IDENTITY").fetchval()
                cur.execute(
                    "INSERT INTO dbo.Feedback(prediction_id,correct_flag) VALUES(?,?)",
                    last_id, 1 if fb=="Yes" else 0
                )
                conn.commit(); cur.close(); conn.close()
                st.success("✅ Feedback recorded")
    footer()

## BATCH UPLOAD ##
elif st.session_state.page == "Batch Upload":
    st.header("🗂️ Batch Upload & Predict")
    zip_file = st.file_uploader("Upload .zip of .wav files", type=["zip"])
    if zip_file:
        z    = zipfile.ZipFile(io.BytesIO(zip_file.read()))
        wavs = [f for f in z.namelist() if f.lower().endswith(".wav")]
        prog = st.progress(0); results=[]
        for i,fname in enumerate(wavs):
            feats,_ = extract_features(z.read(fname))
            sp   = sax_model.predict(feats)[0]
            stl  = style_model.predict(feats)[0]
            results.append({"file":fname,"sax_type":sp,"style":stl})
            prog.progress((i+1)/len(wavs))
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False),
                           "batch.csv","text/csv")
        st.download_button("Download JSON", df.to_json(orient="records"),
                           "batch.json","application/json")
    footer()

## VIEW HISTORY ##
elif st.session_state.page == "View History":
    st.header("📊 Your Prediction History")
    conn = get_connection()
    df = pd.read_sql(
        "SELECT prediction_id,saxophone_type,performance_style,confidence_score,predicted_at "
        "FROM dbo.Predictions WHERE user_id=? ORDER BY predicted_at DESC",
        conn, params=[st.session_state.user_id]
    )
    conn.close()
    if df.empty:
        st.info("No predictions yet.")
    else:
        st.dataframe(df, use_container_width=True)
    footer()

## EXPORT DATA ##
elif st.session_state.page == "Export Data":
    st.header("📥 Export Your Data")
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM dbo.Predictions WHERE user_id=? ORDER BY predicted_at DESC",
        conn, params=[st.session_state.user_id]
    )
    conn.close()
    if df.empty:
        st.info("No data to export.")
    else:
        st.download_button("Download CSV", df.to_csv(index=False),
                           "history.csv","text/csv")
        st.download_button("Download PDF (HTML)", df.to_html(index=False),
                           "history.html","text/html")
    footer()
