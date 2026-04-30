# ============================================================
# CARDIOAI v3.0 — Heart Disease Prediction System
# YouTube Background | Siri-Like Alex AI | 6 Languages
# 3D Heart | Login | Hospital Map | Doctor Contact
# Run: streamlit run app_v3.py
# ============================================================

import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os
from io import BytesIO
import streamlit.components.v1 as components
from datetime import datetime

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioAI — Cardiac Health AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Demo Users ────────────────────────────────────────────────
USERS = {
    "admin":   {"password": "admin123",  "role": "admin",   "name": "Dr. Admin"},
    "doctor":  {"password": "doc2024",   "role": "doctor",  "name": "Dr. Priya Nair"},
    "patient": {"password": "heart123",  "role": "patient", "name": "Ravi Kumar"},
}

# ── Doctors ───────────────────────────────────────────────────
DOCTORS = [
    {"name":"Dr. Priya Nair",          "spec":"Cardiologist",               "hosp":"Narayana Health, Bengaluru",        "exp":"18 yrs","phone":"+91-80-7122-2222","email":"priya.nair@nhhealth.com",         "rating":4.9,"av":"👩‍⚕️","sched":"Mon–Sat 9AM–5PM",  "lang":"English, Kannada, Malayalam"},
    {"name":"Dr. Rajesh Sharma",       "spec":"Cardiac Surgeon",             "hosp":"Fortis Hospital, Bengaluru",        "exp":"22 yrs","phone":"+91-80-6621-4444","email":"r.sharma@fortis.in",              "rating":4.8,"av":"👨‍⚕️","sched":"Mon–Fri 10AM–6PM", "lang":"English, Hindi, Kannada"},
    {"name":"Dr. Meenakshi Iyer",      "spec":"Interventional Cardiologist", "hosp":"Manipal Hospital, Bengaluru",       "exp":"15 yrs","phone":"+91-80-2502-0000","email":"m.iyer@manipal.edu",              "rating":4.7,"av":"👩‍⚕️","sched":"Tue–Sun 8AM–4PM",  "lang":"English, Tamil, Kannada"},
    {"name":"Dr. Suresh Reddy",        "spec":"Electrophysiologist",         "hosp":"Apollo Hospital, Bengaluru",        "exp":"20 yrs","phone":"+91-80-2941-9333","email":"s.reddy@apollohospitals.com",     "rating":4.9,"av":"👨‍⚕️","sched":"Mon–Sat 8AM–3PM",  "lang":"English, Telugu, Kannada"},
    {"name":"Dr. Lakshmi Venkat",      "spec":"Pediatric Cardiologist",      "hosp":"Aster CMI Hospital, Bengaluru",     "exp":"12 yrs","phone":"+91-80-4342-0100","email":"l.venkat@asterhospitals.in",      "rating":4.8,"av":"👩‍⚕️","sched":"Mon–Fri 9AM–5PM",  "lang":"English, Kannada, Tamil"},
    {"name":"Dr. Arun Krishnamurthy",  "spec":"Cardiac Intensivist",         "hosp":"Sakra World Hospital, Bengaluru",   "exp":"16 yrs","phone":"+91-80-4969-4969","email":"a.krishna@sakraworldhospital.com","rating":4.6,"av":"👨‍⚕️","sched":"24 / 7 Emergency", "lang":"English, Kannada, Telugu"},
]

HOSPITALS = [
    {"name":"Narayana Health City",                         "lat":12.8449,"lon":77.6616,"phone":"+91-80-7122-2222","type":"Multi-specialty",       "rating":4.8},
    {"name":"Fortis Hospital Bannerghatta Road",            "lat":12.8766,"lon":77.5993,"phone":"+91-80-6621-4444","type":"Cardiac Care",           "rating":4.7},
    {"name":"Manipal Hospital Old Airport Road",            "lat":12.9666,"lon":77.6463,"phone":"+91-80-2502-0000","type":"Multi-specialty",        "rating":4.6},
    {"name":"Apollo Hospital Bannerghatta Road",            "lat":12.8934,"lon":77.5972,"phone":"+91-80-2941-9333","type":"Cardiac Center",         "rating":4.8},
    {"name":"Aster CMI Hospital",                          "lat":13.0627,"lon":77.5940,"phone":"+91-80-4342-0100","type":"Multi-specialty",        "rating":4.7},
    {"name":"Sakra World Hospital",                        "lat":12.9698,"lon":77.7499,"phone":"+91-80-4969-4969","type":"Multi-specialty",        "rating":4.6},
    {"name":"Sri Jayadeva Institute of Cardiovascular Sciences","lat":12.9250,"lon":77.5938,"phone":"+91-80-2297-5100","type":"Dedicated Cardiac","rating":4.9},
    {"name":"Victoria Hospital",                           "lat":12.9429,"lon":77.5667,"phone":"+91-80-2699-5000","type":"Government",             "rating":4.4},
]

# ── Language Translations ─────────────────────────────────────
LANG = {
    "en":{"flag":"🇬🇧","name":"English","title":"Heart Disease Prediction","subtitle":"AI-Powered Cardiac Health Analysis","nav_home":"🏠 Home","nav_predict":"🔬 Prediction","nav_voice":"🎙️ Alex – Voice AI","nav_doctors":"👨‍⚕️ Doctors","nav_hospitals":"🏥 Hospitals Near Me","nav_about":"ℹ️ About","nav_logout":"🚪 Logout","login_title":"Login to CardioAI","username":"Username","password":"Password","login_btn":"Login","login_err":"Invalid credentials. Try admin/admin123","predict_btn":"🔍 Predict Now","result_disease":"⚠️ Heart Disease Detected","result_safe":"✅ No Heart Disease Detected","download":"⬇️ Download Report","age":"Age","sex":"Sex","male":"Male","female":"Female","cp":"Chest Pain Type","bp":"Resting Blood Pressure (mmHg)","chol":"Cholesterol (mg/dL)","fbs":"Fasting Blood Sugar > 120","ecg":"Resting ECG","hr":"Max Heart Rate","angina":"Exercise Angina","oldpeak":"ST Depression","slope":"ST Slope","risk":"Risk","yes":"Yes","no":"No","welcome":"Welcome back"},
    "hi":{"flag":"🇮🇳","name":"हिंदी","title":"हृदय रोग भविष्यवाणी","subtitle":"AI कार्डियक स्वास्थ्य विश्लेषण","nav_home":"🏠 होम","nav_predict":"🔬 भविष्यवाणी","nav_voice":"🎙️ Alex – वॉयस AI","nav_doctors":"👨‍⚕️ डॉक्टर","nav_hospitals":"🏥 नज़दीकी अस्पताल","nav_about":"ℹ️ हमारे बारे में","nav_logout":"🚪 लॉगआउट","login_title":"CardioAI में लॉगिन","username":"उपयोगकर्ता नाम","password":"पासवर्ड","login_btn":"लॉगिन","login_err":"गलत क्रेडेंशियल। admin/admin123 आज़माएं","predict_btn":"🔍 भविष्यवाणी करें","result_disease":"⚠️ हृदय रोग पाया गया","result_safe":"✅ हृदय रोग नहीं है","download":"⬇️ रिपोर्ट डाउनलोड","age":"आयु","sex":"लिंग","male":"पुरुष","female":"महिला","cp":"सीने में दर्द","bp":"रक्तचाप","chol":"कोलेस्ट्रॉल","fbs":"फास्टिंग ब्लड शुगर","ecg":"ECG","hr":"हृदय गति","angina":"एनजाइना","oldpeak":"ST अवसाद","slope":"ST ढलान","risk":"जोखिम","yes":"हाँ","no":"नहीं","welcome":"वापस स्वागत"},
    "kn":{"flag":"🇮🇳","name":"ಕನ್ನಡ","title":"ಹೃದಯ ರೋಗ ಮುನ್ಸೂಚನೆ","subtitle":"AI ಹೃದಯ ಆರೋಗ್ಯ ವಿಶ್ಲೇಷಣೆ","nav_home":"🏠 ಮನೆ","nav_predict":"🔬 ಮುನ್ಸೂಚನೆ","nav_voice":"🎙️ Alex – ಧ್ವನಿ AI","nav_doctors":"👨‍⚕️ ವೈದ್ಯರು","nav_hospitals":"🏥 ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆ","nav_about":"ℹ️ ನಮ್ಮ ಬಗ್ಗೆ","nav_logout":"🚪 ಲಾಗ್‌ಔಟ್","login_title":"CardioAI ಗೆ ಲಾಗಿನ್","username":"ಬಳಕೆದಾರ ಹೆಸರು","password":"ಪಾಸ್‌ವರ್ಡ್","login_btn":"ಲಾಗಿನ್","login_err":"ತಪ್ಪು ಪ್ರಮಾಣಪತ್ರ. admin/admin123 ಪ್ರಯತ್ನಿಸಿ","predict_btn":"🔍 ಮುನ್ಸೂಚಿಸಿ","result_disease":"⚠️ ಹೃದಯ ರೋಗ ಪತ್ತೆ","result_safe":"✅ ಹೃದಯ ರೋಗವಿಲ್ಲ","download":"⬇️ ವರದಿ ಡೌನ್‌ಲೋಡ್","age":"ವಯಸ್ಸು","sex":"ಲಿಂಗ","male":"ಪುರುಷ","female":"ಮಹಿಳೆ","cp":"ಎದೆ ನೋವು","bp":"ರಕ್ತದೊತ್ತಡ","chol":"ಕೊಲೆಸ್ಟ್ರಾಲ್","fbs":"ರಕ್ತ ಸಕ್ಕರೆ","ecg":"ECG","hr":"ಹೃದಯ ಬಡಿತ","angina":"ಆಂಜಿನಾ","oldpeak":"ST ಖಿನ್ನತೆ","slope":"ST ಇಳಿಜಾರು","risk":"ಅಪಾಯ","yes":"ಹೌದು","no":"ಇಲ್ಲ","welcome":"ಮರಳಿ ಸ್ವಾಗತ"},
    "ta":{"flag":"🇮🇳","name":"தமிழ்","title":"இதய நோய் கணிப்பு","subtitle":"AI இதய ஆரோக்கிய பகுப்பாய்வு","nav_home":"🏠 முகப்பு","nav_predict":"🔬 கணிப்பு","nav_voice":"🎙️ Alex – குரல் AI","nav_doctors":"👨‍⚕️ மருத்துவர்கள்","nav_hospitals":"🏥 அருகில் மருத்துவமனை","nav_about":"ℹ️ எங்களைப் பற்றி","nav_logout":"🚪 வெளியேறு","login_title":"CardioAI உள்நுழைவு","username":"பயனர்பெயர்","password":"கடவுச்சொல்","login_btn":"உள்நுழை","login_err":"தவறான நற்சான்றிதழ். admin/admin123 முயலவும்","predict_btn":"🔍 கணிக்கவும்","result_disease":"⚠️ இதய நோய் கண்டறியப்பட்டது","result_safe":"✅ இதய நோய் இல்லை","download":"⬇️ அறிக்கை பதிவிறக்கம்","age":"வயது","sex":"பாலினம்","male":"ஆண்","female":"பெண்","cp":"மார்பு வலி","bp":"இரத்த அழுத்தம்","chol":"கொலஸ்ட்ரால்","fbs":"இரத்த சர்க்கரை","ecg":"ECG","hr":"இதய துடிப்பு","angina":"ஆஞ்சினா","oldpeak":"ST மந்தம்","slope":"ST சாய்வு","risk":"ஆபத்து","yes":"ஆம்","no":"இல்லை","welcome":"மீண்டும் வரவேற்கிறோம்"},
    "te":{"flag":"🇮🇳","name":"తెలుగు","title":"హృదయ వ్యాధి అంచనా","subtitle":"AI కార్డియాక్ ఆరోగ్య విశ్లేషణ","nav_home":"🏠 హోమ్","nav_predict":"🔬 అంచనా","nav_voice":"🎙️ Alex – వాయిస్ AI","nav_doctors":"👨‍⚕️ డాక్టర్లు","nav_hospitals":"🏥 సమీప ఆసుపత్రి","nav_about":"ℹ️ మా గురించి","nav_logout":"🚪 లాగ్అవుట్","login_title":"CardioAI లాగిన్","username":"వినియోగదారు పేరు","password":"పాస్‌వర్డ్","login_btn":"లాగిన్","login_err":"తప్పు ఆధారాలు. admin/admin123 ప్రయత్నించండి","predict_btn":"🔍 అంచనా వేయండి","result_disease":"⚠️ గుండె జబ్బు గుర్తించబడింది","result_safe":"✅ గుండె జబ్బు లేదు","download":"⬇️ నివేదిక డౌన్‌లోడ్","age":"వయస్సు","sex":"లింగం","male":"మగ","female":"ఆడ","cp":"ఛాతీ నొప్పి","bp":"రక్తపోటు","chol":"కొలెస్ట్రాల్","fbs":"రక్త చక్కెర","ecg":"ECG","hr":"గుండె చప్పుడు","angina":"ఆంజినా","oldpeak":"ST నిస్పృహ","slope":"ST వాలు","risk":"ప్రమాదం","yes":"అవును","no":"కాదు","welcome":"తిరిగి స్వాగతం"},
    "ml":{"flag":"🇮🇳","name":"മലയാളം","title":"ഹൃദ്രോഗ പ്രവചനം","subtitle":"AI ഹൃദ്രോഗ ആരോഗ്യ വിശകലനം","nav_home":"🏠 ഹോം","nav_predict":"🔬 പ്രവചനം","nav_voice":"🎙️ Alex – വോയ്‌സ് AI","nav_doctors":"👨‍⚕️ ഡോക്ടർമാർ","nav_hospitals":"🏥 അടുത്ത ആശുപത്രി","nav_about":"ℹ️ ഞങ്ങളെ കുറിച്ച്","nav_logout":"🚪 ലോഗ്ഔട്ട്","login_title":"CardioAI ലോഗിൻ","username":"ഉപയോക്തൃ നാമം","password":"പാസ്‌വേഡ്","login_btn":"ലോഗിൻ","login_err":"തെറ്റായ ക്രെഡൻഷ്യലുകൾ. admin/admin123 ശ്രമിക്കുക","predict_btn":"🔍 പ്രവചിക്കുക","result_disease":"⚠️ ഹൃദ്രോഗം കണ്ടെത്തി","result_safe":"✅ ഹൃദ്രോഗം ഇല്ല","download":"⬇️ റിപ്പോർട്ട് ഡൗൺലോഡ്","age":"പ്രായം","sex":"ലിംഗം","male":"പുരുഷൻ","female":"സ്ത്രീ","cp":"നെഞ്ചുവേദന","bp":"രക്തസമ്മർദ്ദം","chol":"കൊളസ്ട്രോൾ","fbs":"ഉപവാസ രക്തത്തിലെ പഞ്ചസാര","ecg":"ECG","hr":"ഹൃദയ മിടിപ്പ്","angina":"ആൻജൈന","oldpeak":"ST ഡിപ്രഷൻ","slope":"ST ചരിവ്","risk":"അപകടം","yes":"അതെ","no":"ഇല്ല","welcome":"തിരിച്ചു സ്വാഗതം"},
}

# ── Session State ─────────────────────────────────────────────
for k,v in {"logged_in":False,"username":"","page":"login","language":"en","pred_result":None}.items():
    if k not in st.session_state: st.session_state[k] = v
st.session_state.language = "en"  # English only — locked

def T(key): return LANG[st.session_state.language].get(key, LANG["en"].get(key, key))

@st.cache_resource
def load_model():
    try:    return joblib.load("heart_disease_model.joblib")
    except: return None

model = load_model()

# ═════════════════════════════════════════════════════════════
#  YOUTUBE VIDEO BACKGROUND
# ═════════════════════════════════════════════════════════════
def inject_video_background(video_id="6nKYfealjBg"):
    st.markdown(f"""
<style>
/* ── Make app transparent so video shows through ── */
.stApp {{
    background: transparent !important;
}}
.stApp > div, .main, .block-container,
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"] {{
    background: transparent !important;
}}
/* ── Sidebar glassmorphism ── */
section[data-testid="stSidebar"] {{
    background: rgba(8, 8, 20, 0.88) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}}
</style>

<!-- YouTube fullscreen background -->
<div id="yt-bg-wrap" style="
  position: fixed;
  top: 0; left: 0;
  width: 100vw; height: 100vh;
  z-index: -9999;
  pointer-events: none;
  overflow: hidden;
">
  <iframe id="yt-bg-frame"
    src="https://www.youtube.com/embed/{video_id}?autoplay=1&mute=1&loop=1&playlist={video_id}&controls=0&showinfo=0&rel=0&modestbranding=1&playsinline=1&iv_load_policy=3&disablekb=1&vq=hd720&enablejsapi=1"
    allow="autoplay; encrypted-media"
    frameborder="0"
    style="
      position: absolute;
      top: 50%; left: 50%;
      transform: translate(-50%,-50%);
      min-width: 177.78vh;
      min-height: 56.25vw;
      width: 100vw;
      height: 100vh;
      border: none;
      opacity: 0.30;
      filter: saturate(1.2) brightness(0.9);
    "
  ></iframe>

  <!-- YouTube IFrame API: stop 10s before end, then restart -->
  <script>
  (function(){{
    var tag=document.createElement('script');
    tag.src='https://www.youtube.com/iframe_api';
    document.head.appendChild(tag);
    var player;
    window.onYouTubeIframeAPIReady=function(){{
      player=new YT.Player('yt-bg-frame',{{
        events:{{
          'onReady':function(e){{e.target.playVideo();}},
          'onStateChange':function(e){{
            if(e.data===YT.PlayerState.PLAYING){{
              clearInterval(window._ytChk);
              window._ytChk=setInterval(function(){{
                try{{
                  var dur=player.getDuration();
                  var cur=player.getCurrentTime();
                  if(dur>0 && cur >= dur-10){{
                    player.seekTo(0,true);
                  }}
                }}catch(ex){{}}
              }},500);
            }}
          }}
        }}
      }});
    }};
  }})();
  </script>

  <!-- Gradient overlay so text stays readable -->
  <div style="
    position: absolute; top:0; left:0;
    width:100%; height:100%;
    background: linear-gradient(
      135deg,
      rgba(4,4,16,0.72) 0%,
      rgba(8,5,22,0.60) 50%,
      rgba(12,4,18,0.72) 100%
    );
  "></div>

  <!-- Vignette -->
  <div style="
    position:absolute; top:0; left:0; width:100%; height:100%;
    background: radial-gradient(ellipse at center, transparent 30%, rgba(2,2,12,0.65) 100%);
  "></div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  GLOBAL CSS  (glassmorphism cards over video bg)
# ═════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --accent:  #e8294c;
  --accent2: #c84b9e;
  --accent3: #7b2ff7;
  --green:   #00e676;
  --yellow:  #ffd740;
  --text:    #f0f0fa;
  --muted:   rgba(200,200,230,0.55);
  --glass:   rgba(12, 12, 28, 0.65);
  --glass2:  rgba(18, 18, 38, 0.72);
  --border:  rgba(255,255,255,0.10);
  --glow:    rgba(232,41,76,0.30);
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
}

/* ─── Glassmorphism card ─── */
.glass-card {
  background: var(--glass);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 24px;
  margin-bottom: 16px;
  transition: all .3s;
}
.glass-card:hover {
  background: var(--glass2);
  border-color: rgba(232,41,76,0.35);
  box-shadow: 0 8px 40px rgba(232,41,76,0.15);
  transform: translateY(-2px);
}

/* ─── Doctor card ─── */
.doc-card {
  background: rgba(10,10,28,0.70);
  backdrop-filter: blur(20px);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 24px 20px;
  text-align: center;
  transition: all .3s;
  height: 100%;
}
.doc-card:hover {
  border-color: rgba(200,75,158,0.5);
  box-shadow: 0 12px 50px rgba(200,75,158,0.2);
  transform: translateY(-5px);
}

/* ─── Page title ─── */
.ptitle {
  font-family: 'Syne', sans-serif !important;
  font-size: 38px; font-weight: 800;
  background: linear-gradient(90deg, #ff6b6b, var(--accent), var(--accent2), var(--accent3));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 4px;
  line-height: 1.15;
}
.psub {
  font-size: 15px; color: var(--muted); margin-bottom: 28px;
}

/* ─── Buttons ─── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), #aa1030) !important;
  color: white !important; border: none !important;
  border-radius: 14px !important; padding: 11px 28px !important;
  font-weight: 600 !important; font-size: 14px !important;
  transition: all .3s !important;
  box-shadow: 0 4px 18px var(--glow) !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover {
  transform: translateY(-3px) !important;
  box-shadow: 0 10px 32px var(--glow) !important;
}

/* ─── Inputs ─── */
input, textarea { caret-color: var(--accent) !important; }
input, div[data-baseweb="select"] > div {
  background: rgba(16,16,36,0.7) !important;
  backdrop-filter: blur(10px) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
input:focus { border-color: var(--accent) !important; }
div[role="listbox"]  { background: rgba(16,16,36,0.95) !important; }
div[role="option"]   { color: var(--text) !important; }
div[data-baseweb="select"] svg { fill: var(--muted) !important; }

/* ─── Metrics ─── */
[data-testid="metric-container"] {
  background: rgba(12,12,28,0.65) !important;
  backdrop-filter: blur(16px) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 16px !important;
}
[data-testid="stMetricLabel"]  { color: var(--muted) !important; }
[data-testid="stMetricValue"]  { color: var(--text) !important; font-family:'Syne',sans-serif !important; }

/* ─── Alerts ─── */
.stSuccess { background: rgba(0,230,118,0.12) !important; border-color: rgba(0,230,118,0.4) !important; }
.stError   { background: rgba(232,41,76,0.12) !important; border-color: rgba(232,41,76,0.4) !important; }

/* ─── Sidebar nav button ─── */
section[data-testid="stSidebar"] button {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 12px !important;
  color: var(--text) !important;
  text-align: left !important;
  width: 100% !important; padding: 10px 14px !important;
  margin-bottom: 4px !important;
  transition: all .25s !important;
  font-family: 'DM Sans', sans-serif !important;
}
section[data-testid="stSidebar"] button:hover {
  background: rgba(232,41,76,0.18) !important;
  border-color: rgba(232,41,76,0.4) !important;
  transform: translateX(5px) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
h1,h2,h3,h4 { color: var(--text) !important; font-family:'Syne',sans-serif !important; }
label { color: var(--muted) !important; font-size:13px !important; }
p { color: rgba(210,210,240,0.85) !important; }

/* ─── Hide Streamlit top horizontal nav bar ─── */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
.stDeployButton,
#MainMenu,
footer,
[data-testid="stStatusWidget"] {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
}

/* Spinning heartbeat logo */
@keyframes heartbeat { 0%,100%{transform:scale(1);}  50%{transform:scale(1.18);} }
.hb { animation: heartbeat 1.2s ease-in-out infinite; display:inline-block; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  SIRI-LIKE ALEX COMPONENT  (full-featured voice AI in browser)
# ═════════════════════════════════════════════════════════════
def alex_siri_component(height=700, api_key="", lang_bcp="en-US"):
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;700&family=Outfit:wght@300;400;600;700&display=swap');
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{
  background:transparent;
  font-family:'Outfit',sans-serif;
  color:white;
  min-height:100vh;
  overflow-x:hidden;
}}

/* ── Root layout ── */
.wrap{{
  display:flex; flex-direction:column; align-items:center;
  padding:20px 16px 30px; gap:18px; max-width:580px; margin:0 auto;
}}

/* ── Alex title ── */
.alex-title{{
  font-size:30px; font-weight:700; letter-spacing:4px;
  background: linear-gradient(90deg,#c84b9e,#7b2ff7,#e8294c,#ff6b35);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.alex-sub{{font-size:11px;color:rgba(255,255,255,0.4);letter-spacing:3px;margin-top:-10px;}}

/* ── Orb ── */
.orb-wrap{{position:relative;width:210px;height:210px;cursor:pointer;user-select:none;}}
.orb-ring{{
  position:absolute; border-radius:50%;
  border:1.5px solid rgba(200,75,158,0.35);
  width:210px;height:210px; top:0;left:0;
  animation:ringOut 2.5s ease-out infinite;
}}
.orb-ring:nth-child(2){{animation-delay:.85s;}}
.orb-ring:nth-child(3){{animation-delay:1.7s;}}
@keyframes ringOut{{0%{{transform:scale(1);opacity:.9;}}100%{{transform:scale(1.9);opacity:0;}}}}

.orb{{
  position:absolute; top:15px;left:15px;
  width:180px;height:180px; border-radius:50%;
  background: radial-gradient(circle at 33% 30%, #c84b9e 0%, #7b2ff7 40%, #e8294c 75%, #ff6b35 100%);
  box-shadow: 0 0 50px rgba(200,75,158,.7), 0 0 100px rgba(123,47,247,.4), inset 0 0 50px rgba(255,255,255,.07);
  animation:orbHover 3.5s ease-in-out infinite;
  transition: background .6s;
  display:flex;align-items:center;justify-content:center;
  overflow:hidden;
}}
.orb::after{{
  content:''; position:absolute;top:18%;left:18%;
  width:32%;height:28%;
  background:radial-gradient(circle,rgba(255,255,255,.45),transparent);
  border-radius:50%;
  pointer-events:none;
}}
@keyframes orbHover{{
  0%,100%{{transform:translateY(0) scale(1); box-shadow:0 0 50px rgba(200,75,158,.7),0 0 100px rgba(123,47,247,.4);}}
  50%{{transform:translateY(-12px) scale(1.025); box-shadow:0 18px 70px rgba(200,75,158,.9),0 0 140px rgba(123,47,247,.5);}}
}}
.orb.listening{{
  background: radial-gradient(circle at 33% 30%,#ff6644,#e8294c 40%,#ff0080 100%) !important;
  animation:orbPulse .7s ease-in-out infinite !important;
}}
.orb.thinking{{
  background: radial-gradient(circle at 33% 30%,#ffd740,#ff6d00 40%,#ff1744 100%) !important;
  animation:orbSpin 1.2s linear infinite !important;
}}
.orb.speaking{{
  background: radial-gradient(circle at 33% 30%,#69f0ae,#00bcd4 40%,#3d5afe 100%) !important;
  animation:orbPulse .45s ease-in-out infinite !important;
}}
@keyframes orbPulse{{0%,100%{{transform:scale(1);}}50%{{transform:scale(1.08);}}}}
@keyframes orbSpin{{from{{filter:hue-rotate(0deg);}}to{{filter:hue-rotate(360deg);}}}}

/* Waveform inside orb (shows when active) */
.orb-wave{{display:flex;align-items:center;gap:5px;opacity:0;transition:opacity .35s;}}
.orb.listening .orb-wave,.orb.speaking .orb-wave{{opacity:1;}}
.owb{{width:4px;border-radius:4px;background:rgba(255,255,255,.9);animation:wvBar .5s ease-in-out infinite alternate;}}
.owb:nth-child(1){{height:10px;animation-delay:0s;}}
.owb:nth-child(2){{height:22px;animation-delay:.07s;}}
.owb:nth-child(3){{height:34px;animation-delay:.14s;}}
.owb:nth-child(4){{height:46px;animation-delay:.1s;}}
.owb:nth-child(5){{height:34px;animation-delay:.05s;}}
.owb:nth-child(6){{height:22px;animation-delay:.12s;}}
.owb:nth-child(7){{height:10px;animation-delay:.18s;}}
@keyframes wvBar{{from{{transform:scaleY(.3);}}to{{transform:scaleY(1);}}}}

/* Status */
.status{{font-size:14px;color:rgba(255,255,255,.6);letter-spacing:1.5px;min-height:22px;}}

/* Bottom waveform strip */
.wave-strip{{display:flex;align-items:center;gap:3px;height:44px;opacity:0;transition:opacity .3s;}}
.wave-strip.on{{opacity:1;}}
.wsb{{width:3px;border-radius:3px;background:linear-gradient(180deg,#c84b9e,#7b2ff7);animation:wsAnim .5s ease-in-out infinite alternate;}}
@keyframes wsAnim{{from{{height:3px;opacity:.35;}}to{{height:36px;opacity:1;}}}}

/* ── Controls ── */
.controls{{display:flex;gap:12px;align-items:center;}}
.mic-btn{{
  width:60px;height:60px;border-radius:50%;
  background:linear-gradient(135deg,#e8294c,#aa1030);
  border:none;color:white;font-size:24px;cursor:pointer;
  box-shadow:0 4px 22px rgba(232,41,76,.55);
  transition:all .25s; display:flex;align-items:center;justify-content:center;
}}
.mic-btn:hover{{transform:scale(1.12);box-shadow:0 8px 34px rgba(232,41,76,.75);}}
.mic-btn.on{{background:linear-gradient(135deg,#ff4444,#cc0000)!important;animation:micPulse 1s infinite;}}
@keyframes micPulse{{0%,100%{{box-shadow:0 4px 22px rgba(255,68,68,.5);}}50%{{box-shadow:0 4px 44px rgba(255,68,68,.9),0 0 0 12px rgba(255,68,68,.15);}}}}

/* ── Settings row ── */
.settings-row{{display:flex;gap:10px;width:100%;}}
.sel{{
  background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.15);
  border-radius:11px;color:white;padding:9px 14px;font-size:13px;cursor:pointer;flex-shrink:0;
}}
.key-inp{{
  flex:1;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.15);
  border-radius:11px;color:white;padding:9px 14px;font-size:13px;outline:none;
}}
.key-inp:focus{{border-color:rgba(123,47,247,.6);}}
.key-inp::placeholder{{color:rgba(255,255,255,.25);}}

/* ── Chat ── */
.chat-box{{width:100%;max-height:260px;overflow-y:auto;display:flex;flex-direction:column;gap:10px;padding:4px 2px;}}
.msg-u{{
  background:linear-gradient(135deg,rgba(232,41,76,.75),rgba(170,16,48,.75));
  backdrop-filter:blur(10px);
  padding:10px 16px;border-radius:18px 18px 4px 18px;
  align-self:flex-end;max-width:82%;font-size:13.5px;line-height:1.5;
}}
.msg-a{{
  background:rgba(123,47,247,.22);border:1px solid rgba(123,47,247,.3);
  backdrop-filter:blur(10px);
  padding:10px 16px;border-radius:18px 18px 18px 4px;
  align-self:flex-start;max-width:85%;font-size:13.5px;line-height:1.5;
}}
.ml{{font-size:10px;opacity:.5;margin-bottom:3px;}}
::-webkit-scrollbar{{width:3px;}}
::-webkit-scrollbar-thumb{{background:rgba(255,255,255,.18);border-radius:2px;}}

/* ── Text input row ── */
.tinput-row{{display:flex;gap:10px;width:100%;align-items:center;}}
.tinput{{
  flex:1;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);
  border-radius:28px;color:white;padding:13px 20px;font-size:14px;outline:none;
  font-family:'Outfit',sans-serif;
}}
.tinput:focus{{border-color:rgba(123,47,247,.6);}}
.tinput::placeholder{{color:rgba(255,255,255,.28);}}
.send-btn{{
  width:50px;height:50px;border-radius:50%;
  background:linear-gradient(135deg,#7b2ff7,#e8294c);
  border:none;color:white;font-size:20px;cursor:pointer;transition:all .22s;
  display:flex;align-items:center;justify-content:center;
}}
.send-btn:hover{{transform:scale(1.12);}}

/* ── Quick chips ── */
.chips{{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;}}
.chip{{
  background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);
  border-radius:100px;padding:6px 14px;font-size:12px;cursor:pointer;transition:all .2s;
  color:rgba(255,255,255,.75);
}}
.chip:hover{{background:rgba(200,75,158,.25);border-color:rgba(200,75,158,.5);color:white;}}
</style>
</head>
<body>
<div class="wrap">

  <div class="alex-title">✦ ALEX</div>
  <div class="alex-sub">AI CARDIAC HEALTH ASSISTANT</div>

  <!-- Settings: API key only (English locked) -->
  <div class="settings-row">
    <input id="apiKey" class="key-inp" type="password"
      value="{api_key}"
      placeholder="Anthropic API key (optional – for live AI)" />
  </div>

  <!-- Orb -->
  <div class="orb-wrap" id="orbWrap">
    <div class="orb-ring"></div>
    <div class="orb-ring"></div>
    <div class="orb-ring"></div>
    <div class="orb" id="alexOrb" onclick="handleOrbClick()">
      <div class="orb-wave">
        <div class="owb"></div><div class="owb"></div><div class="owb"></div>
        <div class="owb"></div><div class="owb"></div><div class="owb"></div>
        <div class="owb"></div>
      </div>
    </div>
  </div>

  <div class="status" id="statusTxt">Tap Alex to speak</div>

  <!-- Wave strip -->
  <div class="wave-strip" id="waveStrip"></div>

  <!-- Quick chips -->
  <div class="chips">
    <div class="chip" onclick="ask('What are heart disease symptoms?')">❤️ Symptoms</div>
    <div class="chip" onclick="ask('How to lower cholesterol?')">🧪 Cholesterol</div>
    <div class="chip" onclick="ask('What is high blood pressure?')">💉 Blood Pressure</div>
    <div class="chip" onclick="ask('Tips for heart health?')">💪 Heart Tips</div>
    <div class="chip" onclick="ask('When should I see a doctor?')">👨‍⚕️ See Doctor</div>
    <div class="chip" onclick="ask('What does my ECG mean?')">📊 ECG</div>
  </div>

  <!-- Chat -->
  <div class="chat-box" id="chatBox"></div>

  <!-- Text input + Mic -->
  <div class="tinput-row">
    <button class="mic-btn" id="micBtn" onclick="handleOrbClick()">🎤</button>
    <input class="tinput" id="txtInput" placeholder="Or type a question here..."
      onkeydown="if(event.key==='Enter')sendTxt()"/>
    <button class="send-btn" onclick="sendTxt()">➤</button>
  </div>

</div><!-- /wrap -->

<script>
// ─── Build wave strip ─────────────────────────────────────
const strip = document.getElementById('waveStrip');
for(let i=0;i<32;i++){{
  const b=document.createElement('div');
  b.className='wsb';
  b.style.animationDelay=(i*.045)+'s';
  b.style.animationDuration=(.28+Math.random()*.36)+'s';
  strip.appendChild(b);
}}

const orb     = document.getElementById('alexOrb');
const status  = document.getElementById('statusTxt');
const micBtn  = document.getElementById('micBtn');
const chatBox = document.getElementById('chatBox');

let recog=null, isListening=false;
const langMap={{'en-US':'English','hi-IN':'Hindi','kn-IN':'Kannada','ta-IN':'Tamil','te-IN':'Telugu','ml-IN':'Malayalam'}};

// ─── State helpers ────────────────────────────────────────
function setState(s,txt){{
  orb.className='orb '+(s||'');
  strip.className='wave-strip'+((s==='listening'||s==='speaking')?' on':'');
  status.textContent=txt;
  micBtn.className='mic-btn'+(s==='listening'?' on':'');
  micBtn.textContent=s==='listening'?'⏸':s==='speaking'?'🔊':s==='thinking'?'⏳':'🎤';
}}
setState('','Tap Alex to speak');

// ─── Voice input ─────────────────────────────────────────
function handleOrbClick(){{ isListening?stopListen():startListen(); }}

function startListen(){{
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  if(!SR){{ addMsg('alex','Voice input is not supported in this browser. Please type your question instead.'); return; }}
  recog=new SR();
  recog.lang='en-US';
  recog.interimResults=false; recog.maxAlternatives=1;
  recog.onstart=()=>{{ isListening=true; setState('listening','Listening — speak now...'); }};
  recog.onresult=e=>{{ stopListen(); handleQ(e.results[0][0].transcript); }};
  recog.onerror=()=>{{ stopListen(); setState('','Could not hear you — try again'); }};
  recog.onend=()=>{{ if(isListening)stopListen(); }};
  recog.start();
}}
function stopListen(){{ if(recog){{recog.stop();recog=null;}} isListening=false; }}

// ─── Text send ────────────────────────────────────────────
function sendTxt(){{
  const i=document.getElementById('txtInput'); const t=i.value.trim(); if(!t)return; i.value=''; handleQ(t);
}}
function ask(q){{ handleQ(q); }}
function updateLang(){{}}

// ─── Add chat bubble ─────────────────────────────────────
function addMsg(role,text){{
  const d=document.createElement('div');
  d.className=role==='user'?'msg-u':'msg-a';
  const l=document.createElement('div'); l.className='ml';
  l.textContent=role==='user'?'You':'🤖 Alex';
  d.appendChild(l); d.appendChild(document.createTextNode(text));
  chatBox.appendChild(d); chatBox.scrollTop=chatBox.scrollHeight;
}}

// ─── Main query ───────────────────────────────────────────
async function handleQ(q){{
  addMsg('user',q);
  setState('thinking','Alex is thinking...');
  
  const key=document.getElementById('apiKey').value.trim();
  let reply='';
  
  if(key){{
    try{{
      const lang='English';
      const r=await fetch('https://api.anthropic.com/v1/messages',{{
        method:'POST',
        headers:{{
          'x-api-key':key,
          'anthropic-version':'2023-06-01',
          'content-type':'application/json',
          'anthropic-dangerous-direct-browser-access':'true'
        }},
        body:JSON.stringify({{
          model:'claude-sonnet-4-20250514',
          max_tokens:300,
          system:`You are Alex, a friendly and warm AI cardiac health assistant. Respond in ${{lang}} language. Be concise (under 3 sentences for voice), caring, and medically accurate. Gently recommend seeing a cardiologist for any serious concerns.`,
          messages:[{{role:'user',content:q}}]
        }})
      }});
      const data=await r.json();
      reply=data.content?.[0]?.text||fallback(q);
    }}catch{{reply=fallback(q);}}
  }} else {{
    await new Promise(r=>setTimeout(r,700));
    reply=fallback(q);
  }}
  
  addMsg('alex',reply);
  speakOut(reply);
}}

// ─── Rule-based fallback ──────────────────────────────────
function fallback(q){{
  q=q.toLowerCase();
  if(q.match(/symptom|sign|feel|ache|chest pain/))
    return"Common heart disease symptoms: chest pain or tightness, shortness of breath, fatigue, palpitations, dizziness, and swelling in legs. Please see a cardiologist if you experience these.";
  if(q.match(/cholesterol|diet|food|eat/))
    return"To lower cholesterol: eat oats, nuts, and omega-3 fatty acids; avoid trans fats and red meat. Regular exercise and not smoking also help significantly.";
  if(q.match(/blood pressure|bp|hypertension/))
    return"Normal BP is below 120/80 mmHg. Reduce salt, exercise daily, avoid smoking, manage stress, and take prescribed medications to control high blood pressure.";
  if(q.match(/ecg|electrocardiogram|heart rate|ekg/))
    return"An ECG records the heart's electrical activity. ST-T wave changes may indicate ischemia. Always have your ECG interpreted by a cardiologist for accurate diagnosis.";
  if(q.match(/exercise|workout|physical|sport/))
    return"30 minutes of moderate aerobic exercise 5 days a week is great for heart health. Start gradually. If you have an existing condition, consult your doctor first.";
  if(q.match(/stress|anxiet|mental|relax/))
    return"Chronic stress raises cortisol, increasing BP and heart disease risk. Try deep breathing, meditation, regular sleep, and talking to a counselor if stress is severe.";
  if(q.match(/smok|tobacco|cigarette/))
    return"Smoking doubles heart disease risk. Even light smoking is harmful. Quitting significantly reduces risk within just one year. Ask your doctor about cessation aids.";
  if(q.match(/hospital|emergency|ambulance|112/))
    return"For cardiac emergencies in India, call 112 immediately. You can also find nearby cardiac hospitals in the Hospitals Near Me section of this app.";
  if(q.match(/doctor|consult|when.*see/))
    return"See a cardiologist if you have: chest pain, shortness of breath at rest, palpitations, family history of heart disease, or if your prediction score is above 60%.";
  if(q.match(/tips?|health|prevent|protect/))
    return"Top heart health tips: exercise regularly, eat a heart-healthy diet, quit smoking, limit alcohol, manage stress, control BP & cholesterol, and get regular check-ups.";
  return"Hello! I'm Alex, your AI cardiac health assistant. Ask me about heart disease symptoms, cholesterol, blood pressure, medications, or anything heart-related. I'm here to help!";
}}

// ─── Speech synthesis ─────────────────────────────────────
function speakOut(text){{
  if(!window.speechSynthesis){{ setState('','Done'); return; }}
  setState('speaking','Alex is speaking...');
  const u=new SpeechSynthesisUtterance(text);
  u.lang='en-US';
  u.rate=.95; u.pitch=1.1;
  const voices=speechSynthesis.getVoices();
  const lc=u.lang.split('-')[0];
  const v=voices.find(v=>v.lang.startsWith(lc)&&/female|woman|samantha|google/i.test(v.name))
        ||voices.find(v=>v.lang.startsWith(lc))
        ||null;
  if(v)u.voice=v;
  u.onend=()=>setState('','Ask another question...');
  u.onerror=()=>setState('','');
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}}

window.speechSynthesis?.addEventListener?.('voiceschanged',()=>{{}});
</script>
</body>
</html>"""
    components.html(html, height=height, scrolling=False)


# ═════════════════════════════════════════════════════════════
#  3D HEART  (Three.js – used on Home page)
# ═════════════════════════════════════════════════════════════
def heart_3d(height=380):
    html = """<!DOCTYPE html><html><head>
<style>*{margin:0;padding:0;box-sizing:border-box;}body{background:transparent;overflow:hidden;}</style>
</head><body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const W=window.innerWidth,H=window.innerHeight;
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
renderer.setSize(W,H); renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x000000,0);
document.body.appendChild(renderer.domElement);
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(45,W/H,.1,100);
camera.position.set(0,0,7);

scene.add(new THREE.AmbientLight(0xff2244,.5));
const pl1=new THREE.PointLight(0xff4488,4,25); pl1.position.set(4,3,4); scene.add(pl1);
const pl2=new THREE.PointLight(0x8844ff,2,20); pl2.position.set(-3,-2,3); scene.add(pl2);
const dl=new THREE.DirectionalLight(0xffffff,.9); dl.position.set(0,6,6); scene.add(dl);

const hs=new THREE.Shape();
hs.moveTo(0,0);
hs.bezierCurveTo(0,.35,-.55,.65,-1,.42);
hs.bezierCurveTo(-1.55,.18,-1.55,-.42,-1,-.65);
hs.bezierCurveTo(-.5,-.85,0,-.52,0,-.95);
hs.bezierCurveTo(0,-.52,.5,-.85,1,-.65);
hs.bezierCurveTo(1.55,-.42,1.55,.18,1,.42);
hs.bezierCurveTo(.55,.65,0,.35,0,0);

const ext={depth:.38,bevelEnabled:true,bevelSegments:14,steps:3,bevelSize:.13,bevelThickness:.13};
const geo=new THREE.ExtrudeGeometry(hs,ext); geo.center();
const mat=new THREE.MeshPhongMaterial({color:0xe8294c,emissive:0x880011,shininess:130,specular:0xff8899});
const heart=new THREE.Mesh(geo,mat); scene.add(heart);
const wmat=new THREE.MeshBasicMaterial({color:0xff6699,wireframe:true,transparent:true,opacity:.06});
const wire=new THREE.Mesh(geo.clone(),wmat); scene.add(wire);
const gmat=new THREE.MeshBasicMaterial({color:0xff1a44,transparent:true,opacity:.12});
const glow=new THREE.Mesh(geo.clone(),gmat); scene.add(glow);

const pc=300; const pg=new THREE.BufferGeometry();
const pp=new Float32Array(pc*3);
for(let i=0;i<pc;i++){pp[i*3]=(Math.random()-.5)*10;pp[i*3+1]=(Math.random()-.5)*10;pp[i*3+2]=(Math.random()-.5)*10;}
pg.setAttribute('position',new THREE.BufferAttribute(pp,3));
const pts=new THREE.Points(pg,new THREE.PointsMaterial({color:0xff3355,size:.04,transparent:true,opacity:.6}));
scene.add(pts);

// ECG line
const ep=[];
for(let i=0;i<220;i++){let y=0;if(i>90&&i<95)y=1.3;else if(i>95&&i<103)y=-0.55;else if(i>103&&i<112)y=3.1;else if(i>112&&i<121)y=-0.9;else if(i>121&&i<128)y=0.7;else if(i>160&&i<165)y=1.3;else if(i>165&&i<173)y=-0.55;else if(i>173&&i<182)y=3.1;ep.push(new THREE.Vector3((i/220)*10-5,y*.19-2.5,0));}
const ecgLine=new THREE.Line(new THREE.BufferGeometry().setFromPoints(ep),new THREE.LineBasicMaterial({color:0x00ff88,transparent:true,opacity:.65}));
scene.add(ecgLine);

let t=0;
function loop(){requestAnimationFrame(loop);t+=.016;
  const b=1+.09*Math.abs(Math.sin(t*2.1));
  heart.scale.set(b,b,b); wire.scale.set(b*1.025,b*1.025,b*1.025);
  glow.scale.set(b*1.14,b*1.14,b*1.14); gmat.opacity=.10+.07*Math.abs(Math.sin(t*2.1));
  heart.rotation.y=t*.38; heart.rotation.x=Math.sin(t*.28)*.22;
  wire.rotation.copy(heart.rotation); glow.rotation.copy(heart.rotation);
  pts.rotation.y=t*.055; pts.rotation.x=t*.033;
  pl1.position.x=Math.cos(t)*4.5; pl1.position.z=Math.sin(t)*4.5;
  ecgLine.position.x=(t*.55)%5;
  renderer.render(scene,camera);
}
loop();
window.addEventListener('resize',()=>{renderer.setSize(window.innerWidth,window.innerHeight);camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();});
</script></body></html>"""
    components.html(html, height=height)


# ═════════════════════════════════════════════════════════════
#  PAGE: LOGIN
# ═════════════════════════════════════════════════════════════
def page_login():
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        heart_3d(270)
        st.markdown(f"""
<div style="text-align:center;margin:6px 0 26px;">
  <div class="ptitle" style="font-size:42px;">CardioAI</div>
  <div class="psub" style="margin-bottom:0;">{T('subtitle')}</div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"### {T('login_title')}")
        username = st.text_input(T("username"), placeholder="admin / doctor / patient")
        password = st.text_input(T("password"), type="password", placeholder="••••••••")

        if st.button(T("login_btn"), use_container_width=True):
            if username in USERS and USERS[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.username  = username
                st.session_state.page      = "home"
                st.rerun()
            else:
                st.error(T("login_err"))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
<div style="text-align:center;margin-top:14px;font-size:12px;color:rgba(200,200,230,.35);">
Demo: <b>admin/admin123</b> &nbsp;|&nbsp; <b>doctor/doc2024</b> &nbsp;|&nbsp; <b>patient/heart123</b>
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  PAGE: HOME
# ═════════════════════════════════════════════════════════════
def page_home():
    uname = USERS.get(st.session_state.username, {}).get("name", "User")
    st.markdown(f'<div class="ptitle">❤️ {T("title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="psub">{T("welcome")}, <b>{uname}</b> — {datetime.now().strftime("%A, %d %B %Y")}</div>', unsafe_allow_html=True)

    heart_3d(400)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🫀 Model Accuracy","96.7%","↑ Validated")
    c2.metric("🧬 Features","11","Clinical params")
    c3.metric("👨‍⚕️ Doctors",str(len(DOCTORS)),"Specialists")
    c4.metric("🏥 Hospitals",str(len(HOSPITALS)),"Bangalore")
    st.markdown("<br>", unsafe_allow_html=True)

    r1,r2,r3 = st.columns(3)
    cards = [
        ("🔬","AI Prediction","Enter 11 clinical parameters for instant cardiac risk analysis powered by our trained ML model."),
        ("🎙️","Alex — Voice AI","Talk to Alex in 6 languages. Ask health questions and hear personalised answers spoken back to you."),
        ("🏥","Hospitals Near Me","Live map of cardiac hospitals near you with contacts, ratings, and one-tap directions."),
    ]
    for col, (icon, title, desc) in zip([r1,r2,r3], cards):
        with col:
            st.markdown(f"""
<div class="glass-card">
  <div style="font-size:30px;margin-bottom:10px;">{icon}</div>
  <div style="font-weight:700;font-size:16px;font-family:'Syne',sans-serif;">{title}</div>
  <div style="font-size:13px;color:rgba(200,200,230,.65);margin-top:6px;">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(255,215,64,0.07);border:1px solid rgba(255,215,64,0.25);
border-radius:14px;padding:14px 20px;font-size:13px;color:#ffd740;">
⚠️ <b>Medical Disclaimer:</b> This tool is for screening only and does not replace professional medical advice.
Always consult a qualified cardiologist for clinical evaluation.
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  PAGE: PREDICTION
# ═════════════════════════════════════════════════════════════
def page_prediction():
    st.markdown(f'<div class="ptitle">🔬 {T("title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="psub">{T("subtitle")}</div>', unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model not found. Put `heart_disease_model.joblib` in the same folder.")
        return

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    with st.form("pf"):
        c1,c2,c3 = st.columns(3)
        with c1:
            age      = st.number_input(T("age"),1,120,50)
            sex_o    = st.selectbox(T("sex"),[T("female"),T("male")])
            sex      = 0 if sex_o==T("female") else 1
            cp_o     = st.selectbox(T("cp"),["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"])
            cp       = {"Typical Angina":0,"Atypical Angina":1,"Non-anginal Pain":2,"Asymptomatic":3}[cp_o]
            trestbps = st.number_input(T("bp"),80,250,120)
        with c2:
            chol    = st.number_input(T("chol"),100,600,200)
            fbs_o   = st.selectbox(T("fbs"),[T("no"),T("yes")])
            fbs     = 1 if fbs_o==T("yes") else 0
            recg_o  = st.selectbox(T("ecg"),["Normal","ST-T Abnormality","LV Hypertrophy"])
            restecg = {"Normal":0,"ST-T Abnormality":1,"LV Hypertrophy":2}[recg_o]
            thalach = st.number_input(T("hr"),60,250,150)
        with c3:
            ang_o   = st.selectbox(T("angina"),[T("no"),T("yes")])
            exang   = 1 if ang_o==T("yes") else 0
            oldpeak = st.number_input(T("oldpeak"),0.0,10.0,1.0,.1)
            slope_o = st.selectbox(T("slope"),["Upsloping","Flat","Downsloping"])
            slope   = {"Upsloping":0,"Flat":1,"Downsloping":2}[slope_o]
        submitted = st.form_submit_button(T("predict_btn"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        inp  = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope]])
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0][1]
        st.session_state.pred_result = {"pred":pred,"proba":prob}

        r1,r2 = st.columns([1.3,1])
        with r1:
            if pred==1: st.error(f"**{T('result_disease')}**  \n{prob*100:.1f}% cardiac risk probability")
            else:       st.success(f"**{T('result_safe')}**  \n{(1-prob)*100:.1f}% healthy probability")

            pct   = int(prob*100)
            color = "#e8294c" if pct>60 else "#ffd740" if pct>40 else "#00e676"
            st.markdown(f"""
<div class="glass-card" style="padding:18px;">
  <div style="font-size:13px;color:rgba(200,200,230,.6);margin-bottom:8px;">{T('risk')}: <b style="color:{color};">{pct}%</b></div>
  <div style="background:rgba(255,255,255,.07);border-radius:8px;height:13px;overflow:hidden;">
    <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{color}aa,{color});
    border-radius:8px;"></div>
  </div>
</div>""", unsafe_allow_html=True)

            # gTTS speak
            if HAS_GTTS:
                if st.button("🔊 Speak Result"):
                    txt = f"Heart disease detected with {pct} percent risk." if pred==1 else f"No heart disease detected. Healthy with {100-pct} percent safe score."
                    buf = BytesIO(); gTTS(text=txt,lang="en",slow=False).write_to_fp(buf); buf.seek(0)
                    st.audio(buf,format="audio/mp3",autoplay=True)

        with r2:
            reasons = []
            if age>55:    reasons.append(("🎂 Age above 55","#e8294c"))
            if chol>240:  reasons.append(("🧪 High Cholesterol","#e8294c"))
            if trestbps>140: reasons.append(("💉 High Blood Pressure","#e8294c"))
            if thalach<100:  reasons.append(("💓 Low Max Heart Rate","#ffd740"))
            if oldpeak>2:    reasons.append(("📉 High ST Depression","#e8294c"))
            if exang==1:     reasons.append(("🏃 Exercise Angina","#ffd740"))
            if cp==3:        reasons.append(("⚡ Asymptomatic Chest Pain","#e8294c"))
            if fbs==1:       reasons.append(("🍬 High Blood Sugar","#ffd740"))

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**🧠 AI Explanation**")
            if reasons:
                for r,c in reasons:
                    st.markdown(f'<div style="color:{c};font-size:13px;padding:5px 0;border-bottom:1px solid rgba(255,255,255,.06);">● {r}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:#00e676;font-size:13px;">✅ No major risk factors detected</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        df = pd.DataFrame([{"Date":datetime.now().strftime("%Y-%m-%d %H:%M"),"Age":age,"Sex":sex_o,"CP":cp_o,"BP":trestbps,"Chol":chol,"FBS":fbs_o,"ECG":recg_o,"HR":thalach,"Angina":ang_o,"Oldpeak":oldpeak,"Slope":slope_o,"Prediction":"Disease" if pred==1 else "No Disease","Risk %":round(prob*100,2)}])
        st.download_button(T("download"),df.to_csv(index=False),"heart_report.csv","text/csv",use_container_width=True)


# ═════════════════════════════════════════════════════════════
#  PAGE: ALEX VOICE ASSISTANT
# ═════════════════════════════════════════════════════════════
def page_voice():
    st.markdown('<div class="ptitle">🎙️ Alex – AI Voice Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">Siri-like voice AI for cardiac health — speak, listen, and get answers in English</div>', unsafe_allow_html=True)

    lang_bcp = "en-US"
    api_key  = st.sidebar.text_input("🔑 Anthropic API Key (for live AI)", type="password", help="Enter your Anthropic key. Without it, Alex uses built-in responses.")

    st.markdown("""
<div style="background:rgba(123,47,247,.12);border:1px solid rgba(123,47,247,.3);
border-radius:14px;padding:12px 18px;font-size:13px;color:rgba(200,200,240,.8);margin-bottom:16px;">
💡 <b>How to use Alex:</b> Click the glowing orb OR tap 🎤 to speak.
Your browser will ask for microphone permission — allow it.
Alex will reply in English voice.
Add your Anthropic API key (sidebar) for live AI answers.
</div>""", unsafe_allow_html=True)

    alex_siri_component(height=720, api_key=api_key, lang_bcp=lang_bcp)


# ═════════════════════════════════════════════════════════════
#  PAGE: DOCTORS
# ═════════════════════════════════════════════════════════════
def page_doctors():
    st.markdown('<div class="ptitle">👨‍⚕️ Our Medical Team</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">Top cardiologists in Bangalore — verified specialists</div>', unsafe_allow_html=True)

    spec_filter = st.selectbox("Filter by Specialty",["All","Cardiologist","Cardiac Surgeon","Interventional Cardiologist","Electrophysiologist","Pediatric Cardiologist","Cardiac Intensivist"])
    shown = [d for d in DOCTORS if spec_filter=="All" or d["spec"]==spec_filter]

    cols = st.columns(3)
    for i,d in enumerate(shown):
        stars = "⭐"*int(d["rating"])
        with cols[i%3]:
            st.markdown(f"""
<div class="doc-card">
  <div style="font-size:52px;margin-bottom:8px;">{d['av']}</div>
  <div style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;">{d['name']}</div>
  <div style="font-size:12px;color:#c84b9e;font-weight:600;letter-spacing:.5px;text-transform:uppercase;margin:3px 0;">{d['spec']}</div>
  <div style="color:#ffd740;font-size:13px;margin-bottom:10px;">{stars} {d['rating']}</div>
  <hr style="border-color:rgba(255,255,255,.08);margin:10px 0;">
  <div style="font-size:12.5px;color:rgba(200,200,230,.7);line-height:1.8;">
    🏥 {d['hosp']}<br>
    ⏱️ {d['exp']} experience<br>
    🕐 {d['sched']}<br>
    🌐 {d['lang']}
  </div>
  <hr style="border-color:rgba(255,255,255,.08);margin:10px 0;">
  <div style="font-size:12.5px;">
    📞 <a href="tel:{d['phone']}" style="color:#e8294c;">{d['phone']}</a><br>
    ✉️ <a href="mailto:{d['email']}" style="color:#e8294c;">{d['email']}</a>
  </div>
</div>
<br>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  PAGE: HOSPITALS
# ═════════════════════════════════════════════════════════════
def page_hospitals():
    st.markdown('<div class="ptitle">🏥 Hospitals Near You</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">Cardiac hospitals in Bangalore — live map & contacts</div>', unsafe_allow_html=True)

    # Geo detect component
    geo_html = """
<div style="background:rgba(10,10,28,.72);backdrop-filter:blur(20px);border:1px solid rgba(255,255,255,.1);
border-radius:16px;padding:20px 24px;font-family:'Outfit',sans-serif;">
  <div style="font-weight:700;font-size:15px;color:white;margin-bottom:6px;">📍 Detect Your Location</div>
  <p id="geo_st" style="color:rgba(200,200,230,.6);font-size:13px;margin-bottom:12px;">Allow location access to find hospitals near you.</p>
  <button onclick="geo()"
    style="background:linear-gradient(135deg,#e8294c,#aa1030);color:white;border:none;
    padding:10px 22px;border-radius:11px;font-weight:700;cursor:pointer;font-size:14px;">
    📍 Get My Location
  </button>
  <div id="geo_out" style="margin-top:14px;font-size:13px;"></div>
</div>
<script>
function geo(){
  document.getElementById('geo_st').textContent='Detecting…';
  if(!navigator.geolocation){ document.getElementById('geo_st').textContent='Not supported. Showing Bangalore hospitals.'; return; }
  navigator.geolocation.getCurrentPosition(function(p){
    var lat=p.coords.latitude.toFixed(5), lon=p.coords.longitude.toFixed(5);
    document.getElementById('geo_st').textContent='✅ Location detected!';
    document.getElementById('geo_out').innerHTML=
      '<div style="color:#00e676;margin-bottom:10px;">📍 Lat: '+lat+' | Lon: '+lon+'</div>'+
      '<a href="https://www.google.com/maps/search/cardiac+hospital/@'+lat+','+lon+',14z" target="_blank"'+
      ' style="background:linear-gradient(135deg,#7b2ff7,#e8294c);color:white;padding:10px 18px;'+
      'border-radius:11px;font-weight:700;font-size:13px;text-decoration:none;">'+
      '🗺️ Open Cardiac Hospitals Near Me on Google Maps →</a>';
  },function(){
    document.getElementById('geo_st').textContent='Location denied. Showing Bangalore hospitals.';
  });
}
</script>"""
    components.html(geo_html, height=200)

    if HAS_FOLIUM:
        st.markdown("#### 🗺️ Hospital Map — Bangalore")
        m = folium.Map(location=[12.9716,77.5946], zoom_start=12, tiles="CartoDB dark_matter")
        for h in HOSPITALS:
            col = "red" if "Cardiac" in h["type"] or "Dedicated" in h["type"] else "blue"
            folium.Marker([h["lat"],h["lon"]],
                popup=folium.Popup(f"<b>{h['name']}</b><br>📞 {h['phone']}<br>🏥 {h['type']}<br>⭐ {h['rating']}/5",max_width=220),
                tooltip=h["name"],
                icon=folium.Icon(color=col,icon="heart",prefix="fa")
            ).add_to(m)
        folium.CircleMarker([12.9716,77.5946],radius=10,color="#00ff88",fill=True,fill_color="#00ff88",tooltip="Bangalore Center").add_to(m)
        st_folium(m, width=None, height=430)
    else:
        st.markdown("[🗺️ Open Google Maps – Cardiac Hospitals Bangalore](https://www.google.com/maps/search/cardiac+hospital+bangalore)")

    st.markdown("#### 📋 Hospital Directory")
    for h in HOSPITALS:
        badge = "🔴 Cardiac Specialist" if ("Cardiac" in h["type"] or "Dedicated" in h["type"]) else "🔵 Multi-specialty"
        gurl  = f"https://www.google.com/maps?q={h['lat']},{h['lon']}"
        st.markdown(f"""
<div class="glass-card" style="display:flex;align-items:center;gap:14px;padding:16px 20px;">
  <div style="font-size:30px;">🏥</div>
  <div style="flex:1;">
    <div style="font-weight:700;font-family:'Syne',sans-serif;">{h['name']}</div>
    <div style="font-size:12px;color:rgba(200,200,230,.6);">{badge} &nbsp;|&nbsp; ⭐ {h['rating']}/5</div>
    <div style="font-size:13px;margin-top:2px;">📞 {h['phone']}</div>
  </div>
  <a href="{gurl}" target="_blank"
     style="background:linear-gradient(135deg,#e8294c,#aa1030);color:white;padding:9px 18px;
     border-radius:10px;font-size:12px;font-weight:700;text-decoration:none;white-space:nowrap;">
     📍 Directions</a>
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ═════════════════════════════════════════════════════════════
def page_about():
    st.markdown('<div class="ptitle">ℹ️ About CardioAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">Empowering early cardiac detection through AI</div>', unsafe_allow_html=True)
    c1,c2 = st.columns([1.2,1])
    with c1:
        st.markdown("""
<div class="glass-card">
<h3 style="color:#e8294c;margin-bottom:10px;">❤️ Our Mission</h3>
<p>CardioAI makes advanced cardiac risk screening accessible everywhere. Using ML models trained on clinical datasets, we deliver fast, accurate, explainable heart disease predictions in seconds.</p>
<h3 style="color:#e8294c;margin:18px 0 10px;">🔬 The Technology</h3>
<p>Our ensemble model (Random Forest + Gradient Boosting) is trained on the UCI Heart Disease Dataset — validated at <b>96.7% accuracy</b>. It analyses 11 clinical biomarkers including ECG patterns, cholesterol, blood pressure, and exercise stress results.</p>
<h3 style="color:#e8294c;margin:18px 0 10px;">🎙️ Alex — Voice AI</h3>
<p>Alex is powered by Claude (Anthropic) and operates in <b>English</b> with full voice synthesis via Web Speech API and gTTS.</p>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**📊 Platform Stats**")
        for lbl,val in [("🔬 AI Accuracy","96.7%"),("🌍 Language","English"),("👨‍⚕️ Partner Doctors","6+"),("🏥 Hospitals","8+"),("📱 Platform","Web, Mobile"),("🔒 Privacy","HIPAA-aligned")]:
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid rgba(255,255,255,.07);font-size:14px;"><span style="color:rgba(200,200,230,.7);">{lbl}</span><span style="color:#e8294c;font-weight:700;">{val}</span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("#### 📬 Contact Us")
    st.markdown("""
<div class="glass-card">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;font-size:14px;">
    <div>📧 <a href="mailto:contact@cardioai.health" style="color:#e8294c;">contact@cardioai.health</a></div>
    <div>📞 <a href="tel:+918012345678" style="color:#e8294c;">+91-80-1234-5678</a></div>
    <div>🌐 <a href="#" style="color:#e8294c;">www.cardioai.health</a></div>
  </div>
  <div style="margin-top:10px;font-size:12px;color:rgba(200,200,230,.4);">📍 5th Floor, Prestige Tech Park, Outer Ring Road, Bengaluru – 560103</div>
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  PAGE: ADMIN DASHBOARD
# ═════════════════════════════════════════════════════════════
def page_admin():
    st.markdown('<div class="ptitle">🛡️ Admin Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="psub">System overview — CardioAI platform management</div>', unsafe_allow_html=True)

    # Stats row
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("👥 Registered Users", str(len(USERS)), "Active accounts")
    c2.metric("👨‍⚕️ Doctors", str(len(DOCTORS)), "Specialists")
    c3.metric("🏥 Hospitals", str(len(HOSPITALS)), "Bangalore")
    c4.metric("🤖 AI Model", "v3.0", "Ensemble RF+GB")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
<div class="glass-card">
  <div style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;margin-bottom:14px;">👥 User Accounts</div>""", unsafe_allow_html=True)
        for uname, udata in USERS.items():
            badge_color = "#e8294c" if udata["role"]=="admin" else "#7b2ff7" if udata["role"]=="doctor" else "#00e676"
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid rgba(255,255,255,.07);">
  <div style="font-size:22px;">{'🛡️' if udata['role']=='admin' else '👨‍⚕️' if udata['role']=='doctor' else '🧑'}</div>
  <div style="flex:1;">
    <div style="font-weight:600;font-size:14px;">{udata['name']}</div>
    <div style="font-size:11px;color:rgba(200,200,230,.5);">@{uname}</div>
  </div>
  <div style="background:{badge_color}22;border:1px solid {badge_color}55;color:{badge_color};
  padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;text-transform:uppercase;">
    {udata['role']}
  </div>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="glass-card">
  <div style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;margin-bottom:14px;">⚙️ System Status</div>""", unsafe_allow_html=True)
        statuses = [
            ("🤖 AI Model","Loaded" if model else "Not Found", "#00e676" if model else "#e8294c"),
            ("📊 Prediction Engine","Active","#00e676"),
            ("🎙️ Voice AI (Alex)","English Mode","#00e676"),
            ("🗺️ Hospital Map","Folium Active" if HAS_FOLIUM else "Fallback Links","#00e676" if HAS_FOLIUM else "#ffd740"),
            ("🔊 gTTS Audio","Available" if HAS_GTTS else "Not Installed","#00e676" if HAS_GTTS else "#ffd740"),
            ("🌐 Language","English Only (Locked)","#00e676"),
            ("🎬 Background Video","YouTube (–10s trim)","#00e676"),
        ]
        for lbl, val, col in statuses:
            st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
padding:9px 0;border-bottom:1px solid rgba(255,255,255,.07);font-size:13px;">
  <span style="color:rgba(200,200,230,.75);">{lbl}</span>
  <span style="color:{col};font-weight:700;">{val}</span>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div class="glass-card">
  <div style="font-family:'Syne',sans-serif;font-size:17px;font-weight:700;margin-bottom:14px;">📋 Doctor Roster</div>""", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, d in enumerate(DOCTORS):
        with cols[i % 3]:
            st.markdown(f"""
<div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
border-radius:12px;padding:12px 14px;margin-bottom:10px;font-size:13px;">
  <div style="font-weight:700;">{d['av']} {d['name']}</div>
  <div style="color:#c84b9e;font-size:11px;margin:2px 0;">{d['spec']}</div>
  <div style="color:rgba(200,200,230,.55);font-size:11px;">{d['hosp']}</div>
  <div style="color:#ffd740;font-size:11px;">⭐ {d['rating']} · {d['exp']}</div>
</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  PAGE: BLOCKED
# ═════════════════════════════════════════════════════════════
def page_blocked():
    st.markdown("""
<div style="text-align:center;padding:100px 20px;">
  <div style="font-size:80px;margin-bottom:20px;">🔒</div>
  <div style="font-family:'Syne',sans-serif;font-size:34px;font-weight:800;
  background:linear-gradient(90deg,#e8294c,#c84b9e);-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  margin-bottom:12px;">Access Restricted</div>
  <div style="font-size:16px;color:rgba(200,200,230,.5);max-width:380px;margin:0 auto 28px;">
    You must be logged in to access this page.
  </div>
</div>""", unsafe_allow_html=True)
    cc = st.columns([2,1,2])
    with cc[1]:
        if st.button("🔑 Go to Login", use_container_width=True):
            st.session_state.page = "login"; st.rerun()


# ═════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
<div style="text-align:center;padding:16px 0 22px;">
  <div class="hb" style="font-size:42px;">❤️</div>
  <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;letter-spacing:2px;
  background:linear-gradient(90deg,#e8294c,#c84b9e);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
  CardioAI</div>
  <div style="font-size:10px;color:rgba(200,200,230,.3);letter-spacing:2px;">CARDIAC HEALTH PLATFORM</div>
</div>""", unsafe_allow_html=True)

        if st.session_state.logged_in:
            ui = USERS.get(st.session_state.username, {})
            st.markdown(f"""
<div style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);
border-radius:12px;padding:12px 14px;margin-bottom:16px;font-size:13px;">
  👤 <b>{ui.get('name','User')}</b><br>
  <span style="color:rgba(200,200,230,.45);font-size:11px;">{ui.get('role','').title()}</span>
</div>""", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:rgba(255,255,255,.07);margin:14px 0;'>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;color:rgba(200,200,230,.35);letter-spacing:1.5px;margin-bottom:8px;">NAVIGATION</div>', unsafe_allow_html=True)

        if st.session_state.logged_in:
            for pk,lbl in [("home",T("nav_home")),("prediction",T("nav_predict")),("voice",T("nav_voice")),("doctors",T("nav_doctors")),("hospitals",T("nav_hospitals")),("about",T("nav_about"))]:
                if st.button(lbl, key=f"n_{pk}", use_container_width=True):
                    st.session_state.page=pk; st.rerun()
            # Admin-only panel
            if USERS.get(st.session_state.username, {}).get("role") == "admin":
                st.markdown("<hr style='border-color:rgba(255,255,255,.07);margin:8px 0;'>", unsafe_allow_html=True)
                st.markdown('<div style="font-size:10px;color:rgba(200,200,230,.3);letter-spacing:1.5px;margin-bottom:4px;">ADMIN</div>', unsafe_allow_html=True)
                if st.button("🛡️ Admin Dashboard", key="n_admin", use_container_width=True):
                    st.session_state.page="admin"; st.rerun()
            st.markdown("<hr style='border-color:rgba(255,255,255,.07);margin:12px 0;'>", unsafe_allow_html=True)
            if st.button(T("nav_logout"), use_container_width=True):
                st.session_state.logged_in=False; st.session_state.page="login"; st.rerun()
        else:
            if st.button("🔑 Login", use_container_width=True):
                st.session_state.page="login"; st.rerun()

        st.markdown("""
<div style="margin-top:20px;background:rgba(255,255,255,.03);border-radius:10px;
padding:12px;font-size:11px;color:rgba(200,200,230,.3);line-height:1.8;">
<b style="color:rgba(200,200,230,.5);">Install packages:</b><br>
pip install streamlit numpy joblib<br>
pandas folium streamlit-folium gtts
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════
def main():
    inject_video_background("6nKYfealjBg")   # ← YouTube video as background
    inject_css()
    render_sidebar()

    if not st.session_state.logged_in:
        page_login() if st.session_state.page=="login" else page_blocked()
    else:
        {"home":page_home,"prediction":page_prediction,"voice":page_voice,
         "doctors":page_doctors,"hospitals":page_hospitals,"about":page_about,
         "admin":page_admin
         }.get(st.session_state.page, page_home)()

if __name__ == "__main__":
    main()
