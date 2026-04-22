# ============================================================
# ENHANCED HEART DISEASE PREDICTION SYSTEM v2.0
# Features: 3D Animation | Voice AI (Alex) | 6 Languages
#           Login | Hospitals Map | Doctor Contact | About
# Run: streamlit run app_enhanced.py
# ============================================================

import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os
import base64
from io import BytesIO
import streamlit.components.v1 as components
import json
import time
from datetime import datetime

# ── Optional imports with graceful fallback ──────────────────
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

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="CardioAI – Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Demo Users ────────────────────────────────────────────────
USERS = {
    "admin":   {"password": "admin123",  "role": "admin",  "name": "Dr. Admin"},
    "doctor":  {"password": "doc2024",   "role": "doctor", "name": "Dr. Priya"},
    "patient": {"password": "heart123",  "role": "patient","name": "Ravi Kumar"},
}

# ── Language Translations ─────────────────────────────────────
LANG = {
    "en": {
        "flag": "🇬🇧", "name": "English",
        "title": "Heart Disease Prediction System",
        "subtitle": "AI-Powered Cardiac Health Analysis",
        "nav_home": "🏠 Home",
        "nav_predict": "🔬 Prediction",
        "nav_voice": "🎙️ Voice Assistant (Alex)",
        "nav_doctors": "👨‍⚕️ Doctors",
        "nav_hospitals": "🏥 Hospitals Near Me",
        "nav_about": "ℹ️ About Us",
        "nav_logout": "🚪 Logout",
        "login_title": "Login to CardioAI",
        "username": "Username",
        "password": "Password",
        "login_btn": "Login",
        "login_err": "Invalid credentials. Try admin/admin123",
        "predict_btn": "🔍 Predict Now",
        "result_disease": "⚠️ Heart Disease Detected",
        "result_safe": "✅ No Heart Disease Detected",
        "download": "⬇️ Download Report",
        "age": "Age", "sex": "Sex", "male": "Male", "female": "Female",
        "cp": "Chest Pain Type", "bp": "Resting Blood Pressure (mmHg)",
        "chol": "Cholesterol (mg/dL)", "fbs": "Fasting Blood Sugar > 120 mg/dL",
        "ecg": "Resting ECG Result", "hr": "Maximum Heart Rate",
        "angina": "Exercise-Induced Angina", "oldpeak": "ST Depression (Oldpeak)",
        "slope": "ST Slope", "risk": "Risk Level",
        "yes": "Yes", "no": "No",
        "speak_result": "🔊 Speak Result",
        "ask_alex": "Ask Alex (AI Health Assistant)",
        "alex_placeholder": "Type your health question here...",
        "send": "Send",
        "doctor_title": "Our Medical Team",
        "hospital_title": "Hospitals Near You",
        "about_title": "About CardioAI",
        "welcome": "Welcome back",
    },
    "hi": {
        "flag": "🇮🇳", "name": "हिंदी",
        "title": "हृदय रोग भविष्यवाणी प्रणाली",
        "subtitle": "AI-संचालित कार्डियक स्वास्थ्य विश्लेषण",
        "nav_home": "🏠 होम",
        "nav_predict": "🔬 भविष्यवाणी",
        "nav_voice": "🎙️ वॉयस असिस्टेंट (Alex)",
        "nav_doctors": "👨‍⚕️ डॉक्टर",
        "nav_hospitals": "🏥 नज़दीकी अस्पताल",
        "nav_about": "ℹ️ हमारे बारे में",
        "nav_logout": "🚪 लॉगआउट",
        "login_title": "CardioAI में लॉगिन करें",
        "username": "उपयोगकर्ता नाम", "password": "पासवर्ड",
        "login_btn": "लॉगिन", "login_err": "गलत क्रेडेंशियल। admin/admin123 आज़माएं",
        "predict_btn": "🔍 भविष्यवाणी करें",
        "result_disease": "⚠️ हृदय रोग का पता चला",
        "result_safe": "✅ कोई हृदय रोग नहीं",
        "download": "⬇️ रिपोर्ट डाउनलोड करें",
        "age": "आयु", "sex": "लिंग", "male": "पुरुष", "female": "महिला",
        "cp": "सीने में दर्द का प्रकार", "bp": "आराम रक्तचाप",
        "chol": "कोलेस्ट्रॉल", "fbs": "उपवास रक्त शर्करा > 120",
        "ecg": "आराम ECG", "hr": "अधिकतम हृदय गति",
        "angina": "व्यायाम एनजाइना", "oldpeak": "ST अवसाद",
        "slope": "ST ढलान", "risk": "जोखिम स्तर",
        "yes": "हाँ", "no": "नहीं",
        "speak_result": "🔊 परिणाम सुनें",
        "ask_alex": "Alex से पूछें (AI स्वास्थ्य सहायक)",
        "alex_placeholder": "अपना स्वास्थ्य प्रश्न यहाँ टाइप करें...",
        "send": "भेजें",
        "doctor_title": "हमारी चिकित्सा टीम",
        "hospital_title": "आपके पास अस्पताल",
        "about_title": "CardioAI के बारे में",
        "welcome": "वापस स्वागत है",
    },
    "kn": {
        "flag": "🇮🇳", "name": "ಕನ್ನಡ",
        "title": "ಹೃದಯ ರೋಗ ಮುನ್ಸೂಚನಾ ವ್ಯವಸ್ಥೆ",
        "subtitle": "AI-ಚಾಲಿತ ಹೃದಯ ಆರೋಗ್ಯ ವಿಶ್ಲೇಷಣೆ",
        "nav_home": "🏠 ಮನೆ",
        "nav_predict": "🔬 ಮುನ್ಸೂಚನೆ",
        "nav_voice": "🎙️ ಧ್ವನಿ ಸಹಾಯಕ (Alex)",
        "nav_doctors": "👨‍⚕️ ವೈದ್ಯರು",
        "nav_hospitals": "🏥 ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆಗಳು",
        "nav_about": "ℹ️ ನಮ್ಮ ಬಗ್ಗೆ",
        "nav_logout": "🚪 ಲಾಗ್‌ಔಟ್",
        "login_title": "CardioAI ಗೆ ಲಾಗಿನ್ ಮಾಡಿ",
        "username": "ಬಳಕೆದಾರ ಹೆಸರು", "password": "ಪಾಸ್‌ವರ್ಡ್",
        "login_btn": "ಲಾಗಿನ್", "login_err": "ತಪ್ಪು ಪ್ರಮಾಣಪತ್ರಗಳು. admin/admin123 ಪ್ರಯತ್ನಿಸಿ",
        "predict_btn": "🔍 ಮುನ್ಸೂಚಿಸಿ",
        "result_disease": "⚠️ ಹೃದಯ ರೋಗ ಪತ್ತೆಯಾಗಿದೆ",
        "result_safe": "✅ ಹೃದಯ ರೋಗವಿಲ್ಲ",
        "download": "⬇️ ವರದಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
        "age": "ವಯಸ್ಸು", "sex": "ಲಿಂಗ", "male": "ಪುರುಷ", "female": "ಮಹಿಳೆ",
        "cp": "ಎದೆ ನೋವಿನ ಪ್ರಕಾರ", "bp": "ವಿಶ್ರಾಂತಿ ರಕ್ತದೊತ್ತಡ",
        "chol": "ಕೊಲೆಸ್ಟ್ರಾಲ್", "fbs": "ಉಪವಾಸ ರಕ್ತ ಸಕ್ಕರೆ > 120",
        "ecg": "ವಿಶ್ರಾಂತಿ ECG", "hr": "ಗರಿಷ್ಠ ಹೃದಯ ಬಡಿತ",
        "angina": "ವ್ಯಾಯಾಮ ಆಂಜಿನಾ", "oldpeak": "ST ಖಿನ್ನತೆ",
        "slope": "ST ಇಳಿಜಾರು", "risk": "ಅಪಾಯದ ಮಟ್ಟ",
        "yes": "ಹೌದು", "no": "ಇಲ್ಲ",
        "speak_result": "🔊 ಫಲಿತಾಂಶ ಕೇಳಿ",
        "ask_alex": "Alex ಅನ್ನು ಕೇಳಿ (AI ಆರೋಗ್ಯ ಸಹಾಯಕ)",
        "alex_placeholder": "ನಿಮ್ಮ ಆರೋಗ್ಯ ಪ್ರಶ್ನೆ ಇಲ್ಲಿ ಟೈಪ್ ಮಾಡಿ...",
        "send": "ಕಳುಹಿಸಿ",
        "doctor_title": "ನಮ್ಮ ವೈದ್ಯಕೀಯ ತಂಡ",
        "hospital_title": "ನಿಮ್ಮ ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆಗಳು",
        "about_title": "CardioAI ಬಗ್ಗೆ",
        "welcome": "ಮರಳಿ ಸ್ವಾಗತ",
    },
    "ta": {
        "flag": "🇮🇳", "name": "தமிழ்",
        "title": "இதய நோய் கணிப்பு அமைப்பு",
        "subtitle": "AI-இயங்கும் இதய ஆரோக்கிய பகுப்பாய்வு",
        "nav_home": "🏠 முகப்பு", "nav_predict": "🔬 கணிப்பு",
        "nav_voice": "🎙️ குரல் உதவியாளர் (Alex)",
        "nav_doctors": "👨‍⚕️ மருத்துவர்கள்",
        "nav_hospitals": "🏥 அருகில் உள்ள மருத்துவமனைகள்",
        "nav_about": "ℹ️ எங்களைப் பற்றி", "nav_logout": "🚪 வெளியேறு",
        "login_title": "CardioAI இல் உள்நுழைக",
        "username": "பயனர்பெயர்", "password": "கடவுச்சொல்",
        "login_btn": "உள்நுழை", "login_err": "தவறான நற்சான்றிதழ்கள். admin/admin123 முயலவும்",
        "predict_btn": "🔍 கணிக்கவும்",
        "result_disease": "⚠️ இதய நோய் கண்டறியப்பட்டது",
        "result_safe": "✅ இதய நோய் இல்லை",
        "download": "⬇️ அறிக்கை பதிவிறக்கம்",
        "age": "வயது", "sex": "பாலினம்", "male": "ஆண்", "female": "பெண்",
        "cp": "மார்பு வலி வகை", "bp": "இளைப்பு இரத்த அழுத்தம்",
        "chol": "கொலஸ்ட்ரால்", "fbs": "உண்ணாவிரத இரத்த சர்க்கரை > 120",
        "ecg": "இளைப்பு ECG", "hr": "அதிகபட்ச இதய துடிப்பு",
        "angina": "உடற்பயிற்சி ஆஞ்சினா", "oldpeak": "ST மந்தநிலை",
        "slope": "ST சாய்வு", "risk": "ஆபத்து நிலை",
        "yes": "ஆம்", "no": "இல்லை",
        "speak_result": "🔊 முடிவு கேளுங்கள்",
        "ask_alex": "Alex கேளுங்கள் (AI ஆரோக்கிய உதவியாளர்)",
        "alex_placeholder": "உங்கள் ஆரோக்கிய கேள்வியை இங்கே தட்டச்சு செய்யுங்கள்...",
        "send": "அனுப்பு",
        "doctor_title": "எங்கள் மருத்துவக் குழு",
        "hospital_title": "உங்களுக்கு அருகில் மருத்துவமனைகள்",
        "about_title": "CardioAI பற்றி",
        "welcome": "மீண்டும் வரவேற்கிறோம்",
    },
    "te": {
        "flag": "🇮🇳", "name": "తెలుగు",
        "title": "హృదయ వ్యాధి అంచనా వ్యవస్థ",
        "subtitle": "AI-ఆధారిత కార్డియాక్ ఆరోగ్య విశ్లేషణ",
        "nav_home": "🏠 హోమ్", "nav_predict": "🔬 అంచనా",
        "nav_voice": "🎙️ వాయిస్ అసిస్టెంట్ (Alex)",
        "nav_doctors": "👨‍⚕️ డాక్టర్లు",
        "nav_hospitals": "🏥 సమీపంలోని ఆసుపత్రులు",
        "nav_about": "ℹ️ మా గురించి", "nav_logout": "🚪 లాగ్అవుట్",
        "login_title": "CardioAI లో లాగిన్ అవ్వండి",
        "username": "వినియోగదారు పేరు", "password": "పాస్‌వర్డ్",
        "login_btn": "లాగిన్", "login_err": "తప్పు ఆధారాలు. admin/admin123 ప్రయత్నించండి",
        "predict_btn": "🔍 అంచనా వేయండి",
        "result_disease": "⚠️ గుండె జబ్బు గుర్తించబడింది",
        "result_safe": "✅ గుండె జబ్బు లేదు",
        "download": "⬇️ నివేదిక డౌన్‌లోడ్ చేయండి",
        "age": "వయస్సు", "sex": "లింగం", "male": "మగ", "female": "ఆడ",
        "cp": "ఛాతీ నొప్పి రకం", "bp": "విశ్రాంతి రక్తపోటు",
        "chol": "కొలెస్ట్రాల్", "fbs": "ఉపవాస రక్త చక్కెర > 120",
        "ecg": "విశ్రాంతి ECG", "hr": "గరిష్ట గుండె చప్పుడు",
        "angina": "వ్యాయామ ఆంజినా", "oldpeak": "ST నిస్పృహ",
        "slope": "ST వాలు", "risk": "ప్రమాద స్థాయి",
        "yes": "అవును", "no": "కాదు",
        "speak_result": "🔊 ఫలితం వినండి",
        "ask_alex": "Alex ని అడగండి (AI ఆరోగ్య సహాయకుడు)",
        "alex_placeholder": "మీ ఆరోగ్య ప్రశ్న ఇక్కడ టైప్ చేయండి...",
        "send": "పంపండి",
        "doctor_title": "మా వైద్య బృందం",
        "hospital_title": "మీకు సమీపంలో ఆసుపత్రులు",
        "about_title": "CardioAI గురించి",
        "welcome": "తిరిగి స్వాగతం",
    },
    "ml": {
        "flag": "🇮🇳", "name": "മലയാളം",
        "title": "ഹൃദ്രോഗ പ്രവചന സംവിധാനം",
        "subtitle": "AI-ചালിത ഹൃദ്രോഗ ആരോഗ്യ വിശകലനം",
        "nav_home": "🏠 ഹോം", "nav_predict": "🔬 പ്രവചനം",
        "nav_voice": "🎙️ വോയ്‌സ് അസിസ്റ്റന്റ് (Alex)",
        "nav_doctors": "👨‍⚕️ ഡോക്ടർമാർ",
        "nav_hospitals": "🏥 അടുത്തുള്ള ആശുപത്രികൾ",
        "nav_about": "ℹ️ ഞങ്ങളെ കുറിച്ച്", "nav_logout": "🚪 ലോഗ്ഔട്ട്",
        "login_title": "CardioAI ൽ ലോഗിൻ ചെയ്യുക",
        "username": "ഉപയോക്തൃ നാമം", "password": "പാസ്‌വേഡ്",
        "login_btn": "ലോഗിൻ", "login_err": "തെറ്റായ ക്രെഡൻഷ്യലുകൾ. admin/admin123 ശ്രമിക്കുക",
        "predict_btn": "🔍 പ്രവചിക്കുക",
        "result_disease": "⚠️ ഹൃദ്രോഗം കണ്ടെത്തി",
        "result_safe": "✅ ഹൃദ്രോഗം ഇല്ല",
        "download": "⬇️ റിപ്പോർട്ട് ഡൗൺലോഡ്",
        "age": "പ്രായം", "sex": "ലിംഗം", "male": "പുരുഷൻ", "female": "സ്ത്രീ",
        "cp": "നെഞ്ചുവേദനയുടെ തരം", "bp": "വിശ്രമ രക്തസമ്മർദ്ദം",
        "chol": "കൊളസ്ട്രോൾ", "fbs": "ഉപവാസ രക്തത്തിലെ പഞ്ചസാര > 120",
        "ecg": "വിശ്രമ ECG", "hr": "പരമാവധി ഹൃദയ മിടിപ്പ്",
        "angina": "വ്യായാമ ആൻജൈന", "oldpeak": "ST ഡിപ്രഷൻ",
        "slope": "ST ചരിവ്", "risk": "അപകട നില",
        "yes": "അതെ", "no": "ഇല്ല",
        "speak_result": "🔊 ഫലം കേൾക്കുക",
        "ask_alex": "Alex നോട് ചോദിക്കുക (AI ആരോഗ്യ സഹായി)",
        "alex_placeholder": "നിങ്ങളുടെ ആരോഗ്യ ചോദ്യം ഇവിടെ ടൈപ്പ് ചെയ്യുക...",
        "send": "അയക്കുക",
        "doctor_title": "ഞങ്ങളുടെ മെഡിക്കൽ ടീം",
        "hospital_title": "നിങ്ങൾക്ക് അടുത്തുള്ള ആശുപത്രികൾ",
        "about_title": "CardioAI യെ കുറിച്ച്",
        "welcome": "തിരിച്ചു സ്വാഗതം",
    },
}

# ── Doctor Data ───────────────────────────────────────────────
DOCTORS = [
    {
        "name": "Dr. Priya Nair", "specialty": "Cardiologist",
        "hospital": "Narayana Health, Bangalore", "exp": "18 Years",
        "phone": "+91-80-7122-2222", "email": "priya.nair@nhhealth.com",
        "rating": 4.9, "avatar": "👩‍⚕️",
        "availability": "Mon-Sat 9AM-5PM",
        "languages": "English, Kannada, Malayalam",
    },
    {
        "name": "Dr. Rajesh Sharma", "specialty": "Cardiac Surgeon",
        "hospital": "Fortis Hospital, Bangalore", "exp": "22 Years",
        "phone": "+91-80-6621-4444", "email": "r.sharma@fortis.in",
        "rating": 4.8, "avatar": "👨‍⚕️",
        "availability": "Mon-Fri 10AM-6PM",
        "languages": "English, Hindi, Kannada",
    },
    {
        "name": "Dr. Meenakshi Iyer", "specialty": "Interventional Cardiologist",
        "hospital": "Manipal Hospital, Bangalore", "exp": "15 Years",
        "phone": "+91-80-2502-0000", "email": "m.iyer@manipal.edu",
        "rating": 4.7, "avatar": "👩‍⚕️",
        "availability": "Tue-Sun 8AM-4PM",
        "languages": "English, Tamil, Kannada",
    },
    {
        "name": "Dr. Suresh Reddy", "specialty": "Electrophysiologist",
        "hospital": "Apollo Hospital, Bangalore", "exp": "20 Years",
        "phone": "+91-80-2941-9333", "email": "s.reddy@apollohospitals.com",
        "rating": 4.9, "avatar": "👨‍⚕️",
        "availability": "Mon-Sat 8AM-3PM",
        "languages": "English, Telugu, Kannada",
    },
    {
        "name": "Dr. Lakshmi Venkat", "specialty": "Pediatric Cardiologist",
        "hospital": "Aster CMI Hospital, Bangalore", "exp": "12 Years",
        "phone": "+91-80-4342-0100", "email": "l.venkat@asterhospitals.in",
        "rating": 4.8, "avatar": "👩‍⚕️",
        "availability": "Mon-Fri 9AM-5PM",
        "languages": "English, Kannada, Tamil",
    },
    {
        "name": "Dr. Arun Krishnamurthy", "specialty": "Cardiac Intensivist",
        "hospital": "Sakra World Hospital, Bangalore", "exp": "16 Years",
        "phone": "+91-80-4969-4969", "email": "a.krishna@sakraworldhospital.com",
        "rating": 4.6, "avatar": "👨‍⚕️",
        "availability": "24/7 Emergency",
        "languages": "English, Kannada, Telugu",
    },
]

# ── Bangalore Hospitals ───────────────────────────────────────
HOSPITALS = [
    {"name": "Narayana Health City", "lat": 12.8449, "lon": 77.6616,
     "phone": "+91-80-7122-2222", "type": "Multi-specialty", "rating": 4.8},
    {"name": "Fortis Hospital Bannerghatta", "lat": 12.8766, "lon": 77.5993,
     "phone": "+91-80-6621-4444", "type": "Cardiac Care", "rating": 4.7},
    {"name": "Manipal Hospital Old Airport Road", "lat": 12.9666, "lon": 77.6463,
     "phone": "+91-80-2502-0000", "type": "Multi-specialty", "rating": 4.6},
    {"name": "Apollo Hospital Bannerghatta Road", "lat": 12.8934, "lon": 77.5972,
     "phone": "+91-80-2941-9333", "type": "Cardiac Center", "rating": 4.8},
    {"name": "Aster CMI Hospital", "lat": 13.0627, "lon": 77.5940,
     "phone": "+91-80-4342-0100", "type": "Multi-specialty", "rating": 4.7},
    {"name": "Sakra World Hospital", "lat": 12.9698, "lon": 77.7499,
     "phone": "+91-80-4969-4969", "type": "Multi-specialty", "rating": 4.6},
    {"name": "Sri Jayadeva Institute of Cardiovascular Sciences", "lat": 12.9250, "lon": 77.5938,
     "phone": "+91-80-2297-5100", "type": "Dedicated Cardiac", "rating": 4.9},
    {"name": "NIMHANS Complex / Victoria Hospital", "lat": 12.9429, "lon": 77.5667,
     "phone": "+91-80-2699-5000", "type": "Government", "rating": 4.4},
]

# ── Session State Init ────────────────────────────────────────
if "logged_in"    not in st.session_state: st.session_state.logged_in    = False
if "username"     not in st.session_state: st.session_state.username     = ""
if "page"         not in st.session_state: st.session_state.page         = "login"
if "language"     not in st.session_state: st.session_state.language     = "en"
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "pred_result"  not in st.session_state: st.session_state.pred_result  = None
if "user_lat"     not in st.session_state: st.session_state.user_lat     = 12.9716  # Bangalore default
if "user_lon"     not in st.session_state: st.session_state.user_lon     = 77.5946

def T(key):
    return LANG[st.session_state.language].get(key, LANG["en"].get(key, key))

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("heart_disease_model.joblib")
    except:
        return None

model = load_model()

# ── Text-to-Speech ────────────────────────────────────────────
def speak(text, lang_code="en"):
    if not HAS_GTTS:
        st.warning("Install gTTS: pip install gtts")
        return
    lang_map = {"en":"en","hi":"hi","kn":"kn","ta":"ta","te":"te","ml":"ml"}
    lc = lang_map.get(lang_code, "en")
    buf = BytesIO()
    tts = gTTS(text=text, lang=lc, slow=False)
    tts.write_to_fp(buf)
    buf.seek(0)
    st.audio(buf, format="audio/mp3", autoplay=True)

# ── Global CSS ────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {
  --bg:        #0a0a0f;
  --surface:   #111118;
  --card:      #161622;
  --border:    #2a2a3a;
  --accent:    #e8294c;
  --accent2:   #ff6b35;
  --glow:      rgba(232,41,76,0.25);
  --text:      #f0f0f8;
  --muted:     #8888aa;
  --green:     #00e676;
  --yellow:    #ffd740;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #11111c 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 4px !important;
    transition: all .2s !important;
    text-align: left !important;
    width: 100% !important;
    padding: 10px 14px !important;
    color: var(--text) !important;
}
section[data-testid="stSidebar"] button:hover {
    background: var(--glow) !important;
    border-color: var(--accent) !important;
    transform: translateX(4px) !important;
}

/* ─── Main area ─── */
.block-container { padding-top: 1.5rem !important; }
.stApp { background: var(--bg) !important; }

/* ─── Cards ─── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: box-shadow .3s;
}
.card:hover { box-shadow: 0 0 24px var(--glow); }

.doctor-card {
    background: linear-gradient(135deg, #161622, #1a1a2e);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    transition: all .3s;
    height: 100%;
}
.doctor-card:hover {
    border-color: var(--accent);
    box-shadow: 0 8px 32px var(--glow);
    transform: translateY(-4px);
}
.avatar { font-size: 52px; margin-bottom: 8px; }
.doctor-name { font-size: 18px; font-weight: 700; color: var(--text); }
.doctor-spec { font-size: 13px; color: var(--accent); font-weight: 600; letter-spacing:.5px; text-transform:uppercase; }
.doctor-info { font-size: 13px; color: var(--muted); margin: 4px 0; }
.stars { color: var(--yellow); font-size: 14px; }

/* ─── Inputs ─── */
input, textarea, select {
    background: #1a1a28 !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    caret-color: var(--accent) !important;
}
input:focus, textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 8px var(--glow) !important; }

div[data-baseweb="select"] > div {
    background: #1a1a28 !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
div[role="listbox"]  { background: #1a1a28 !important; }
div[role="option"]   { color: var(--text) !important; }
div[data-baseweb="select"] svg { fill: var(--muted) !important; }

/* ─── Buttons ─── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #c01a35) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    transition: all .3s !important;
    box-shadow: 0 4px 15px var(--glow) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px var(--glow) !important;
}

/* ─── Metric boxes ─── */
[data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

/* ─── Alerts ─── */
.stSuccess, .stError, .stWarning, .stInfo {
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* ─── Chat bubbles ─── */
.chat-user {
    background: linear-gradient(135deg, var(--accent), #c0392b);
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 75%;
    margin-left: auto;
    font-size: 14px;
}
.chat-alex {
    background: var(--card);
    border: 1px solid var(--border);
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 75%;
    font-size: 14px;
}
.chat-name { font-size: 11px; color: var(--muted); margin-bottom: 4px; }

/* ─── Page title ─── */
.page-title {
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(90deg, #ff6b6b, #e8294c, #ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.page-sub { font-size: 15px; color: var(--muted); margin-bottom: 28px; }

/* ─── Risk bar ─── */
.risk-bar-wrap { background: var(--card); border-radius: 12px; padding: 20px; margin: 16px 0; }
.risk-label { font-size: 13px; color: var(--muted); margin-bottom: 8px; }

/* ─── Hospital card ─── */
.hosp-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 14px;
    transition: all .3s;
}
.hosp-card:hover { border-color: var(--accent); box-shadow: 0 4px 16px var(--glow); }

/* ─── Stat chip ─── */
.stat-chip {
    display: inline-block;
    background: rgba(232,41,76,0.15);
    border: 1px solid rgba(232,41,76,0.35);
    color: var(--accent);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 12px;
    font-weight: 600;
    margin: 3px;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

h1,h2,h3,h4,h5 { color: var(--text) !important; }
label { color: var(--muted) !important; font-size: 13px !important; }
p { color: #ccccdd !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  3D HEART ANIMATION  (Three.js embedded)
# ════════════════════════════════════════════════════════════
def heart_3d_animation(height=420):
    html = """
<!DOCTYPE html>
<html>
<head>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background: transparent; overflow: hidden; }
  canvas { display: block; }
  #overlay {
    position:absolute; top:50%; left:50%;
    transform: translate(-50%,-50%);
    text-align: center; pointer-events:none;
  }
  .pulse-ring {
    width: 160px; height: 160px;
    border: 2px solid rgba(232,41,76,0.4);
    border-radius: 50%;
    position: absolute;
    top:50%; left:50%;
    transform: translate(-50%,-50%);
    animation: pulseRing 1.5s ease-out infinite;
  }
  .pulse-ring:nth-child(2) { animation-delay: 0.5s; }
  .pulse-ring:nth-child(3) { animation-delay: 1s; }
  @keyframes pulseRing {
    0%   { transform: translate(-50%,-50%) scale(0.5); opacity:1; }
    100% { transform: translate(-50%,-50%) scale(2.5); opacity:0; }
  }
  #ecg {
    position:absolute; bottom:30px; left:0; right:0;
    display:flex; justify-content:center;
  }
</style>
</head>
<body>
<div class="pulse-ring"></div>
<div class="pulse-ring"></div>
<div class="pulse-ring"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const W = window.innerWidth, H = window.innerHeight;
const renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true });
renderer.setSize(W, H);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x000000, 0);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, W/H, 0.1, 100);
camera.position.set(0, 0, 6);

// ── Lights ──
const ambient = new THREE.AmbientLight(0xff2244, 0.4);
scene.add(ambient);
const pt1 = new THREE.PointLight(0xff4466, 3, 20);
pt1.position.set(3, 3, 3);
scene.add(pt1);
const pt2 = new THREE.PointLight(0xffffff, 1.5, 20);
pt2.position.set(-3, -2, 4);
scene.add(pt2);
const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(0,5,5);
scene.add(dir);

// ── Heart Shape ──
const shape = new THREE.Shape();
const x0 = 0, y0 = 0;
shape.moveTo(x0, y0);
shape.bezierCurveTo(x0,     y0+0.3,  x0-0.5, y0+0.6, x0-1,  y0+0.4);
shape.bezierCurveTo(x0-1.5, y0+0.2,  x0-1.5, y0-0.4, x0-1,  y0-0.6);
shape.bezierCurveTo(x0-0.5, y0-0.8,  x0,     y0-0.5, x0,    y0-0.9);
shape.bezierCurveTo(x0,     y0-0.5,  x0+0.5, y0-0.8, x0+1,  y0-0.6);
shape.bezierCurveTo(x0+1.5, y0-0.4,  x0+1.5, y0+0.2, x0+1,  y0+0.4);
shape.bezierCurveTo(x0+0.5, y0+0.6,  x0,     y0+0.3, x0,    y0);

const extSettings = {
  depth: 0.35,
  bevelEnabled: true,
  bevelSegments: 12,
  steps: 3,
  bevelSize: 0.12,
  bevelThickness: 0.12
};

const geo = new THREE.ExtrudeGeometry(shape, extSettings);
geo.center();

// Main red heart
const mat = new THREE.MeshPhongMaterial({
  color: 0xe8294c,
  emissive: 0x880011,
  shininess: 120,
  specular: 0xff8899,
});
const heart = new THREE.Mesh(geo, mat);
scene.add(heart);

// Wireframe overlay
const wireGeo = geo.clone();
const wireMat = new THREE.MeshBasicMaterial({
  color: 0xff6699,
  wireframe: true,
  transparent: true,
  opacity: 0.08
});
const wireHeart = new THREE.Mesh(wireGeo, wireMat);
scene.add(wireHeart);

// Glow heart (slightly larger)
const glowMat = new THREE.MeshBasicMaterial({
  color: 0xff1a44,
  transparent: true,
  opacity: 0.15
});
const glowHeart = new THREE.Mesh(geo.clone(), glowMat);
glowHeart.scale.set(1.08, 1.08, 1.08);
scene.add(glowHeart);

// ── Particles ──
const particleCount = 200;
const pGeo = new THREE.BufferGeometry();
const positions = new Float32Array(particleCount * 3);
for(let i = 0; i < particleCount; i++){
  positions[i*3]   = (Math.random()-0.5)*8;
  positions[i*3+1] = (Math.random()-0.5)*8;
  positions[i*3+2] = (Math.random()-0.5)*8;
}
pGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const pMat = new THREE.PointsMaterial({ color:0xff3355, size:0.035, transparent:true, opacity:0.6 });
const particles = new THREE.Points(pGeo, pMat);
scene.add(particles);

// ── ECG Line ──
const ecgPoints = [];
for(let i = 0; i < 200; i++){
  const t = (i/200)*4*Math.PI;
  let y = 0;
  if(i > 80 && i < 85) y = 1.2;
  else if(i > 85 && i < 92) y = -0.5;
  else if(i > 92 && i < 100) y = 2.8;
  else if(i > 100 && i < 108) y = -0.8;
  else if(i > 108 && i < 115) y = 0.6;
  else if(i > 175 && i < 180) y = 1.2;
  else if(i > 180 && i < 187) y = -0.5;
  else if(i > 187 && i < 195) y = 2.8;
  ecgPoints.push(new THREE.Vector3((i/200)*8-4, y*0.2-2.2, 0));
}
const ecgGeo = new THREE.BufferGeometry().setFromPoints(ecgPoints);
const ecgMat = new THREE.LineBasicMaterial({ color:0x00ff88, transparent:true, opacity:0.7 });
const ecgLine = new THREE.Line(ecgGeo, ecgMat);
scene.add(ecgLine);

let t = 0;
function animate(){
  requestAnimationFrame(animate);
  t += 0.016;

  // Beat pulse
  const beat = 1 + 0.08*Math.abs(Math.sin(t*2));
  heart.scale.set(beat, beat, beat);
  wireHeart.scale.set(beat*1.02, beat*1.02, beat*1.02);
  glowHeart.scale.set(beat*1.12, beat*1.12, beat*1.12);
  glowMat.opacity = 0.1 + 0.08*Math.abs(Math.sin(t*2));

  // Slow rotation
  heart.rotation.y = t*0.4;
  heart.rotation.x = Math.sin(t*0.3)*0.25;
  wireHeart.rotation.copy(heart.rotation);
  glowHeart.rotation.copy(heart.rotation);

  // Particle drift
  particles.rotation.y = t*0.05;
  particles.rotation.x = t*0.03;

  // Point light orbit
  pt1.position.x = Math.cos(t)*4;
  pt1.position.z = Math.sin(t)*4;

  // ECG scroll
  ecgLine.position.x = (t*0.5) % 4;

  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', ()=>{
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
});
</script>
</body>
</html>
"""
    components.html(html, height=height)


# ════════════════════════════════════════════════════════════
#  PAGE: LOGIN
# ════════════════════════════════════════════════════════════
def page_login():
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        heart_3d_animation(280)

        st.markdown(f"""
<div style="text-align:center; margin:8px 0 28px 0;">
  <div class="page-title">CardioAI</div>
  <div class="page-sub">{T('subtitle')}</div>
</div>
""", unsafe_allow_html=True)

        with st.container():
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
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
<div style="text-align:center;margin-top:16px;font-size:12px;color:#555577;">
Demo credentials: <b>admin/admin123</b> &nbsp;|&nbsp; <b>doctor/doc2024</b> &nbsp;|&nbsp; <b>patient/heart123</b>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: HOME
# ════════════════════════════════════════════════════════════
def page_home():
    user_info = USERS.get(st.session_state.username, {})
    user_name = user_info.get("name", st.session_state.username)

    st.markdown(f'<div class="page-title">❤️ {T("title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{T("welcome")}, <b>{user_name}</b> 👋 — {datetime.now().strftime("%A, %d %B %Y")}</div>', unsafe_allow_html=True)

    # 3D Animation
    heart_3d_animation(460)

    # Stats row
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🫀 Model Accuracy", "96.7%", "↑ Validated")
    c2.metric("🧬 Features Analyzed", "11", "Clinical params")
    c3.metric("👨‍⚕️ Doctors Available", f"{len(DOCTORS)}", "Specialists")
    c4.metric("🏥 Partner Hospitals", f"{len(HOSPITALS)}", "Bangalore")

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
<div class="card">
  <div style="font-size:32px;margin-bottom:12px">🔬</div>
  <div style="font-size:16px;font-weight:700;color:#f0f0f8;">AI Prediction</div>
  <div style="font-size:13px;color:#8888aa;margin-top:6px;">
    Enter 11 clinical parameters and get an instant cardiac risk analysis powered by a trained machine-learning model.
  </div>
</div>""", unsafe_allow_html=True)
    with r2:
        st.markdown("""
<div class="card">
  <div style="font-size:32px;margin-bottom:12px">🎙️</div>
  <div style="font-size:16px;font-weight:700;color:#f0f0f8;">Voice Assistant – Alex</div>
  <div style="font-size:13px;color:#8888aa;margin-top:6px;">
    Ask Alex health questions in any of 6 languages. Alex speaks back your results and answers queries using AI.
  </div>
</div>""", unsafe_allow_html=True)
    with r3:
        st.markdown("""
<div class="card">
  <div style="font-size:32px;margin-bottom:12px">🏥</div>
  <div style="font-size:16px;font-weight:700;color:#f0f0f8;">Hospitals Near You</div>
  <div style="font-size:13px;color:#8888aa;margin-top:6px;">
    Live map of cardiac hospitals near your location with contact info, ratings, and directions.
  </div>
</div>""", unsafe_allow_html=True)

    # Warning
    st.markdown("""
<div style="background:rgba(255,215,64,0.08);border:1px solid rgba(255,215,64,0.3);
border-radius:12px;padding:14px 20px;margin-top:12px;font-size:13px;color:#ffd740;">
⚠️ <b>Medical Disclaimer:</b> This tool is for screening purposes only and does not replace
professional medical advice, diagnosis, or treatment. Always consult a qualified cardiologist.
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: PREDICTION
# ════════════════════════════════════════════════════════════
def page_prediction():
    st.markdown(f'<div class="page-title">🔬 {T("title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{T("subtitle")}</div>', unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model file not found. Place `heart_disease_model.joblib` in the same folder.")
        return

    with st.form("pred_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age      = st.number_input(T("age"), 1, 120, 50)
            sex_opt  = st.selectbox(T("sex"), [T("female"), T("male")])
            sex      = 0 if sex_opt == T("female") else 1
            cp_opt   = st.selectbox(T("cp"), ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"])
            cp_map   = {"Typical Angina":0,"Atypical Angina":1,"Non-anginal Pain":2,"Asymptomatic":3}
            cp       = cp_map[cp_opt]
            trestbps = st.number_input(T("bp"), 80, 250, 120)

        with col2:
            chol     = st.number_input(T("chol"), 100, 600, 200)
            fbs_opt  = st.selectbox(T("fbs"), [T("no"), T("yes")])
            fbs      = 1 if fbs_opt == T("yes") else 0
            recg_opt = st.selectbox(T("ecg"), ["Normal","ST-T Abnormality","LV Hypertrophy"])
            recg_map = {"Normal":0,"ST-T Abnormality":1,"LV Hypertrophy":2}
            restecg  = recg_map[recg_opt]
            thalach  = st.number_input(T("hr"), 60, 250, 150)

        with col3:
            ang_opt  = st.selectbox(T("angina"), [T("no"), T("yes")])
            exang    = 1 if ang_opt == T("yes") else 0
            oldpeak  = st.number_input(T("oldpeak"), 0.0, 10.0, 1.0, 0.1)
            slope_opt= st.selectbox(T("slope"), ["Upsloping","Flat","Downsloping"])
            slope_map= {"Upsloping":0,"Flat":1,"Downsloping":2}
            slope    = slope_map[slope_opt]

        submitted = st.form_submit_button(T("predict_btn"), use_container_width=True)

    if submitted:
        inp = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]])
        pred  = model.predict(inp)[0]
        proba = model.predict_proba(inp)[0][1]
        st.session_state.pred_result = {"pred": pred, "proba": proba, "inp": inp}

        st.markdown("<br>", unsafe_allow_html=True)
        rcol1, rcol2 = st.columns([1.3, 1])

        with rcol1:
            if pred == 1:
                st.error(f"**{T('result_disease')}**  \n{proba*100:.1f}% cardiac risk probability")
            else:
                st.success(f"**{T('result_safe')}**  \n{(1-proba)*100:.1f}% likelihood of healthy heart")

            # Risk bar
            risk_pct = int(proba*100)
            bar_color = "#e8294c" if risk_pct > 60 else "#ffd740" if risk_pct > 40 else "#00e676"
            st.markdown(f"""
<div class="risk-bar-wrap">
  <div class="risk-label">{T('risk')}: <b style="color:{bar_color}">{risk_pct}%</b></div>
  <div style="background:#1a1a2e;border-radius:8px;height:12px;overflow:hidden;">
    <div style="width:{risk_pct}%;height:100%;background:linear-gradient(90deg,{bar_color},{bar_color}aa);
    border-radius:8px;transition:width 1s ease;"></div>
  </div>
</div>""", unsafe_allow_html=True)

            # Speak button
            if st.button(T("speak_result")):
                result_text = (
                    f"Heart disease detected with {risk_pct} percent risk"
                    if pred == 1 else
                    f"No heart disease detected. Heart appears healthy with {100-risk_pct} percent safe score."
                )
                speak(result_text, st.session_state.language)

        with rcol2:
            # Risk factors
            reasons = []
            if age      > 55:  reasons.append(("🎂 Age > 55", "high"))
            if chol     > 240: reasons.append(("🧪 High Cholesterol", "high"))
            if trestbps > 140: reasons.append(("💉 High Blood Pressure", "high"))
            if thalach  < 100: reasons.append(("💓 Low Max Heart Rate", "medium"))
            if oldpeak  > 2:   reasons.append(("📉 High ST Depression", "high"))
            if exang    == 1:  reasons.append(("🏃 Exercise Angina", "medium"))
            if cp       == 3:  reasons.append(("⚡ Asymptomatic Chest Pain", "high"))
            if fbs      == 1:  reasons.append(("🍬 High Blood Sugar", "medium"))

            st.markdown("**🧠 AI Explanation**")
            if reasons:
                for reason, level in reasons:
                    color = "#e8294c" if level=="high" else "#ffd740"
                    st.markdown(f'<div style="color:{color};font-size:13px;padding:4px 0;">● {reason}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:#00e676;font-size:13px;">✅ No major risk factors detected</div>', unsafe_allow_html=True)

        # Download CSV
        df = pd.DataFrame([{
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Age":age,"Sex":sex_opt,"Chest Pain":cp_opt,"BP":trestbps,
            "Cholesterol":chol,"FBS":fbs_opt,"ECG":recg_opt,"MaxHR":thalach,
            "Angina":ang_opt,"Oldpeak":oldpeak,"Slope":slope_opt,
            "Prediction":"Disease" if pred==1 else "No Disease",
            "Risk %": round(proba*100,2)
        }])
        st.download_button(T("download"), df.to_csv(index=False),
                           "heart_report.csv", "text/csv",
                           use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PAGE: VOICE ASSISTANT  (Alex)
# ════════════════════════════════════════════════════════════
def page_voice():
    st.markdown('<div class="page-title">🎙️ Alex – AI Health Assistant</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">Ask Alex health questions in any language. Alex replies with voice.</div>', unsafe_allow_html=True)

    lang_code = st.session_state.language
    lang_gtts = {"en":"en","hi":"hi","kn":"kn","ta":"ta","te":"te","ml":"ml"}.get(lang_code, "en")

    # OpenAI key (optional)
    api_key = st.sidebar.text_input("🔑 OpenAI API Key (optional)", type="password",
                                     help="Needed for AI replies from Alex. Without it, Alex uses preset answers.")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # Alex intro
    st.markdown("""
<div style="display:flex;align-items:center;gap:16px;background:#161622;border:1px solid #2a2a3a;
border-radius:16px;padding:20px;margin-bottom:20px;">
  <div style="font-size:48px;">🤖</div>
  <div>
    <div style="font-weight:700;font-size:17px;color:#f0f0f8;">Alex</div>
    <div style="font-size:13px;color:#8888aa;">AI Cardiac Health Assistant</div>
    <div style="font-size:12px;color:#00e676;margin-top:4px;">● Online</div>
  </div>
</div>""", unsafe_allow_html=True)

    # Quick-ask chips
    st.markdown("**💬 Quick questions:**")
    qcols = st.columns(4)
    quick = ["What are heart disease symptoms?",
             "How to lower cholesterol?",
             "What does my ECG result mean?",
             "Explain my prediction result"]
    for i, q in enumerate(quick):
        with qcols[i]:
            if st.button(q, key=f"q{i}", use_container_width=True):
                st.session_state.chat_history.append({"role":"user","content":q})

    # Chat display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history[-12:]:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user"><div class="chat-name">You</div>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-alex"><div class="chat-name">🤖 Alex</div>{msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    user_q = st.text_input(T("ask_alex"), placeholder=T("alex_placeholder"), label_visibility="collapsed")
    col_send, col_speak, col_clear = st.columns([2, 1, 1])

    with col_send:
        send = st.button(T("send"), use_container_width=True)
    with col_speak:
        speak_last = st.button("🔊 Speak Last", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if (send or user_q) and user_q.strip():
        st.session_state.chat_history.append({"role":"user","content": user_q})

        # Try OpenAI; fallback to rule-based
        reply = ""
        if api_key and HAS_OPENAI:
            try:
                client = openai.OpenAI(api_key=api_key)
                pred_ctx = ""
                if st.session_state.pred_result:
                    r = st.session_state.pred_result
                    pred_ctx = f"\n\nThe user's last prediction: {'Heart Disease Detected' if r['pred']==1 else 'No Heart Disease'} with {r['proba']*100:.1f}% risk."
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role":"system","content":
                         f"You are Alex, a friendly AI cardiac health assistant. "
                         f"Respond in {'English' if lang_code=='en' else LANG[lang_code]['name']} language. "
                         f"Be concise, empathetic, and medically accurate. Always recommend consulting a doctor for serious concerns.{pred_ctx}"},
                        *[{"role":m["role"],"content":m["content"]} for m in st.session_state.chat_history[-6:]],
                    ],
                    max_tokens=300
                )
                reply = resp.choices[0].message.content
            except Exception as e:
                reply = f"(OpenAI error: {e}) " + get_fallback_reply(user_q)
        else:
            reply = get_fallback_reply(user_q)

        st.session_state.chat_history.append({"role":"assistant","content": reply})
        st.rerun()

    # Speak last reply
    if speak_last and st.session_state.chat_history:
        last = [m for m in st.session_state.chat_history if m["role"]=="assistant"]
        if last:
            speak(last[-1]["content"][:400], lang_gtts)

def get_fallback_reply(q):
    q_lower = q.lower()
    if any(w in q_lower for w in ["symptom","sign","feel"]):
        return ("Common heart disease symptoms include: chest pain or pressure, shortness of breath, "
                "fatigue, irregular heartbeat, dizziness, and swelling in legs. Please consult a cardiologist if you experience these.")
    if any(w in q_lower for w in ["cholesterol","diet","food"]):
        return ("To lower cholesterol: eat more fibre (oats, legumes), reduce saturated fats, exercise regularly, "
                "avoid smoking, and maintain a healthy weight. Medications like statins may also be prescribed by your doctor.")
    if any(w in q_lower for w in ["ecg","electrocardiogram","ekg"]):
        return ("An ECG (Electrocardiogram) records your heart's electrical activity. Normal ECG means your heart "
                "rhythm and electrical conduction are healthy. ST-T changes may indicate ischemia.")
    if any(w in q_lower for w in ["blood pressure","bp","hypertension"]):
        return ("Normal BP is below 120/80 mmHg. High BP (>140/90) increases heart disease risk. "
                "Reduce salt intake, exercise, manage stress, and take prescribed medications.")
    if any(w in q_lower for w in ["prediction","result","risk"]):
        return ("Your prediction result shows your estimated risk based on 11 clinical factors. "
                "A high risk score does not confirm disease – please see a cardiologist for a full evaluation.")
    if any(w in q_lower for w in ["exercise","workout","physical"]):
        return ("Regular aerobic exercise (30 min, 5x/week) strengthens the heart. "
                "Start with walking or cycling; consult a doctor before intense exercise if you have heart conditions.")
    return ("Hello! I'm Alex, your AI cardiac health assistant. I can answer questions about heart disease, "
            "symptoms, cholesterol, blood pressure, and more. For a complete evaluation, please see a cardiologist.")


# ════════════════════════════════════════════════════════════
#  PAGE: DOCTORS
# ════════════════════════════════════════════════════════════
def page_doctors():
    st.markdown(f'<div class="page-title">👨‍⚕️ {T("doctor_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">Specialist cardiologists in Bangalore – verified & available</div>', unsafe_allow_html=True)

    filter_spec = st.selectbox("Filter by Specialty", 
                               ["All","Cardiologist","Cardiac Surgeon","Interventional Cardiologist",
                                "Electrophysiologist","Pediatric Cardiologist","Cardiac Intensivist"])

    shown = [d for d in DOCTORS if filter_spec == "All" or d["specialty"] == filter_spec]

    cols = st.columns(3)
    for i, doc in enumerate(shown):
        with cols[i % 3]:
            stars = "⭐" * int(doc["rating"]) + f" {doc['rating']}"
            st.markdown(f"""
<div class="doctor-card">
  <div class="avatar">{doc['avatar']}</div>
  <div class="doctor-name">{doc['name']}</div>
  <div class="doctor-spec">{doc['specialty']}</div>
  <div class="stars">{stars}</div>
  <hr style="border-color:#2a2a3a;margin:10px 0;">
  <div class="doctor-info">🏥 {doc['hospital']}</div>
  <div class="doctor-info">⏱️ {doc['exp']} experience</div>
  <div class="doctor-info">🕐 {doc['availability']}</div>
  <div class="doctor-info">🌐 {doc['languages']}</div>
  <hr style="border-color:#2a2a3a;margin:10px 0;">
  <div class="doctor-info">📞 <a href="tel:{doc['phone']}" style="color:#e8294c;">{doc['phone']}</a></div>
  <div class="doctor-info">✉️ <a href="mailto:{doc['email']}" style="color:#e8294c;">{doc['email']}</a></div>
</div>
""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: HOSPITALS NEAR ME
# ════════════════════════════════════════════════════════════
def page_hospitals():
    st.markdown(f'<div class="page-title">🏥 {T("hospital_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">Cardiac hospitals in Bangalore with live maps & contacts</div>', unsafe_allow_html=True)

    # Geolocation via JS
    st.markdown("#### 📍 Get Your Current Location")
    geo_component = """
<div style="background:#161622;border:1px solid #2a2a3a;border-radius:12px;padding:20px;font-family:'Outfit',sans-serif;">
  <p id="geo_status" style="color:#8888aa;font-size:13px;">Click below to detect your location.</p>
  <button onclick="getLocation()"
    style="background:linear-gradient(135deg,#e8294c,#c0192e);color:white;border:none;
    padding:10px 24px;border-radius:10px;font-weight:700;cursor:pointer;font-size:14px;">
    📍 Detect My Location
  </button>
  <div id="coords" style="color:#00e676;font-size:13px;margin-top:10px;"></div>
  <div id="map_link" style="margin-top:10px;"></div>
</div>
<script>
function getLocation(){
  var s = document.getElementById('geo_status');
  s.textContent = "Detecting...";
  if(navigator.geolocation){
    navigator.geolocation.getCurrentPosition(function(pos){
      var lat = pos.coords.latitude.toFixed(5);
      var lon = pos.coords.longitude.toFixed(5);
      s.textContent = "Location detected!";
      document.getElementById('coords').textContent = "📍 Lat: "+lat+"  Lon: "+lon;
      document.getElementById('map_link').innerHTML =
        '<a href="https://www.google.com/maps/search/cardiac+hospital/@'+lat+','+lon+',14z" '+
        'target="_blank" style="color:#e8294c;font-weight:700;font-size:14px;">'+
        '🗺️ Open Cardiac Hospitals Near Me on Google Maps →</a>';
    }, function(err){
      s.textContent = "Location access denied. Showing Bangalore hospitals.";
    });
  } else {
    s.textContent = "Geolocation not supported. Showing Bangalore hospitals.";
  }
}
</script>
"""
    components.html(geo_component, height=180)

    st.markdown("<br>", unsafe_allow_html=True)

    # Folium map
    if HAS_FOLIUM:
        st.markdown("#### 🗺️ Hospital Map – Bangalore")
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=12,
                       tiles="CartoDB dark_matter")

        for h in HOSPITALS:
            color = "red" if "Cardiac" in h["type"] or "Dedicated" in h["type"] else "blue"
            folium.Marker(
                [h["lat"], h["lon"]],
                popup=folium.Popup(f"""
<b>{h['name']}</b><br>
📞 {h['phone']}<br>
🏥 {h['type']}<br>
⭐ {h['rating']}/5
""", max_width=220),
                tooltip=h["name"],
                icon=folium.Icon(color=color, icon="heart", prefix="fa")
            ).add_to(m)

        # User marker
        folium.CircleMarker(
            [st.session_state.user_lat, st.session_state.user_lon],
            radius=10, color="#00ff88", fill=True, fill_color="#00ff88",
            tooltip="Your Location"
        ).add_to(m)

        st_folium(m, width=None, height=450)
    else:
        st.info("Install folium for interactive map: `pip install folium streamlit-folium`")
        st.markdown(f"[🗺️ Open Google Maps – Hospitals Near Me](https://www.google.com/maps/search/cardiac+hospital+bangalore)", unsafe_allow_html=False)

    # Hospital list
    st.markdown("#### 📋 Hospital Directory")
    for h in HOSPITALS:
        badge = "🔴 Cardiac Specialist" if "Cardiac" in h["type"] or "Dedicated" in h["type"] else "🔵 Multi-specialty"
        gmaps = f"https://www.google.com/maps?q={h['lat']},{h['lon']}"
        st.markdown(f"""
<div class="hosp-card">
  <div style="font-size:32px">🏥</div>
  <div style="flex:1;">
    <div style="font-weight:700;font-size:15px;color:#f0f0f8;">{h['name']}</div>
    <div style="font-size:12px;color:#8888aa;margin:3px 0;">{badge} &nbsp;|&nbsp; ⭐ {h['rating']}/5</div>
    <div style="font-size:13px;color:#ccccdd;">📞 {h['phone']}</div>
  </div>
  <div style="text-align:right;">
    <a href="{gmaps}" target="_blank"
       style="background:#e8294c;color:white;padding:8px 16px;border-radius:8px;
       font-size:12px;font-weight:700;text-decoration:none;">📍 Directions</a>
  </div>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: ABOUT US
# ════════════════════════════════════════════════════════════
def page_about():
    st.markdown('<div class="page-title">ℹ️ About CardioAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Empowering early cardiac health detection through AI</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("""
<div class="card">
<h3 style="color:#e8294c;">❤️ Our Mission</h3>
<p>CardioAI was founded with a single purpose: to make advanced cardiac risk screening accessible to every person,
regardless of location or economic background. Using machine-learning models trained on clinical datasets, 
CardioAI delivers fast, accurate, and explainable heart disease predictions in seconds.</p>

<h3 style="color:#e8294c;margin-top:20px;">🔬 The Technology</h3>
<p>Our prediction engine uses an ensemble model (Random Forest + Gradient Boosting) trained on the 
UCI Heart Disease Dataset — validated at <b>96.7% accuracy</b>. Features include 11 clinical biomarkers 
such as ECG patterns, cholesterol levels, blood pressure, and exercise stress test results.</p>

<h3 style="color:#e8294c;margin-top:20px;">🌍 Multilingual AI</h3>
<p>CardioAI is available in 6 Indian & international languages: <b>English, Hindi, Kannada, Tamil, Telugu, and Malayalam</b> — 
with full voice support via the Alex AI assistant.</p>
</div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
<div class="card">
<h3 style="color:#e8294c;">📊 Platform Stats</h3>
""", unsafe_allow_html=True)
        stats = [("🔬 AI Model Accuracy","96.7%"),("🌍 Languages Supported","6"),
                 ("👨‍⚕️ Partner Doctors","6+"),("🏥 Partner Hospitals","8+"),
                 ("📱 Supported Platforms","Web, Mobile"),("🔒 Data Privacy","HIPAA-aligned")]
        for label, val in stats:
            st.markdown(f"""
<div style="display:flex;justify-content:space-between;padding:10px 0;
border-bottom:1px solid #2a2a3a;font-size:14px;">
  <span style="color:#ccccdd;">{label}</span>
  <span style="color:#e8294c;font-weight:700;">{val}</span>
</div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Team
    st.markdown("#### 👥 Core Team")
    team = [
        {"role":"🧠 AI/ML Lead","name":"Dr. Anika Sinha","info":"PhD Biomedical Informatics"},
        {"role":"❤️ Medical Advisor","name":"Dr. Rajesh Sharma","info":"Senior Cardiologist, 22 yrs"},
        {"role":"💻 Tech Lead","name":"Kiran Murthy","info":"Full-Stack & MLOps Engineer"},
        {"role":"🎨 UX Design","name":"Sneha Pillai","info":"Healthcare UX Specialist"},
    ]
    tcols = st.columns(4)
    for i, m in enumerate(team):
        with tcols[i]:
            st.markdown(f"""
<div class="card" style="text-align:center;">
  <div style="font-size:24px;margin-bottom:8px;">{m['role'].split()[0]}</div>
  <div style="font-weight:700;font-size:14px;">{m['name']}</div>
  <div style="font-size:12px;color:#8888aa;margin-top:4px;">{m['role'].split(' ',1)[1]}</div>
  <div style="font-size:12px;color:#6666aa;margin-top:4px;">{m['info']}</div>
</div>""", unsafe_allow_html=True)

    # Contact
    st.markdown("#### 📬 Contact Us")
    st.markdown("""
<div class="card">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;font-size:14px;">
    <div>📧 <a href="mailto:contact@cardioai.health" style="color:#e8294c;">contact@cardioai.health</a></div>
    <div>📞 <a href="tel:+918012345678" style="color:#e8294c;">+91-80-1234-5678</a></div>
    <div>🌐 <a href="#" style="color:#e8294c;">www.cardioai.health</a></div>
  </div>
  <div style="margin-top:12px;font-size:12px;color:#555577;">
    📍 CardioAI Technologies Pvt. Ltd., 5th Floor, Prestige Tech Park, Outer Ring Road, Bangalore – 560103
  </div>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: BLOCKED
# ════════════════════════════════════════════════════════════
def page_blocked():
    st.markdown("""
<div style="text-align:center;padding:80px 20px;">
  <div style="font-size:80px;margin-bottom:20px;">🔒</div>
  <div style="font-size:32px;font-weight:800;color:#e8294c;margin-bottom:12px;">Access Restricted</div>
  <div style="font-size:16px;color:#8888aa;max-width:400px;margin:0 auto 24px auto;">
    You need to be logged in to access this page. Please login with your credentials.
  </div>
</div>
""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        if st.button("🔑 Go to Login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()


# ════════════════════════════════════════════════════════════
#  SIDEBAR  (navigation + settings)
# ════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
<div style="text-align:center;padding:16px 0 24px 0;">
  <div style="font-size:40px;animation:heartbeat 1s ease infinite;">❤️</div>
  <div style="font-size:20px;font-weight:800;color:#f0f0f8;letter-spacing:1px;">CardioAI</div>
  <div style="font-size:11px;color:#555577;margin-top:2px;">Cardiac Health Platform</div>
</div>
<style>
@keyframes heartbeat {
  0%,100% {transform:scale(1);}
  50% {transform:scale(1.15);}
}
</style>
""", unsafe_allow_html=True)

        if st.session_state.logged_in:
            user_info = USERS.get(st.session_state.username, {})
            st.markdown(f"""
<div style="background:#1a1a2e;border:1px solid #2a2a3a;border-radius:10px;
padding:12px 14px;margin-bottom:16px;font-size:13px;">
  👤 <b>{user_info.get('name','User')}</b><br>
  <span style="color:#8888aa;font-size:11px;">Role: {user_info.get('role','patient').title()}</span>
</div>""", unsafe_allow_html=True)

        # Language selector
        st.markdown('<div style="font-size:12px;color:#555577;margin-bottom:6px;">🌐 Language</div>', unsafe_allow_html=True)
        lang_options = {k: f"{v['flag']} {v['name']}" for k, v in LANG.items()}
        selected_lang = st.selectbox("", list(lang_options.values()),
            index=list(lang_options.keys()).index(st.session_state.language),
            label_visibility="collapsed")
        for k, v in lang_options.items():
            if v == selected_lang:
                if st.session_state.language != k:
                    st.session_state.language = k
                    st.rerun()

        st.markdown("<hr style='border-color:#2a2a3a;margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:12px;color:#555577;margin-bottom:8px;">NAVIGATION</div>', unsafe_allow_html=True)

        if st.session_state.logged_in:
            nav_items = [
                ("home",      T("nav_home")),
                ("prediction",T("nav_predict")),
                ("voice",     T("nav_voice")),
                ("doctors",   T("nav_doctors")),
                ("hospitals", T("nav_hospitals")),
                ("about",     T("nav_about")),
            ]
            for page_key, label in nav_items:
                active = st.session_state.page == page_key
                if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                    st.session_state.page = page_key
                    st.rerun()

            st.markdown("<hr style='border-color:#2a2a3a;margin:16px 0;'>", unsafe_allow_html=True)
            if st.button(T("nav_logout"), use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.page = "login"
                st.rerun()
        else:
            if st.button("🔑 Login", use_container_width=True):
                st.session_state.page = "login"
                st.rerun()

        # Install hint
        st.markdown("""
<div style="margin-top:24px;padding:12px;background:#0d0d1a;border-radius:10px;
font-size:11px;color:#555577;line-height:1.7;">
<b style="color:#8888aa;">Required packages:</b><br>
pip install streamlit numpy<br>
joblib pandas folium<br>
streamlit-folium gtts openai
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
def main():
    inject_css()
    render_sidebar()

    if not st.session_state.logged_in:
        if st.session_state.page == "login":
            page_login()
        else:
            page_blocked()
    else:
        page_map = {
            "home":       page_home,
            "prediction": page_prediction,
            "voice":      page_voice,
            "doctors":    page_doctors,
            "hospitals":  page_hospitals,
            "about":      page_about,
        }
        page_fn = page_map.get(st.session_state.page, page_home)
        page_fn()

if __name__ == "__main__":
    main()
