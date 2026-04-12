import streamlit as st
import numpy as np
import io
import time
import yaml
import pickle
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.translator import WhisperTranslator

# ==============================================================================
# BEEP-FOR-ABUSE | CORE APPLICATION FILE
# ==============================================================================
# This application provides a real-time (simulated via 1s chunks) audio 
# monitoring and censoring system. It uses:
# 1. Faster-Whisper: For high-speed speech-to-text transcription.
# 2. 1D-CNN: A custom-trained model for toxic speech classification.
# 3. Streamlit: For a modern, immersive Discord-style web interface.
# ==============================================================================

# ───────────────────────────────────────────────
# Page Config - Sets up the browser tab and layout
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Beep-for-Abuse | Digital Bouncer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────
# Discord-Clone CSS - Full Immersion
# ───────────────────────────────────────────────
# ───────────────────────────────────────────────
# Discord-Clone CSS - Visual Immersion Layer
# ───────────────────────────────────────────────
# This block injects custom CSS to transform the Streamlit UI into a 
# Discord-inspired interface, including colors, typography, and layout.
st.markdown("""
<style>
/* ============================================ */
/*  GOOGLE FONTS IMPORT                         */
/* ============================================ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ============================================ */
/*  DISCORD CORE PALETTE                        */
/* ============================================ */
:root {
    --dc-bg-tertiary:   #1e1f22;
    --dc-bg-secondary:  #2b2d31;
    --dc-bg-primary:    #313338;
    --dc-bg-modifier:   #383a40;
    --dc-bg-floating:   #111214;
    --dc-text-normal:   #dbdee1;
    --dc-text-muted:    #949ba4;
    --dc-text-link:     #00a8fc;
    --dc-brand:         #5865f2;
    --dc-brand-hover:   #4752c4;
    --dc-brand-active:  #3c45a5;
    --dc-green:         #23a559;
    --dc-green-dark:    #1a7d41;
    --dc-red:           #da373c;
    --dc-yellow:        #f0b232;
    --dc-header-bg:     #2b2d31;
    --dc-channel-hover: #35373c;
    --dc-channel-active:#404249;
    --dc-divider:       #3f4147;
    --dc-scrollbar:     #1a1b1e;
    --dc-scrollbar-thumb:#1a1b1e;
}

/* ============================================ */
/*  GLOBAL RESET & BASE                         */
/* ============================================ */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: var(--dc-bg-primary) !important;
    color: var(--dc-text-normal) !important;
    font-family: 'Inter', 'Segoe UI', 'Noto Sans', sans-serif !important;
}

/* Custom Scrollbar Styling */
* {
    scrollbar-width: thin;
    scrollbar-color: var(--dc-scrollbar-thumb) transparent;
}
*::-webkit-scrollbar { width: 8px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb { background: var(--dc-scrollbar-thumb); border-radius: 4px; }

/* ============================================ */
/*  SIDEBAR ⇒ Discord Server/Channel Panel      */
/* ============================================ */
[data-testid="stSidebar"] {
    background-color: var(--dc-bg-secondary) !important;
    border-right: none !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: var(--dc-bg-secondary) !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: var(--dc-text-muted) !important;
    font-size: 0.82rem;
    letter-spacing: 0.02em;
}

/* ============================================ */
/*  HEADER BAR                                  */
/* ============================================ */
header[data-testid="stHeader"] {
    background-color: var(--dc-bg-primary) !important;
    border-bottom: 1px solid var(--dc-divider) !important;
}
.stDeployButton { display: none !important; }

/* ============================================ */
/*  TYPOGRAPHY                                  */
/* ============================================ */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}
h1 { font-size: 1.65rem !important; letter-spacing: -0.02em; }
h2 { font-size: 1.25rem !important; }
h3 { font-size: 1.05rem !important; }
p, li, span, label, .stMarkdown {
    color: var(--dc-text-normal) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ============================================ */
/*  BUTTONS ⇒ Discord Brand Style               */
/* ============================================ */
.stButton > button {
    background-color: var(--dc-brand) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 10px 20px !important;
    font-size: 0.9rem !important;
    transition: background-color 0.17s ease, transform 0.1s ease !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    background-color: var(--dc-brand-hover) !important;
    transform: translateY(-1px);
}
.stButton > button:active {
    background-color: var(--dc-brand-active) !important;
    transform: translateY(0px);
}

/* Download Button - Green (Discord Success) */
.stDownloadButton > button {
    background-color: var(--dc-green) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 10px 20px !important;
    transition: background-color 0.17s ease !important;
}
.stDownloadButton > button:hover {
    background-color: var(--dc-green-dark) !important;
}

/* ============================================ */
/*  FILE UPLOADER ⇒ Discord Drag Area            */
/* ============================================ */
[data-testid="stFileUploader"] {
    background-color: var(--dc-bg-secondary) !important;
    border: 2px dashed var(--dc-bg-modifier) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--dc-brand) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--dc-text-muted) !important;
}

/* ============================================ */
/*  SLIDER                                      */
/* ============================================ */
.stSlider > div > div > div[role="slider"] {
    background-color: var(--dc-brand) !important;
}
.stSlider [data-baseweb="slider"] div {
    color: var(--dc-text-normal) !important;
}

/* ============================================ */
/*  METRIC CONTAINERS ⇒ Discord Embed Card      */
/* ============================================ */
div[data-testid="metric-container"] {
    background-color: var(--dc-bg-secondary) !important;
    border-radius: 8px !important;
    border-left: 4px solid var(--dc-brand) !important;
    padding: 1rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
}
div[data-testid="metric-container"] label {
    color: var(--dc-text-muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 800 !important;
}

/* ============================================ */
/*  ALERTS / INFO BOXES ⇒ Discord Embeds         */
/* ============================================ */
.stAlert {
    border-radius: 4px !important;
    border-left-width: 4px !important;
}
div[data-testid="stAlert"] {
    background-color: rgba(88, 101, 242, 0.08) !important;
}
.stSuccess {
    background-color: rgba(35, 165, 89, 0.08) !important;
    border-left-color: var(--dc-green) !important;
}

/* ============================================ */
/*  PROGRESS BAR ⇒ Discord Boost Bar             */
/* ============================================ */
.stProgress > div > div > div {
    background-color: var(--dc-brand) !important;
    border-radius: 4px !important;
}
.stProgress > div > div {
    background-color: var(--dc-bg-modifier) !important;
    border-radius: 4px !important;
}

/* ============================================ */
/*  TABS                                         */
/* ============================================ */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 2px solid var(--dc-divider) !important;
    background-color: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--dc-text-muted) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 16px !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
    background-color: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--dc-text-normal) !important;
    border-bottom-color: var(--dc-text-muted) !important;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom-color: var(--dc-brand) !important;
}

/* ============================================ */
/*  EXPANDER ⇒ Discord Collapsible               */
/* ============================================ */
.streamlit-expanderHeader {
    background-color: var(--dc-bg-secondary) !important;
    border-radius: 4px !important;
    color: var(--dc-text-normal) !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent {
    background-color: var(--dc-bg-secondary) !important;
    border: 1px solid var(--dc-divider) !important;
    border-top: none !important;
}

/* ============================================ */
/*  HORIZONTAL RULE (Discord Divider)            */
/* ============================================ */
hr {
    border: none !important;
    border-top: 1px solid var(--dc-divider) !important;
    margin: 1rem 0 !important;
}

/* ============================================ */
/*  AUDIO PLAYER                                 */
/* ============================================ */
audio {
    width: 100% !important;
    border-radius: 8px !important;
    background-color: var(--dc-bg-secondary) !important;
}

/* ============================================ */
/*  CUSTOM DISCORD COMPONENT CLASSES             */
/* ============================================ */

/* Server Icon in Sidebar */
.dc-server-icon {
    width: 48px;
    height: 48px;
    border-radius: 16px;
    background-color: var(--dc-brand);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 6px auto;
    font-size: 1.4rem;
    transition: border-radius 0.2s ease, background-color 0.2s ease;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.dc-server-icon:hover {
    border-radius: 12px;
    background-color: var(--dc-brand-hover);
}

/* Channel Category Header */
.dc-category {
    color: var(--dc-text-muted);
    font-size: 0.7rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 16px 8px 4px 8px;
    display: flex;
    align-items: center;
    gap: 4px;
    cursor: default;
}

/* Channel Item */
.dc-channel {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    border-radius: 4px;
    color: var(--dc-text-muted);
    font-size: 0.9rem;
    font-weight: 500;
    transition: background-color 0.1s ease, color 0.1s ease;
    cursor: pointer;
    margin: 1px 0;
}
.dc-channel:hover {
    background-color: var(--dc-channel-hover);
    color: var(--dc-text-normal);
}
.dc-channel.active {
    background-color: var(--dc-channel-active);
    color: #ffffff;
}

/* Status Indicator (Online Dot) */
.dc-status {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    animation: dc-pulse 2s infinite;
}
.dc-status.online { background-color: var(--dc-green); }
.dc-status.idle    { background-color: var(--dc-yellow); }
.dc-status.dnd     { background-color: var(--dc-red); }

@keyframes dc-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(35, 165, 89, 0.4); }
    50%      { box-shadow: 0 0 0 6px rgba(35, 165, 89, 0); }
}

/* Embed Card (Discord-style) */
.dc-embed {
    background-color: var(--dc-bg-secondary);
    border-left: 4px solid var(--dc-brand);
    border-radius: 4px;
    padding: 12px 16px;
    margin: 8px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.dc-embed.success { border-left-color: var(--dc-green); }
.dc-embed.danger  { border-left-color: var(--dc-red); }
.dc-embed.warning { border-left-color: var(--dc-yellow); }

.dc-embed-title {
    color: #ffffff;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 4px;
}
.dc-embed-desc {
    color: var(--dc-text-normal);
    font-size: 0.85rem;
    line-height: 1.45;
}
.dc-embed-footer {
    color: var(--dc-text-muted);
    font-size: 0.7rem;
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Message Bubble */
.dc-message {
    display: flex;
    gap: 16px;
    padding: 4px 16px;
    margin: 0;
    transition: background-color 0.1s;
}
.dc-message:hover {
    background-color: rgba(0,0,0,0.06);
}
.dc-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    background-color: var(--dc-brand);
}
.dc-msg-header {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 2px;
}
.dc-msg-username {
    font-weight: 600;
    font-size: 0.95rem;
}
.dc-msg-timestamp {
    color: var(--dc-text-muted);
    font-size: 0.7rem;
}
.dc-msg-content {
    color: var(--dc-text-normal);
    font-size: 0.9rem;
    line-height: 1.4;
}

/* Title Bar / Channel Header */
.dc-header-bar {
    background-color: var(--dc-bg-primary);
    border-bottom: 1px solid var(--dc-divider);
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    margin: -1rem -1rem 1rem -1rem;
}
.dc-header-hash {
    color: var(--dc-text-muted);
    font-size: 1.3rem;
    font-weight: 400;
}
.dc-header-name {
    color: #ffffff;
    font-weight: 700;
    font-size: 1rem;
}
.dc-header-divider {
    width: 1px;
    height: 24px;
    background-color: var(--dc-divider);
    margin: 0 8px;
}
.dc-header-topic {
    color: var(--dc-text-muted);
    font-size: 0.8rem;
    flex: 1;
}

/* Transcript Log (Discord Chat Style) */
.dc-chat-log {
    background-color: var(--dc-bg-primary);
    border-radius: 8px;
    padding: 0;
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid var(--dc-divider);
}
.dc-log-entry {
    padding: 6px 16px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
    transition: background-color 0.1s;
}
.dc-log-entry:hover {
    background-color: rgba(0,0,0,0.06);
}
.dc-log-icon {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 0.85rem;
}
.dc-log-icon.toxic { background-color: rgba(218, 55, 60, 0.25); }
.dc-log-icon.clean { background-color: rgba(35, 165, 89, 0.25); }
.dc-log-text {
    flex: 1;
    font-size: 0.88rem;
    line-height: 1.45;
    color: var(--dc-text-normal);
}
.dc-log-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.dc-log-badge.toxic {
    background-color: rgba(218, 55, 60, 0.2);
    color: #f38688;
}
.dc-log-badge.clean {
    background-color: rgba(35, 165, 89, 0.2);
    color: #57f287;
}

/* GitHub Button */
.dc-github-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background-color: var(--dc-bg-primary);
    color: var(--dc-text-normal) !important;
    text-decoration: none !important;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.85rem;
    transition: background-color 0.15s ease, color 0.15s ease;
    border: 1px solid var(--dc-divider);
}
.dc-github-btn:hover {
    background-color: var(--dc-channel-hover);
    color: #ffffff !important;
    text-decoration: none !important;
}
.dc-github-btn svg {
    fill: currentColor;
    width: 18px;
    height: 18px;
}

/* User Panel (Bottom of Sidebar) */
.dc-user-panel {
    background-color: var(--dc-bg-tertiary);
    padding: 8px;
    border-radius: 0;
    margin: 0 -1rem -1rem -1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.dc-user-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--dc-brand), #eb459e);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    position: relative;
}
.dc-user-info {
    flex: 1;
}
.dc-user-name {
    color: #ffffff;
    font-size: 0.82rem;
    font-weight: 600;
    line-height: 1.1;
}
.dc-user-status {
    color: var(--dc-text-muted);
    font-size: 0.7rem;
}

/* Nitro Badge */
.dc-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f47fff, #a958ff);
    color: #fff;
    font-size: 0.55rem;
    font-weight: 800;
    padding: 1px 5px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-left: 4px;
    vertical-align: middle;
}

/* Section Divider with text */
.dc-section-divider {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 16px 0;
    color: var(--dc-text-muted);
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.dc-section-divider::before,
.dc-section-divider::after {
    content: "";
    flex: 1;
    height: 1px;
    background-color: var(--dc-divider);
}

/* Hide default Streamlit footer */
footer { display: none !important; }

/* Column gap tightening */
[data-testid="column"] {
    padding: 0 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────
# Core System Setup & Cache
# ───────────────────────────────────────────────
@st.cache_resource
def load_system():
    """
    Loads all heavy AI components once and caches them for future use.
    - config.yaml: System parameters (toxicity thresholds, etc.)
    - Whisper Engine: Faster-Whisper for high-fidelity STT.
    - CNN Model: 1D Convolutional Neural Network for text classification.
    - Tokenizer: For converting text to sequences compatible with the CNN.
    """
    # 1. Load system configuration
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 2. Initialize the Whisper Engine (Speech-to-Text)
    translator = WhisperTranslator()
    
    # 3. Load CNN Model & Tokenizer (Toxicity Classifier)
    model = tf.keras.models.load_model("assets/models/toxic_cnn.h5")
    with open('assets/models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    return translator, model, tokenizer, config

# Attempt to initialize the system
try:
    translator, model, tokenizer, config = load_system()
    system_online = True
except Exception as e:
    system_online = False
    system_error = str(e)
    
# Audio energy below this level is treated as silence to save computation
SILENCE_THRESHOLD = 0.01

def predict_toxicity(text, mdl, tok):
    """
    Predicts the toxicity score of a given piece of text.
    - text: The transcribed speech text.
    - mdl: The loaded Keras/TF model.
    - tok: The fitted tokenizer.
    Returns: Probability score [0.0 to 1.0].
    """
    sequences = tok.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=50, padding='post')
    prediction = mdl.predict(padded, verbose=0)
    return prediction[0][0]

# Generate standard beep for muting (16000Hz)
def get_beep_array(chunk_size=16000):
    """
    Creates a sine wave array representing a 1000Hz 'censor beep'.
    - chunk_size: Duration in samples (1s @ 16kHz = 16000).
    """
    t = np.linspace(0, chunk_size/16000, chunk_size, False)
    beep = 0.2 * np.sin(1000 * 2 * np.pi * t) # Create sine tone
    return beep.astype(np.float32)


# ───────────────────────────────────────────────
# SIDEBAR ⇒ Discord Server/Channel Style UI
# ───────────────────────────────────────────────
with st.sidebar:
    
    # ── Server Icon ──
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 4px 0;">
        <div class="dc-server-icon" title="Beep-for-Abuse">🛡️</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Server Name / Header ──
    st.markdown("""
    <div style="padding: 10px 12px; border-bottom: 1px solid var(--dc-divider); margin-bottom: 6px;">
        <span style="color: #ffffff; font-weight: 800; font-size: 1rem; letter-spacing: -0.01em;">
            Beep-for-Abuse
        </span>
        <span class="dc-badge">AI</span>
    </div>
    """, unsafe_allow_html=True)
    
    # ── System Status ──
    if system_online:
        st.markdown("""
        <div style="padding: 4px 12px;">
            <span class="dc-status online"></span>
            <span style="color: #57f287; font-size: 0.78rem; font-weight: 600;">Systems Online</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding: 4px 12px;">
            <span class="dc-status dnd"></span>
            <span style="color: #f38688; font-size: 0.78rem; font-weight: 600;">Systems Offline</span>
        </div>
        """, unsafe_allow_html=True)
    
    # ── CHANNELS ──
    st.markdown("""
    <div class="dc-category">▾ AI ENGINE</div>
    <div class="dc-channel active">
        <span style="opacity: 0.5;">#</span> audio-censor
    </div>
    <div class="dc-channel">
        <span style="opacity: 0.5;">#</span> transcript-log
    </div>
    <div class="dc-channel">
        <span style="opacity: 0.5;">🔊</span> voice-monitor
    </div>
    
    <div class="dc-category">▾ SETTINGS</div>
    <div class="dc-channel">
        <span style="opacity: 0.5;">⚙️</span> configuration
    </div>
    <div class="dc-channel">
        <span style="opacity: 0.5;">📊</span> analytics
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Sensitivity Slider ──
    st.markdown("""<div class="dc-category">▾ BOUNCER SENSITIVITY</div>""", unsafe_allow_html=True)
    strictness = st.slider(
        "Threshold",
        0.0, 1.0,
        float(config['toxicity']['threshold']) if system_online else 0.85,
        0.05,
        help="0.0 = Chill Mode  ·  1.0 = Maximum Enforcement"
    )
    
    # Sensitivity label
    if strictness < 0.3:
        s_color, s_label = "#57f287", "🟢 Chill"
    elif strictness < 0.6:
        s_color, s_label = "#f0b232", "🟡 Moderate"
    elif strictness < 0.85:
        s_color, s_label = "#faa61a", "🟠 Strict"
    else:
        s_color, s_label = "#ed4245", "🔴 Maximum"
    
    st.markdown(f"""
    <div style="text-align: center; margin-top: -8px;">
        <span style="color: {s_color}; font-weight: 700; font-size: 0.85rem;">{s_label}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── GitHub Quick Link ──
    st.markdown("""
    <div style="padding: 4px 8px;">
        <a href="https://github.com/mayank-goyal09/beep-for-abuse" target="_blank" class="dc-github-btn" style="width: 100%; justify-content: center;">
            <svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
            View on GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Star Badge ──
    st.markdown("""
    <div style="text-align: center; padding: 6px 0;">
        <a href="https://github.com/mayank-goyal09/beep-for-abuse/stargazers" target="_blank" style="text-decoration: none;">
            <img src="https://img.shields.io/github/stars/mayank-goyal09/beep-for-abuse?style=social" alt="GitHub Stars"/>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # ── User Panel (Bottom) ──
    # Displays the "bot" status at the very bottom of the sidebar
    st.markdown("""
    <div style="flex: 1;"></div>
    <div class="dc-user-panel" style="margin-top: 2rem;">
        <div class="dc-user-avatar">🧠</div>
        <div class="dc-user-info">
            <div class="dc-user-name">Digital Bouncer <span class="dc-badge">BOT</span></div>
            <div class="dc-user-status">Monitoring voice channels</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────
# MAIN CONTENT ⇒ Channel View (Chat & Upload)
# ───────────────────────────────────────────────

# Stop if system is offline
if not system_online:
    st.markdown(f"""
    <div class="dc-embed danger">
        <div class="dc-embed-title">⚠️ System Initialization Failed</div>
        <div class="dc-embed-desc">Could not load AI models from <code>assets/</code>. Please verify model files exist.</div>
        <div class="dc-embed-footer">Error: {system_error}</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Channel Header Bar ──
# Displays the channel name (#audio-censor) and the topic description
st.markdown("""
<div class="dc-header-bar">
    <span class="dc-header-hash">#</span>
    <span class="dc-header-name">audio-censor</span>
    <span class="dc-header-divider"></span>
    <span class="dc-header-topic">Upload audio → Whisper STT → 1D-CNN Classification → Instant Censor Beep</span>
</div>
""", unsafe_allow_html=True)


# ── Welcome Embed / Bot Message ──
# Displays a friendly introduction message from the "Beep-for-Abuse" bot
st.markdown(f"""
<div class="dc-message" style="padding-top: 16px;">
    <div class="dc-avatar">🛡️</div>
    <div>
        <div class="dc-msg-header">
            <span class="dc-msg-username" style="color: var(--dc-brand);">Beep-for-Abuse</span>
            <span class="dc-badge">BOT</span>
            <span class="dc-msg-timestamp">Today at {time.strftime('%I:%M %p')}</span>
        </div>
        <div class="dc-msg-content">
            Welcome to <strong>#audio-censor</strong>! Upload an audio file below to run the 
            <strong>Digital Bouncer</strong> AI pipeline. Toxic speech segments will be automatically 
            replaced with a censor beep. 🔇
        </div>
        <div class="dc-embed" style="margin-top: 10px; max-width: 480px;">
            <div class="dc-embed-title">🧠 How it works</div>
            <div class="dc-embed-desc">
                <strong>1.</strong> Upload your audio file (WAV/MP3, max 2 min)<br/>
                <strong>2.</strong> Faster-Whisper transcribes each 1s chunk<br/>
                <strong>3.</strong> 1D-CNN classifies toxicity in real-time<br/>
                <strong>4.</strong> Toxic segments → Censor beep 🔊
            </div>
            <div class="dc-embed-footer">
                <span class="dc-status online" style="width:8px; height:8px; animation:none;"></span>
                Powered by TensorFlow + CTranslate2
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---") # Visual divider

# ── Upload Section ──
col_upload, col_engine = st.columns([1, 1], gap="medium")

with col_upload:
    st.markdown("""
    <div class="dc-embed" style="border-left-color: var(--dc-brand);">
        <div class="dc-embed-title">📎 Upload Audio File</div>
        <div class="dc-embed-desc">Drag & drop or browse for a WAV, MP3, OGG, or M4A file (max 2 minutes).</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Audio",
        type=['wav', 'mp3', 'ogg', 'm4a'],
        label_visibility="collapsed"
    )

with col_engine:
    st.markdown("""
    <div class="dc-embed" style="border-left-color: var(--dc-yellow);">
        <div class="dc-embed-title">⚙️ Processing Engine</div>
        <div class="dc-embed-desc">Preview your audio and launch the AI pipeline when ready.</div>
    </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("🚀  Run AI Bouncer Pipeline", use_container_width=True):
            st.session_state['process_triggered'] = True
            st.session_state['audio'] = uploaded_file
            st.session_state['strictness'] = strictness
    else:
        st.markdown("""
        <div style="
            border: 2px dashed var(--dc-bg-modifier);
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            color: var(--dc-text-muted);
            font-size: 0.85rem;
        ">
            <div style="font-size: 2rem; margin-bottom: 8px; opacity: 0.5;">🎤</div>
            Waiting for audio input...
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────────
# PROCESSING PIPELINE - The core logic happens here
# ───────────────────────────────────────────────

if st.session_state.get('process_triggered', False):
    
    st.markdown("""<div class="dc-section-divider">AI Investigation Results</div>""", unsafe_allow_html=True)
    
    progress_bar = st.progress(0.0, text="Loading Audio & Re-sampling to 16kHz...")
    
    # 1. Load and Standardize Audio
    try:
        audio, sr = librosa.load(st.session_state['audio'], sr=16000, mono=True)
    except Exception as e:
        st.markdown(f"""
        <div class="dc-embed danger">
            <div class="dc-embed-title">❌ Audio Load Failed</div>
            <div class="dc-embed-desc">{e}</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
        
    duration = librosa.get_duration(y=audio, sr=sr)
    if duration > 120:
        st.markdown("""
        <div class="dc-embed danger">
            <div class="dc-embed-title">⏱️ Duration Limit Exceeded</div>
            <div class="dc-embed-desc">Audio exceeds the 2-minute limit. Please provide shorter audio.</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
        
    # Split audio into 1-second chunks (16000 samples each)
    chunk_size = 16000
    total_chunks = int(np.ceil(len(audio) / chunk_size))
    
    processed_audio = []
    transcripts = []
    
    progress_bar.progress(0.1, text="Audio loaded. Beginning AI Bouncer pipeline...")
    
    # 2. Iterate through 1s chunks
    # We process the audio in 1-second slices to simulate real-time performance.
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(audio))
        chunk = audio[start_idx:end_idx]
        
        # Pad last chunk if it's too short for consistent processing
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
            
        # Efficiently calculate RMS energy to identify silence
        energy = np.sqrt(np.mean(chunk ** 2))
        
        p_text = f"Processing Chunk [{i+1}/{total_chunks}]... "
        
        if energy < SILENCE_THRESHOLD:
            # If silence, just keep the original audio slice
            processed_audio.extend(chunk)
            p_text += "Skipping Silence."
        else:
            # A. Transcribe audio to text (Whisper)
            text = translator.translate_buffer(chunk)
            if not text:
                processed_audio.extend(chunk)
                p_text += "No words found."
            else:
                # B. Predict toxicity of the transcribed text (1D-CNN)
                score = predict_toxicity(text, model, tokenizer)
                safe_score = float(score)
                
                # C. Censor if toxicity exceeds the current threshold
                if safe_score > st.session_state['strictness']:
                    transcripts.append({"text": text, "toxic": True, "score": safe_score})
                    processed_audio.extend(get_beep_array(chunk_size)) # Replace with BEEP
                    p_text += f"🚫 TOXIC [{safe_score:.2f}]"
                else:
                    transcripts.append({"text": text, "toxic": False, "score": safe_score})
                    processed_audio.extend(chunk) # Keep original
                    p_text += f"🟢 CLEAN [{safe_score:.2f}]"
                    
        # Update UI progress for a reactive experience
        progress_bar.progress(0.1 + 0.9 * (i / total_chunks), text=p_text)
     
    progress_bar.progress(1.0, text="✅ Processing Complete!")
    
    # 3. Create Processed Audio File
    output_audio = np.array(processed_audio, dtype=np.float32)
    wav_out = io.BytesIO()
    sf.write(wav_out, output_audio, 16000, format='WAV', subtype='PCM_16')
    wav_out.seek(0)
    
    # 4. Results — Discord Embed Style
    st.balloons() # Success celebration
    
    # ── Summary Stats Row ──
    total_flagged = sum(1 for t in transcripts if t.get('toxic', False))
    total_clean = sum(1 for t in transcripts if not t.get('toxic', True))
    
    stat1, stat2, stat3, stat4 = st.columns(4)
    with stat1:
        st.metric("Duration", f"{duration:.1f}s")
    with stat2:
        st.metric("Chunks", f"{total_chunks}")
    with stat3:
        st.metric("🟢 Clean", f"{total_clean}")
    with stat4:
        st.metric("🔴 Flagged", f"{total_flagged}")
    
    st.markdown("---")
    
    # ── Detailed Results ──
    res_col1, res_col2 = st.columns([1, 1], gap="medium")
    
    with res_col1:
        # User plays the censored audio here
        st.markdown("""
        <div class="dc-embed success">
            <div class="dc-embed-title">✅ Censored Audio Ready</div>
            <div class="dc-embed-desc">All toxic segments have been replaced with a standard censor beep.</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.audio(wav_out, format='audio/wav')
        
        st.download_button(
            label="📥  Download Censored Audio",
            data=wav_out,
            file_name="censored_output.wav",
            mime="audio/wav",
            use_container_width=True
        )
        
    with res_col2:
        # The transcription log showing what was said and if it was flagged
        st.markdown("""
        <div class="dc-embed" style="border-left-color: var(--dc-text-muted);">
            <div class="dc-embed-title">📝 Transcription & Toxicity Log</div>
            <div class="dc-embed-desc">Real-time classification results from the 1D-CNN</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Build Discord-style chat log (HTML/CSS injection)
        log_html = '<div class="dc-chat-log">'
        
        if not transcripts:
            log_html += """
            <div style="padding: 24px; text-align: center; color: var(--dc-text-muted); font-size: 0.85rem;">
                <div style="font-size: 1.5rem; margin-bottom: 6px;">🤫</div>
                No speech segments detected in this audio.
            </div>
            """
        else:
            for idx, entry in enumerate(transcripts):
                if not isinstance(entry, dict):
                    continue
                
                w_text = entry.get('text', '')
                is_toxic = entry.get('toxic', False)
                val = entry.get('score', 0.0)
                
                if is_toxic:
                    icon_class = "toxic"
                    icon_emoji = "🔇"
                    badge_class = "toxic"
                    badge_text = "CENSORED"
                    text_display = f'<span style="text-decoration: line-through; opacity: 0.5;">{w_text}</span>'
                else:
                    icon_class = "clean"
                    icon_emoji = "✅"
                    badge_class = "clean"
                    badge_text = "PASSED"
                    text_display = w_text
                
                log_html += f"""
                <div class="dc-log-entry">
                    <div class="dc-log-icon {icon_class}">{icon_emoji}</div>
                    <div class="dc-log-text">
                        <span class="dc-log-badge {badge_class}">{badge_text}</span>
                        <span style="color: var(--dc-text-muted); font-size: 0.72rem; margin-left: 6px;">
                            Score: {val:.3f}
                        </span>
                        <br/>
                        {text_display}
                    </div>
                </div>
                """
        
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)

    # ── Reset functionality ──
    st.markdown("---")
    if st.button("🔄  Process Another Audio", use_container_width=True):
        st.session_state['process_triggered'] = False
        st.rerun()
