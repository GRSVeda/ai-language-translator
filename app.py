import streamlit as dict
import streamlit as st
from transformers import pipeline

# Set up page configuration
st.set_page_config(page_title="AI Translator", page_icon="🌐", layout="wide")

# 1. Cache the AI Model to prevent high memory usage and reload lag
@st.cache_resource
def load_translator():
    # Using Meta's NLLB-200 distilled model
    return pipeline(task="translation", model="facebook/nllb-200-distilled-600M")

# Initialize the pipeline securely
try:
    with st.spinner("Loading AI translation engine... (This takes a moment on first load)"):
        translator = load_translator()
except Exception as e:
    st.error(f"Error loading model: {e}")

# 2. Language to Model Code Map
LANGUAGE_MAP = {
    "English": "eng_Latn",
    "Spanish": "spa_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Hindi": "hin_Deva",
    "Telugu": "tel_Telu",
    "Tamil": "tam_Tamil",
    "Japanese": "jpn_Jpan",
    "Mandarin Chinese": "cmn_Hans"
}

# 3. Main UI Styling
st.title("🌐 AI Language Translator")
st.markdown("Translate text seamlessly using Meta's No Language Left Behind (NLLB-200) deep learning model.")
st.markdown("---")

# Layout columns for side-by-side processing
col1, col2 = st.columns(2)

with col1:
    st.subheader("Source Configuration")
    source_lang = st.selectbox("From Language", list(LANGUAGE_MAP.keys()), index=0)
    input_text = st.text_area("Enter Text", placeholder="Type your text here...", height=200)

with col2:
    st.subheader("Target Configuration")
    target_lang = st.selectbox("To Language", list(LANGUAGE_MAP.keys()), index=1)
    
    # Trigger translation processing
    if st.button("Translate Text", type="primary"):
        if not input_text.strip():
            st.warning("Please input some text to translate.")
        else:
            with st.spinner("Translating..."):
                try:
                    src_code = LANGUAGE_MAP[source_lang]
                    tgt_code = LANGUAGE_MAP[target_lang]
                    
                    # Run inference via pipeline
                    prediction = translator(
                        input_text, 
                        src_lang=src_code, 
                        tgt_lang=tgt_code, 
                        max_length=400
                    )
                    
                    # Display result inside a readable block
                    st.text_area("Translated Output", value=prediction[0]['translation_text'], height=200, disabled=True)
                except Exception as e:
                    st.error(f"Translation failed: {e}")