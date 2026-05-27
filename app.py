import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set up page configuration
st.set_page_config(page_title="AI Translator", page_icon="🌐", layout="wide")

# 1. Load tokenizer and model directly (Bypasses deprecated v5 pipeline tasks)
@st.cache_resource
def load_translator_engine():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

try:
    with st.spinner("Loading AI translation engine... (This takes a moment on first load)"):
        tokenizer, model = load_translator_engine()
except Exception as e:
    st.error(f"Error loading model components: {e}")

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

col1, col2 = st.columns(2)

with col1:
    st.subheader("Source Configuration")
    source_lang = st.selectbox("From Language", list(LANGUAGE_MAP.keys()), index=0)
    input_text = st.text_area("Enter Text", placeholder="Type your text here...", height=200)

with col2:
    st.subheader("Target Configuration")
    target_lang = st.selectbox("To Language", list(LANGUAGE_MAP.keys()), index=1)
    
    if st.button("Translate Text", type="primary"):
        if not input_text.strip():
            st.warning("Please input some text to translate.")
        else:
            with st.spinner("Translating..."):
                try:
                    src_code = LANGUAGE_MAP[source_lang]
                    tgt_code = LANGUAGE_MAP[target_lang]
                    
                    # 4. Generation Inference Strategy
                    # Set the source language code in the tokenizer
                    tokenizer.src_lang = src_code
                    inputs = tokenizer(input_text, return_tensors="pt")
                    
                    # Force generation target to match selected language code
                    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
                    
                    generated_tokens = model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=400
                    )
                    
                    # Decode tokens back into a readable string
                    translation_result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                    
                    st.text_area("Translated Output", value=translation_result, height=200, disabled=True)
                except Exception as e:
                    st.error(f"Translation failed: {e}")
