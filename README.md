🌐 AI Multi-Language Translator
A production-ready, open-source AI language translation web application built from scratch using Python. The application utilizes deep learning to translate text across multiple languages seamlessly and features an interactive UI optimized for web deployment.

🚀 Live Demo
[👉 Click here to view the live application on Streamlit Cloud](PASTE_YOUR_STREAMLIT_SHARING_URL_HERE)

🛠️ Core Features & Technical Architecture
* State-of-the-Art Translation Engine: Powered by Meta's **NLLB-200 (No Language Left Behind)** distilled model via the Hugging Face `transformers` ecosystem, capable of cross-lingual translation across hundreds of languages.
* Performance Optimization: Features efficient RAM management utilizing `@st.cache_resource` to persist model weights in system memory, preventing reloading lag and high-concurrency server crashes.
* Lightweight Deployment Strategy: Configured specifically with CPU-targeted dependencies (`torch --index-url`) to remain completely compatible with free-tier cloud environments while processing high-accuracy text inference.
* Responsive User Experience: Designed with a clean, two-column interactive layout in **Streamlit** for quick input evaluation and intuitive output rendering.

📂 Project Structure
```
├── app.py              # Main application source code & Streamlit UI logic
├── requirements.txt    # Python package dependencies & CPU pip wheels
└── packages.txt        # System-level dependencies required for tokenization compiled layers
