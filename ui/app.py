"""
RyzenAI-LocalLab - Streamlit UI Entry Point

Modern HomeLab interface with dark theme and real-time monitoring.
"""

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="RyzenAI-LocalLab",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS Theme
# =============================================================================
def load_custom_css():
    """Load custom HomeLab dark theme."""
    st.markdown(
        """
        <style>
        /* =====================================================================
           RyzenAI-LocalLab - HomeLab Dark Theme
           ===================================================================== */
        
        /* Root variables */
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-cyan: #00d4ff;
            --accent-purple: #a855f7;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-orange: #d29922;
            --border-color: #30363d;
            --glass-bg: rgba(22, 27, 34, 0.8);
        }
        
        /* Main background */
        .stApp {
            background: linear-gradient(135deg, var(--bg-primary) 0%, #1a1f2e 100%);
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: var(--bg-secondary) !important;
            border-right: 1px solid var(--border-color) !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: var(--text-primary) !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        h1 {
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700 !important;
        }
        
        /* Cards / Containers */
        .stMetric {
            background: var(--glass-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            backdrop-filter: blur(10px) !important;
        }
        
        .stMetric label {
            color: var(--text-secondary) !important;
        }
        
        .stMetric [data-testid="stMetricValue"] {
            color: var(--accent-cyan) !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4) !important;
        }
        
        /* Secondary buttons */
        .stButton > button[kind="secondary"] {
            background: var(--bg-tertiary) !important;
            border: 1px solid var(--border-color) !important;
        }
        
        /* Text input */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: var(--accent-cyan) !important;
            box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.2) !important;
        }
        
        /* Select boxes */
        .stSelectbox > div > div {
            background: var(--bg-tertiary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
        }
        
        /* Sliders */
        .stSlider > div > div > div > div {
            background: var(--accent-cyan) !important;
        }
        
        /* Progress bars */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple)) !important;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            background: var(--bg-tertiary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
        }
        
        /* Dataframes */
        .stDataFrame {
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
        }
        
        /* Chat messages */
        .stChatMessage {
            background: var(--bg-tertiary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
        }
        
        /* Code blocks */
        .stCodeBlock {
            background: var(--bg-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
        }
        
        /* Alerts / Info boxes */
        .stAlert {
            background: var(--glass-bg) !important;
            border-left: 4px solid var(--accent-cyan) !important;
            border-radius: 0 8px 8px 0 !important;
        }
        
        /* Gauge-like metrics container */
        .gauge-container {
            background: var(--glass-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            margin: 0.5rem 0;
        }
        
        .gauge-label {
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        
        .gauge-value {
            color: var(--accent-cyan);
            font-size: 2rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .gauge-value.warning {
            color: var(--accent-orange);
        }
        
        .gauge-value.critical {
            color: var(--accent-red);
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-secondary);
        }
        
        /* Logo animation */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .logo-pulse {
            animation: pulse 2s ease-in-out infinite;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Main App
# =============================================================================
def main():
    """Main application entry point."""
    load_custom_css()

    # Sidebar
    with st.sidebar:
        st.markdown("# 🚀 RyzenAI-LocalLab")
        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigation",
            ["💬 Chat", "📚 Models", "📊 Dashboard", "⚙️ Settings"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Quick stats
        st.markdown("### 📈 Quick Stats")

        # These will be replaced with real data
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU", "15%", delta="-2%")
        with col2:
            st.metric("RAM", "45%", delta="+3%")

        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #8b949e; font-size: 0.8rem;'>
                v0.1.0 | Made with ❤️
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Main content based on navigation
    if page == "💬 Chat":
        show_chat_page()
    elif page == "📚 Models":
        show_models_page()
    elif page == "📊 Dashboard":
        show_dashboard_page()
    elif page == "⚙️ Settings":
        show_settings_page()


def show_chat_page():
    """Display the chat interface."""
    st.title("💬 Chat")

    # Model selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        model = st.selectbox(
            "Model",
            ["No models loaded - Please load a model first"],
            key="model_selector",
        )
    with col2:
        st.button("🔄 Reload", key="reload_model")
    with col3:
        st.button("⚡ Load Model", key="load_model")

    st.markdown("---")

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Placeholder for assistant response
        with st.chat_message("assistant"):
            st.markdown("⏳ Loading model required to generate responses...")

    # Generation parameters in expander
    with st.expander("⚙️ Generation Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, key="temperature")
            st.slider("Top-P", 0.0, 1.0, 0.9, 0.05, key="top_p")
        with col2:
            st.slider("Top-K", 1, 200, 50, 1, key="top_k")
            st.slider("Max Tokens", 64, 8192, 2048, 64, key="max_tokens")


def show_models_page():
    """Display the models management page."""
    st.title("📚 Model Library")

    # Download new model section
    st.markdown("### 📥 Download New Model")

    col1, col2 = st.columns([3, 1])
    with col1:
        repo_id = st.text_input(
            "HuggingFace Repository ID",
            placeholder="mistralai/Devstral-Small-2505",
            key="repo_id",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        download_btn = st.button("📥 Download", type="primary", key="download_btn")

    if download_btn and repo_id:
        with st.status("Downloading model...", expanded=True) as status:
            st.write(f"Fetching: {repo_id}")
            # Placeholder for download progress
            progress = st.progress(0)
            for i in range(100):
                import time

                time.sleep(0.01)
                progress.progress(i + 1)
            status.update(label="Download complete!", state="complete")

    st.markdown("---")

    # Local models
    st.markdown("### 💾 Local Models")

    # Placeholder - will be replaced with real data
    models_data = [
        {
            "Name": "Devstral-Small-2505",
            "Size": "16.2 GB",
            "Format": "safetensors",
            "Status": "✅ Ready",
        },
        {
            "Name": "Qwen3-30B-A3B",
            "Size": "17.4 GB",
            "Format": "safetensors",
            "Status": "✅ Ready",
        },
    ]

    if models_data:
        for model in models_data:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                with col1:
                    st.markdown(f"**{model['Name']}**")
                with col2:
                    st.caption(model["Size"])
                with col3:
                    st.caption(model["Format"])
                with col4:
                    st.caption(model["Status"])
                with col5:
                    if st.button("🗑️", key=f"delete_{model['Name']}"):
                        st.warning("Delete functionality coming soon")
                st.markdown("---")
    else:
        st.info("No models downloaded yet. Use the form above to download a model.")


def show_dashboard_page():
    """Display the monitoring dashboard."""
    st.title("📊 HomeLab Dashboard")

    # Hardware stats
    st.markdown("### 🔧 Hardware Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="gauge-container">
                <div class="gauge-label">CPU USAGE</div>
                <div class="gauge-value">15.2%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Cores", "16 / 32 threads")
        st.metric("Frequency", "4.2 GHz")

    with col2:
        st.markdown(
            """
            <div class="gauge-container">
                <div class="gauge-label">RAM USAGE</div>
                <div class="gauge-value warning">67.8%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Used", "84.1 GB / 124 GB")
        st.metric("Available", "39.9 GB")

    with col3:
        st.markdown(
            """
            <div class="gauge-container">
                <div class="gauge-label">GPU USAGE</div>
                <div class="gauge-value">32.5%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Device", "Radeon 8060S")
        st.metric("VRAM", "Unified Memory")

    st.markdown("---")

    # Inference stats
    st.markdown("### ⚡ Inference Performance")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tokens/sec (Gen)", "42.5", delta="+5.2")
    with col2:
        st.metric("Tokens/sec (Prompt)", "156.3", delta="-12.1")
    with col3:
        st.metric("Time to First Token", "0.23s", delta="-0.05s")

    # Graphs placeholder
    st.markdown("### 📈 Performance History")
    st.info("Real-time graphs will appear here when monitoring is active.")


def show_settings_page():
    """Display the settings page."""
    st.title("⚙️ Settings")

    # User info
    st.markdown("### 👤 Account")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Username", value="admin", disabled=True)
    with col2:
        st.text_input("API Key", value="sk-**********************", type="password")

    if st.button("🔄 Regenerate API Key"):
        st.success("API Key regenerated!")

    st.markdown("---")

    # Model settings
    st.markdown("### 🧠 Inference Settings")

    device = st.selectbox("Device", ["auto", "cuda (ROCm)", "cpu"])
    max_memory = st.slider("Max Memory Usage", 50, 100, 90, 5, format="%d%%")

    st.markdown("---")

    # Paths
    st.markdown("### 📁 Paths")

    models_path = st.text_input("Models Directory", value="/srv/models")
    data_path = st.text_input("Data Directory", value="./data")

    st.markdown("---")

    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved!")


# =============================================================================
# Run App
# =============================================================================
if __name__ == "__main__":
    main()
