import streamlit as st


def inject_custom_styles() -> None:
    """Injects high-quality production-grade custom CSS styles into the Streamlit app page."""
    css = """
    <style>
    /* Import modern typography from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600;700&display=swap');

    /* Global Typography overrides */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
    }

    /* Header Gradient branding style */
    .brand-header {
        background: linear-gradient(135deg, #FF3366, #FF9933, #33CCFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.1rem;
        text-align: center;
    }

    .brand-subtitle {
        color: #8892B0;
        font-size: 1.1rem;
        margin-bottom: 1.8rem;
        font-weight: 300;
        text-align: center;
    }

    /* Glassmorphic settings panel style */
    .glass-card {
        background: rgba(17, 25, 40, 0.65);
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    /* Dynamic Dashboard Metric Panels */
    .metric-panel {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 0.8rem;
        border-left: 4px solid #FF3366;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-panel:hover {
        transform: translateY(-2px);
        border-left-color: #33CCFF;
    }

    .metric-title {
        color: #8892B0;
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        color: #E2E8F0;
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
    }

    .metric-sub {
        color: #718096;
        font-size: 0.7rem;
        margin-top: 0.2rem;
    }

    /* Custom Streamlit Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #FF3366, #FF9933);
        color: white !important;
        font-weight: 600;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.8rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(255, 51, 102, 0.25);
        width: 100%;
        display: block;
        margin-top: 1rem;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #FF5588, #FFAA55) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(255, 51, 102, 0.4) !important;
    }

    div.stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Border Styling for Code Output Blocks */
    .summary-box {
        background-color: rgba(255, 255, 255, 0.015);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1.2rem;
        color: #F7FAFC;
        font-size: 1rem;
        line-height: 1.6;
        margin-top: 1rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }

    .summary-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #33CCFF;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Remove Streamlit default footer */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
