import streamlit as st

def inject_custom_css():
    """Injects Google Fonts and premium dark-mode custom CSS into the Streamlit app."""
    css = """
    <style>
    /* --------------------------------------------------------------------------
       Google Fonts Import
       -------------------------------------------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Sora:wght@400;600;700;800&display=swap');

    /* Global Typography Defaults */
    html, body, [class*="css"], .stMarkdown, p, div, label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #E6E6EB;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Sora', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #FFFFFF !important;
    }

    code, pre, [class*="stCode"], .metric-value, .mono-text {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* --------------------------------------------------------------------------
       App Container & Background
       -------------------------------------------------------------------------- */
    .stApp {
        background-color: #0B0E14 !important;
        background-image: 
            radial-gradient(at 10% 20%, rgba(124, 92, 255, 0.07) 0px, transparent 50%),
            radial-gradient(at 90% 80%, rgba(34, 211, 184, 0.05) 0px, transparent 50%);
        background-attachment: fixed;
    }

    /* Main Block Padding */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 4rem !important;
        max-width: 1400px !important;
    }

    /* --------------------------------------------------------------------------
       Card Container Styling (st.container border=True)
       -------------------------------------------------------------------------- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #151922 !important;
        border: 1px solid rgba(124, 92, 255, 0.18) !important;
        border-radius: 14px !important;
        padding: 1.25rem !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
        margin-bottom: 1.25rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(124, 92, 255, 0.35) !important;
    }

    /* --------------------------------------------------------------------------
       KPI Metric Cards
       -------------------------------------------------------------------------- */
    [data-testid="stMetric"] {
        background: #1C2230 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        padding: 0.9rem 1.1rem !important;
        transition: border-color 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: #7C5CFF !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #94A3B8 !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 1.65rem !important;
        color: #7C5CFF !important;
    }

    /* --------------------------------------------------------------------------
       Buttons & CTAs
       -------------------------------------------------------------------------- */
    .stButton > button {
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
        background: linear-gradient(135deg, #7C5CFF 0%, #5B32E4 100%) !important;
        color: #FFFFFF !important;
        padding: 0.55rem 1.4rem !important;
        box-shadow: 0 4px 14px rgba(124, 92, 255, 0.35) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(124, 92, 255, 0.5) !important;
        background: linear-gradient(135deg, #8B6EFF 0%, #683FEA 100%) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* Secondary / Download Buttons */
    .stDownloadButton > button {
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: 1px solid rgba(34, 211, 184, 0.4) !important;
        background: rgba(34, 211, 184, 0.08) !important;
        color: #22D3B8 !important;
        transition: all 0.2s ease !important;
    }

    .stDownloadButton > button:hover {
        background: rgba(34, 211, 184, 0.18) !important;
        border-color: #22D3B8 !important;
        color: #FFFFFF !important;
    }

    /* --------------------------------------------------------------------------
       Banners & Custom Callouts
       -------------------------------------------------------------------------- */
    .banner-success {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        color: #4ADE80;
        font-weight: 500;
        margin: 1rem 0;
    }

    .banner-warning {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        color: #FBBF24;
        font-weight: 500;
        margin: 1rem 0;
    }

    .banner-info {
        background: rgba(124, 92, 255, 0.12);
        border: 1px solid rgba(124, 92, 255, 0.4);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        color: #A78BFA;
        font-weight: 500;
        margin: 1rem 0;
    }

    /* --------------------------------------------------------------------------
       Step Progress Navigation Bar
       -------------------------------------------------------------------------- */
    .step-nav-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
        padding: 0.75rem 1rem;
        background: #151922;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        align-items: center;
    }

    .step-pill {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }

    .step-pill.active {
        color: #FFFFFF;
        background: linear-gradient(135deg, rgba(124, 92, 255, 0.3) 0%, rgba(91, 50, 228, 0.3) 100%);
        border-color: #7C5CFF;
        box-shadow: 0 0 12px rgba(124, 92, 255, 0.25);
    }

    .step-pill.completed {
        color: #22D3B8;
        background: rgba(34, 211, 184, 0.08);
        border-color: rgba(34, 211, 184, 0.2);
    }

    .step-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        opacity: 0.9;
    }

    /* --------------------------------------------------------------------------
       Tabs Styling
       -------------------------------------------------------------------------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0E121A;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 6px;
        color: #94A3B8;
        font-weight: 500;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #7C5CFF !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }

    /* --------------------------------------------------------------------------
       Dataframe & Table Enhancements
       -------------------------------------------------------------------------- */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Micro Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animated-section {
        animation: fadeIn 0.35s ease forwards;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
