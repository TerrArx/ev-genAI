import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import numpy as np
import google.generativeai as genai
import tensorflow as tf
from tensorflow import keras
import joblib
import sklearn
import json
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from io import BytesIO
import base64

# --- PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(
    page_title="EV Spec Generator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOAD EXTERNAL CSS ---
def load_css():
    """Load external CSS file with theme detection."""
    css_file = os.path.join(os.path.dirname(__file__), 'style.css')
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Add theme detection script
    st.markdown("""
    <script>
    (function() {
        const updateTheme = () => {
            const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            const streamlitTheme = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
            
            if (streamlitTheme) {
                const computedBg = window.getComputedStyle(streamlitTheme).backgroundColor;
                const isDarkMode = computedBg.match(/rgb\\((\\d+),\\s*(\\d+),\\s*(\\d+)\\)/) || [];
                const brightness = isDarkMode.length > 3 ? 
                    (parseInt(isDarkMode[1]) * 299 + parseInt(isDarkMode[2]) * 587 + parseInt(isDarkMode[3]) * 114) / 1000 : 255;
                
                document.documentElement.setAttribute('data-theme', brightness < 128 ? 'dark' : 'light');
            }
        };
        
        updateTheme();
        
        // Watch for theme changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', updateTheme);
        }
        
        // Re-check on Streamlit reruns
        const observer = new MutationObserver(updateTheme);
        observer.observe(document.body, { attributes: true, childList: true, subtree: true });
    })();
    </script>
    """, unsafe_allow_html=True)

load_css()


# --- DEFINE CUSTOM FUNCTIONS ---
def sample_z(args):
    """Custom sampling function for VAE latent space."""
    mean, log_var = args
    eps = tf.random.normal(shape=tf.shape(mean))
    return mean + tf.exp(0.5 * log_var) * eps

# --- ARTIFACT LOADING ---
@st.cache_resource
def load_models_and_artifacts():
    """Loads all models and preprocessing objects."""
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'keras format')
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data-encoded-ev')
    
    scaler_y = joblib.load(os.path.join(DATA_PATH, 'scaler_y.pkl'))
    encoder_c = joblib.load(os.path.join(DATA_PATH, 'encoder_c.pkl'))
    y_features = joblib.load(os.path.join(DATA_PATH, 'y_features.pkl'))
    c_feature_names = joblib.load(os.path.join(DATA_PATH, 'c_features_names.pkl'))
    
    with open(os.path.join(DATA_PATH, 'budget_bins.json'), 'r') as f:
        budget_bins = json.load(f)

    encoder = keras.models.load_model(
        os.path.join(MODEL_PATH, 'encoder.keras'),
        custom_objects={'sample_z': sample_z}
    )
    decoder = keras.models.load_model(os.path.join(MODEL_PATH, 'decoder.keras'))
    
    Y_DIM = len(y_features)
    C_DIM = len(c_feature_names)
    LATENT_DIM = 12
    
    return (encoder, decoder, scaler_y, encoder_c, y_features, 
            c_feature_names, budget_bins, LATENT_DIM, Y_DIM, C_DIM)

# --- LOAD MODELS ---
try:
    (encoder, decoder, scaler_y, encoder_c, y_features, 
     c_names, budget_bins, LATENT_DIM, Y_DIM, C_DIM) = load_models_and_artifacts()
except Exception as e:
    st.error("FATAL ERROR: Could not load models or artifacts.")
    st.exception(e)
    st.stop()

# --- CONFIGURE GEMINI API ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.warning(f"Could not configure Gemini API. Analysis will be disabled. Error: {e}")
    gemini_model = None

# --- HELPER FUNCTIONS ---
def generate_cvae_specs(inputs_dict, cat_features, num_features):
    """Generates EV specs using CVAE."""
    cat_values = [inputs_dict[f] for f in cat_features]
    num_values = [inputs_dict[f] for f in num_features]
    
    cat_encoded = encoder_c.transform([cat_values])
    num_encoded = np.array(num_values).reshape(1, -1)
    
    C_vector = np.hstack([cat_encoded, num_encoded]).astype(np.float32)
    z_sample = np.random.normal(size=(1, LATENT_DIM))
    
    Y_scaled_gen = decoder.predict([z_sample, C_vector], verbose=0)
    Y_real_gen = scaler_y.inverse_transform(Y_scaled_gen)
    
    return dict(zip(y_features, Y_real_gen[0]))

def get_gemini_highlights(specs_dict, inputs_dict):
    """Gets positive highlights about the generated specs."""
    if not gemini_model:
        return "⚡ Analysis service is not configured. Please check your API key."

    prompt = f"""
    You are an enthusiastic EV technology expert. A generative model has created an EV specification with these features:
    
    **Generated Specifications:**
    - Range: {specs_dict['Range_km']:.0f} km
    - Price: ${specs_dict['Price_USD']:,.0f} USD
    - Battery Capacity: {specs_dict['Battery_Capacity_kWh']:.0f} kWh
    
    **User Preferences:**
    - Region: {inputs_dict['C_Region']}
    - Battery Chemistry: {inputs_dict['C_Battery_Chem']}
    - Charging Connector: {inputs_dict['C_Connector']}
    - Charging Speed: {inputs_dict['C_Charge_Speed']}
    - Budget Class: {inputs_dict['C_Budget']}
    - V2X Enabled: {"Yes" if inputs_dict['C_Has_V2X'] else "No"}
    
    Provide 3-4 brief, enthusiastic bullet points highlighting ONLY the positive aspects, strengths, 
    and advantages of this specification. Focus on market competitiveness, value proposition, 
    technology benefits, and user advantages. Be concise and exciting!
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Error communicating with analysis service: {e}"

def chatbot_response(user_message, specs_dict, inputs_dict):
    """Handles chatbot conversations about the generated EV."""
    if not gemini_model:
        return "Analysis service is not configured. Please check your API key."
    
    prompt = f"""
    You are an expert EV consultant helping a user understand their generated EV specification.
    
    **Generated EV Specs:**
    - Range: {specs_dict.get('Range_km', 'N/A'):.0f} km
    - Price: ${specs_dict.get('Price_USD', 'N/A'):,.0f} USD
    - Battery: {specs_dict.get('Battery_Capacity_kWh', 'N/A'):.0f} kWh
    - Region: {inputs_dict.get('C_Region', 'N/A')}
    - Battery Chemistry: {inputs_dict.get('C_Battery_Chem', 'N/A')}
    - Connector: {inputs_dict.get('C_Connector', 'N/A')}
    - Charging Speed: {inputs_dict.get('C_Charge_Speed', 'N/A')}
    - V2X: {"Yes" if inputs_dict.get('C_Has_V2X') else "No"}
    
    User Question: {user_message}
    
    Provide a helpful, concise response addressing their concerns or questions about this EV specification.
    Focus on being informative and addressing future concerns, market viability, or technical questions.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- INITIALIZE SESSION STATE ---
if 'generated_specs' not in st.session_state:
    st.session_state.generated_specs = None
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'is_thinking' not in st.session_state:
    st.session_state.is_thinking = False

# --- MAIN APP ---
st.markdown("<h1 class='main-title'>⚡ Electric Vehicle Specification Generator</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Design your perfect electric vehicle with data-driven recommendations</p>", unsafe_allow_html=True)

# --- LAYOUT: LEFT COLUMN (INPUTS) AND RIGHT COLUMN (CHATBOT) ---
input_col, chat_col = st.columns([1.5, 1])

with input_col:
    # --- CONTAINER 1: USER INPUT VALUES ---
    st.markdown("<div class='spec-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🎯 Your Design Preferences</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        region = st.selectbox("🌍 Market Region", encoder_c.categories_[0], key="region")
        battery_chem = st.selectbox("🔋 Battery Chemistry", encoder_c.categories_[1], key="battery")
        connector = st.selectbox("🔌 Charging Connector", encoder_c.categories_[2], key="connector")
    
    with col2:
        charge_speed = st.selectbox("⚡ Charging Speed", encoder_c.categories_[3], key="speed")
        budget = st.selectbox("💰 Budget Class", budget_bins['labels'], key="budget")
        v2x = st.checkbox("🔄 Enable V2X Technology", key="v2x")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # --- GENERATE BUTTON ---
    if st.button("🚀 Generate EV Specification", width='stretch'):
        # Prepare inputs
        cat_features_list = ['C_Region', 'C_Battery_Chem', 'C_Connector', 'C_Charge_Speed', 'C_Budget']
        num_features_list = ['C_Has_V2X']
        
        inputs = {
            'C_Region': region,
            'C_Battery_Chem': battery_chem,
            'C_Connector': connector,
            'C_Charge_Speed': charge_speed,
            'C_Budget': budget,
            'C_Has_V2X': 1 if v2x else 0
        }
        
        with st.spinner("🤖 Generating your perfect EV..."):
            generated_specs = generate_cvae_specs(inputs, cat_features_list, num_features_list)
            st.session_state.generated_specs = generated_specs
            st.session_state.user_inputs = inputs

# --- RIGHT COLUMN: CHATBOT (BESIDE INPUTS) ---
with chat_col:
    st.markdown("<div class='chat-header'>💬 EV Assistant</div>", unsafe_allow_html=True)
    
    # Chat messages container
    chat_container = st.container(height=320)
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Show thinking indicator
        if st.session_state.is_thinking:
            st.markdown("""
                <div class='thinking-indicator'>
                    <div class='thinking-dots'>
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask about your EV specs...", key="chat_input"):
        if st.session_state.generated_specs is None:
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "⚠️ Please generate an EV specification first!"
            })
            st.rerun()
        else:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.is_thinking = True
            st.rerun()
    
    # Process response if thinking
    if st.session_state.is_thinking:
        bot_response = chatbot_response(
            st.session_state.chat_history[-1]["content"], 
            st.session_state.generated_specs,
            st.session_state.user_inputs
        )
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        st.session_state.is_thinking = False
        st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", width='stretch', key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

# --- FULL WIDTH: DISPLAY RESULTS ---
if st.session_state.generated_specs is not None:
    specs = st.session_state.generated_specs
    inputs = st.session_state.user_inputs
    
    # --- CONTAINER 2: GENERATED SPECIFICATIONS ---
    st.markdown("<div class='spec-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📊 Generated EV Specifications</div>", unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric(
            label="💵 Target Price",
            value=f"${specs['Price_USD']:,.0f}",
            delta="Optimized for market"
        )
    
    with metric_col2:
        st.metric(
            label="🛣️ Driving Range",
            value=f"{specs['Range_km']:.0f} km",
            delta="Full charge range"
        )
    
    with metric_col3:
        st.metric(
            label="🔋 Battery Capacity",
            value=f"{specs['Battery_Capacity_kWh']:.0f} kWh",
            delta="High efficiency"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # --- CONTAINER 3: MARKET HIGHLIGHTS ---
    st.markdown("<div class='spec-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>✨ Market Highlights</div>", unsafe_allow_html=True)
    
    with st.spinner("🔍 Analyzing market advantages..."):
        analysis = get_gemini_highlights(specs, inputs)
        st.markdown(f"<div class='analysis-box'>{analysis}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # --- DOWNLOAD REPORT BUTTON ---
    import datetime
    if 'generation_time' not in st.session_state:
        st.session_state.generation_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate PDF report
    def create_pdf_report():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles with better color palette
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=26,
            textColor=colors.HexColor('#1e3a8a'),  # Deep blue
            spaceAfter=20,
            alignment=1  # Center
        )
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#64748b'),  # Gray
            spaceAfter=30,
            alignment=1  # Center
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#0f766e'),  # Teal
            spaceAfter=12,
            spaceBefore=16,
            borderPadding=5,
            leftIndent=0
        )
        normal_style = styles['Normal']
        disclaimer_style = ParagraphStyle(
            'DisclaimerStyle',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#dc2626'),  # Red
            alignment=1,  # Center
            spaceAfter=10,
            borderWidth=1,
            borderColor=colors.HexColor('#dc2626'),
            borderPadding=10,
            backColor=colors.HexColor('#fef2f2')  # Light red background
        )
        footer_style = ParagraphStyle(
            'FooterStyle',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#475569'),
            alignment=1,
            spaceAfter=5
        )
        
        # Title
        elements.append(Paragraph("ELECTRIC VEHICLE SPECIFICATION REPORT", title_style))
        elements.append(Paragraph(f"Generated on: {st.session_state.generation_time}", subtitle_style))
        
        # Disclaimer Section
        disclaimer_text = """
        <b>⚠️ IMPORTANT DISCLAIMER</b><br/>
        This is a <b>DEMONSTRATION PROJECT ONLY</b> using synthetic data generated for educational purposes. 
        The specifications shown are <b>NOT REAL</b> and should <b>NOT</b> be used for actual vehicle purchasing decisions 
        or any commercial purposes. This report is generated using a machine learning model trained on hypothetical data.
        """
        elements.append(Paragraph(disclaimer_text, disclaimer_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Design Preferences Section
        elements.append(Paragraph("DESIGN PREFERENCES", heading_style))
        prefs_data = [
            ['Market Region:', inputs['C_Region']],
            ['Battery Chemistry:', inputs['C_Battery_Chem']],
            ['Charging Connector:', inputs['C_Connector']],
            ['Charging Speed:', inputs['C_Charge_Speed']],
            ['Budget Class:', inputs['C_Budget']],
            ['V2X Technology:', 'Enabled' if inputs['C_Has_V2X'] else 'Disabled']
        ]
        prefs_table = Table(prefs_data, colWidths=[2.5*inch, 3.5*inch])
        prefs_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e0f2fe')),  # Light blue
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1e293b')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1'))
        ]))
        elements.append(prefs_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Generated Specifications Section
        elements.append(Paragraph("GENERATED SPECIFICATIONS", heading_style))
        specs_data = [
            ['Target Price:', f"${specs['Price_USD']:,.2f} USD"],
            ['Driving Range:', f"{specs['Range_km']:.2f} km"],
            ['Battery Capacity:', f"{specs['Battery_Capacity_kWh']:.2f} kWh"]
        ]
        specs_table = Table(specs_data, colWidths=[2.5*inch, 3.5*inch])
        specs_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ccfbf1')),  # Light teal
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1e293b')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1'))
        ]))
        elements.append(specs_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Performance Metrics Section
        elements.append(Paragraph("PERFORMANCE METRICS", heading_style))
        metrics_data = [
            ['Battery Efficiency:', f"{(specs['Battery_Capacity_kWh'] / specs['Range_km'] * 100):.2f} Wh/km"],
            ['Cost per Kilometer:', f"${specs['Price_USD'] / specs['Range_km']:.2f} USD/km"],
            ['Range per kWh:', f"{specs['Range_km'] / specs['Battery_Capacity_kWh']:.2f} km/kWh"]
        ]
        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 3.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#fef3c7')),  # Light amber
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1e293b')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1'))
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Market Analysis Section
        elements.append(Paragraph("MARKET ANALYSIS", heading_style))
        # Clean up the analysis text for PDF
        analysis_clean = analysis.replace('*', '').replace('#', '')
        analysis_paras = analysis_clean.split('\n')
        for para in analysis_paras:
            if para.strip():
                elements.append(Paragraph(para.strip(), normal_style))
                elements.append(Spacer(1, 0.1*inch))
        
        # Add page break before visualizations
        elements.append(PageBreak())
        
        # Visualizations Section
        elements.append(Paragraph("PERFORMANCE VISUALIZATIONS", heading_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Helper function to convert plotly figure to image
        def fig_to_image(fig, width=5*inch, height=3*inch):
            img_bytes = fig.to_image(format="png", width=800, height=500)
            img_buffer = BytesIO(img_bytes)
            img = Image(img_buffer, width=width, height=height)
            return img
        
        try:
            # Create the same charts as in the web app
            # Chart 1: Price vs Range
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=[specs['Price_USD']],
                y=[specs['Range_km']],
                mode='markers',
                marker=dict(size=20, color='#4a9eff', symbol='star'),
                name='Your EV'
            ))
            ref_prices = [30000, 45000, 60000, 75000, 90000]
            ref_ranges = [250, 350, 450, 550, 600]
            fig1.add_trace(go.Scatter(
                x=ref_prices,
                y=ref_ranges,
                mode='markers',
                marker=dict(size=8, color='#666666', opacity=0.5),
                name='Market Average'
            ))
            fig1.update_layout(
                title='Price vs Range Comparison',
                xaxis_title='Price (USD)',
                yaxis_title='Range (km)',
                template='plotly_white',
                showlegend=True,
                height=400
            )
            
            elements.append(Paragraph("Price vs Range Comparison", normal_style))
            elements.append(Spacer(1, 0.1*inch))
            elements.append(fig_to_image(fig1))
            elements.append(Spacer(1, 0.3*inch))
            
            # Chart 2: Specs Breakdown
            fig3 = go.Figure()
            categories = ['Range (km)', 'Battery (kWh)', 'Price (K USD)']
            values = [specs['Range_km'], specs['Battery_Capacity_kWh'], specs['Price_USD']/1000]
            fig3.add_trace(go.Bar(
                x=categories,
                y=values,
                marker=dict(color=['#4a9eff', '#60d394', '#ffa07a']),
                text=[f"{v:.0f}" for v in values],
                textposition='outside'
            ))
            fig3.update_layout(
                title='Specification Overview',
                template='plotly_white',
                showlegend=False,
                height=400,
                yaxis={'title': 'Value'}
            )
            
            elements.append(Paragraph("Specification Overview", normal_style))
            elements.append(Spacer(1, 0.1*inch))
            elements.append(fig_to_image(fig3))
            elements.append(Spacer(1, 0.3*inch))
            
            # Chart 3: Cost per Kilometer
            cost_per_km = specs['Price_USD'] / specs['Range_km']
            budget_cost_map = {'Budget': 120, 'Mid-Range': 180, 'Luxury': 250}
            
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                x=['Your EV', inputs['C_Budget'] + ' Avg'],
                y=[cost_per_km, budget_cost_map.get(inputs['C_Budget'], 180)],
                marker=dict(color=['#4a9eff', '#666666']),
                text=[f"${cost_per_km:.2f}", f"${budget_cost_map.get(inputs['C_Budget'], 180):.2f}"],
                textposition='outside'
            ))
            fig4.update_layout(
                title='Cost per Kilometer Analysis',
                template='plotly_white',
                showlegend=False,
                height=400,
                yaxis={'title': 'USD per km'}
            )
            
            elements.append(Paragraph("Cost per Kilometer Analysis", normal_style))
            elements.append(Spacer(1, 0.1*inch))
            elements.append(fig_to_image(fig4))
            
        except Exception as e:
            elements.append(Paragraph(f"Note: Visualization generation encountered an issue: {str(e)}", normal_style))
        
        # Add project information footer
        elements.append(PageBreak())
        elements.append(Spacer(1, 1*inch))
        
        footer_title = ParagraphStyle(
            'FooterTitle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1e3a8a'),
            alignment=1,
            spaceAfter=15
        )
        
        elements.append(Paragraph("PROJECT INFORMATION", footer_title))
        elements.append(Spacer(1, 0.2*inch))
        
        project_info = """
        <b>Project:</b> Generative AI for Electric Vehicle Design<br/>
        <b>Purpose:</b> Educational Demonstration - Edunet Foundation & Shell Skills4Future AICTE Internship<br/>
        <b>Focus:</b> Green Skills and Artificial Intelligence for Sustainable Innovation<br/>
        <br/>
        <b>Dataset Source:</b><br/>
        EV Electrical Vehicles Dataset - 3K+ Records 2025<br/>
        Kaggle: <font color="#1e40af">https://www.kaggle.com/datasets/pratyushpuri/ev-electrical-vehicles-dataset-3k-records-2025</font><br/>
        <br/>
        <b>Developer:</b><br/>
        <b>Nabil Ahmed</b><br/>
        B.Tech in Artificial Intelligence and Machine Learning<br/>
        Netaji Subhash Engineering College, Kolkata, India<br/>
        <br/>
        <b>Contact Information:</b><br/>
        📧 Email: nabil13147@gmail.com<br/>
        🐙 GitHub: @TerrArx (https://github.com/TerrArx)<br/>
        💼 LinkedIn: https://www.linkedin.com/in/nabil-ahmed-876b30240/<br/>
        """
        elements.append(Paragraph(project_info, footer_style))
        elements.append(Spacer(1, 0.3*inch))
        
        final_disclaimer = """
        <b><font color="#dc2626">⚠️ FINAL DISCLAIMER:</font></b><br/>
        This report uses <b>SYNTHETIC DATA</b> and machine learning predictions for <b>DEMONSTRATION PURPOSES ONLY</b>. 
        All specifications, prices, and analyses are <b>GENERATED BY AN AI MODEL</b> and do not represent real products. 
        This information is <b>NOT SUITABLE</b> for making actual purchasing decisions, investment analysis, or any commercial use. 
        The creator assumes <b>NO LIABILITY</b> for any decisions made based on this report.
        """
        elements.append(Paragraph(final_disclaimer, disclaimer_style))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer
    
    pdf_buffer = create_pdf_report()
    
    col_left, col_center, col_right = st.columns([1, 1, 1])
    with col_center:
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"EV_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            width='stretch',
            key="download_report"
        )
    
    # --- CONTAINER 4: VISUALIZATIONS ---
    st.markdown("<div class='spec-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📈 Performance Analysis</div>", unsafe_allow_html=True)
    
    # Create visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Price vs Range comparison
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=[specs['Price_USD']],
            y=[specs['Range_km']],
            mode='markers',
            marker=dict(size=20, color='#4a9eff', symbol='star'),
            name='Your EV'
        ))
        
        # Add reference points (sample data)
        ref_prices = [30000, 45000, 60000, 75000, 90000]
        ref_ranges = [250, 350, 450, 550, 600]
        fig1.add_trace(go.Scatter(
            x=ref_prices,
            y=ref_ranges,
            mode='markers',
            marker=dict(size=8, color='#666666', opacity=0.5),
            name='Market Average'
        ))
        
        fig1.update_layout(
            title='Price vs Range Comparison',
            xaxis_title='Price (USD)',
            yaxis_title='Range (km)',
            template='plotly_dark',
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font=dict(size=12, color='#cccccc'),
            showlegend=True,
            height=350
        )
        st.plotly_chart(fig1, width='stretch')
    
    with viz_col2:
        # Battery Efficiency (kWh per km)
        efficiency = specs['Battery_Capacity_kWh'] / specs['Range_km']
        
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=efficiency * 100,
            title={'text': "Battery Efficiency (Wh/km)", 'font': {'size': 16, 'color': '#ffffff'}},
            delta={'reference': 20, 'increasing': {'color': "#ff6b6b"}, 'decreasing': {'color': "#60d394"}},
            gauge={
                'axis': {'range': [None, 30], 'tickcolor': "#cccccc"},
                'bar': {'color': "#4a9eff"},
                'bgcolor': "#2a2a2a",
                'borderwidth': 2,
                'bordercolor': "#404040",
                'steps': [
                    {'range': [0, 15], 'color': '#2a2a2a'},
                    {'range': [15, 20], 'color': '#2a2a2a'},
                    {'range': [20, 30], 'color': '#2a2a2a'}
                ],
                'threshold': {
                    'line': {'color': "#60d394", 'width': 4},
                    'thickness': 0.75,
                    'value': 18
                }
            }
        ))
        
        fig2.update_layout(
            template='plotly_dark',
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font=dict(color='#cccccc'),
            height=350
        )
        st.plotly_chart(fig2, width='stretch')
    
    # Full width charts
    viz_col3, viz_col4 = st.columns(2)
    
    with viz_col3:
        # Specs Breakdown Bar Chart
        fig3 = go.Figure()
        
        categories = ['Range (km)', 'Battery (kWh)', 'Price (K USD)']
        values = [specs['Range_km'], specs['Battery_Capacity_kWh'], specs['Price_USD']/1000]
        
        fig3.add_trace(go.Bar(
            x=categories,
            y=values,
            marker=dict(
                color=['#4a9eff', '#60d394', '#ffa07a'],
                line=dict(color='#404040', width=1)
            ),
            text=[f"{v:.0f}" for v in values],
            textposition='outside'
        ))
        
        fig3.update_layout(
            title='Specification Overview',
            template='plotly_dark',
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font=dict(size=12, color='#cccccc'),
            showlegend=False,
            height=350,
            yaxis={'title': 'Value'}
        )
        st.plotly_chart(fig3, width='stretch')
    
    with viz_col4:
        # Cost per kilometer of range
        cost_per_km = specs['Price_USD'] / specs['Range_km']
        
        # Comparison with budget categories
        budget_cost_map = {
            'Budget': 120,
            'Mid-Range': 180,
            'Luxury': 250
        }
        
        fig4 = go.Figure()
        
        fig4.add_trace(go.Bar(
            x=['Your EV', inputs['C_Budget'] + ' Avg'],
            y=[cost_per_km, budget_cost_map.get(inputs['C_Budget'], 180)],
            marker=dict(
                color=['#4a9eff', '#666666'],
                line=dict(color='#404040', width=1)
            ),
            text=[f"${cost_per_km:.2f}", f"${budget_cost_map.get(inputs['C_Budget'], 180):.2f}"],
            textposition='outside'
        ))
        
        fig4.update_layout(
            title='Cost per Kilometer Analysis',
            template='plotly_dark',
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font=dict(size=12, color='#cccccc'),
            showlegend=False,
            height=350,
            yaxis={'title': 'USD per km', 'range': [0, max(cost_per_km, budget_cost_map.get(inputs['C_Budget'], 180)) * 1.3]}
        )
        st.plotly_chart(fig4, width='stretch')
    
    # Line chart showing trend
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    # Hypothetical performance over time
    fig5 = go.Figure()
    
    years = list(range(2020, 2031))
    range_trend = [300 + (i * 35) for i in range(len(years))]
    price_trend = [50000 - (i * 2000) for i in range(len(years))]
    
    # Mark current position
    current_year_idx = 5  # 2025
    
    fig5.add_trace(go.Scatter(
        x=years,
        y=range_trend,
        mode='lines+markers',
        name='Average Range (km)',
        line=dict(color='#4a9eff', width=2),
        marker=dict(size=6)
    ))
    
    fig5.add_trace(go.Scatter(
        x=years,
        y=[p/100 for p in price_trend],
        mode='lines+markers',
        name='Average Price (100 USD)',
        line=dict(color='#60d394', width=2),
        marker=dict(size=6),
        yaxis='y2'
    ))
    
    # Highlight current spec
    fig5.add_trace(go.Scatter(
        x=[2025],
        y=[specs['Range_km']],
        mode='markers',
        name='Your EV Range',
        marker=dict(size=15, color='#ffa07a', symbol='star')
    ))
    
    fig5.update_layout(
        title='EV Market Evolution (2020-2030)',
        xaxis=dict(title='Year', color='#cccccc'),
        yaxis=dict(title='Range (km)', color='#4a9eff'),
        yaxis2=dict(title='Price (100 USD)', overlaying='y', side='right', color='#60d394'),
        template='plotly_dark',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(size=12, color='#cccccc'),
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig5, width='stretch')
    
    st.markdown("</div>", unsafe_allow_html=True)