import streamlit as st
import tensorflow as tf
import numpy as np
import time
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Add model caching for better performance
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('trained_model.keras')

# Tensorflow Prediction
def model_prediction(test_image):
    model  = load_model()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    return prediction

# Enhanced link button function
def link_button(url, text, color="#4CAF50", icon=None):
    return st.markdown(
        f'<a href="{url}" target="_blank" style="'
        f'display: inline-block;'
        f'padding: 0.7em 1.5em;'
        f'background-color: {color};'
        f'color: white;'
        f'text-align: center;'
        f'text-decoration: none;'
        f'border-radius: 8px;'
        f'font-weight: bold;'
        f'box-shadow: 0 4px 6px rgba(0,0,0,0.1);'
        f'transition: all 0.3s ease;'
        f'margin: 10px 0;'
        f'border: none;'
        f'cursor: pointer;'
        f'font-size: 16px;'
        f'">'
        f'{icon + " " if icon else ""}{text}'
        f'</a>',
        unsafe_allow_html=True
    )

# Custom CSS for UI enhancements
st.markdown(
    """
    <style>
    /* Main page styling */
    .css-18e3th9 {
        padding: 2rem 5rem;
    }
    
    /* Header styling */
    h1 {
        color: #2e7d32;
        border-bottom: 3px solid #4caf50;
        padding-bottom: 10px;
    }
    
    h2 {
        color: #388e3c;
        border-bottom: 2px solid #81c784;
        padding-bottom: 8px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
    
    .stButton>button:hover {
        background-color: #388e3c !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15) !important;
    }
    
    /* File uploader styling */
    .stFileUploader>div>div>div {
        border: 2px dashed #4CAF50 !important;
        border-radius: 12px !important;
        padding: 30px !important;
        background-color: #f1f8e9 !important;
    }
    
    /* Spinner styling */
    .stSpinner>div>div {
        border-top-color: #4CAF50 !important;
    }
    
    /* Custom card styling */
    .custom-card {
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        padding: 25px;
        margin: 20px 0;
        background: white;
        border-left: 5px solid #4CAF50;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Custom tabs for sidebar */
    .css-1oe5cao {
        border-radius: 12px !important;
        background: #f1f8e9 !important;
    }
    
    /* Footer styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #1b5e20;
        color: white;
        text-align: center;
        padding: 15px;
        font-size: 14px;
    }
    
    /* Progress bar */
    .stProgress>div>div>div>div {
        background-color: #4CAF50 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# StreamLit Sidebar
st.sidebar.title("🌱 Plant Disease Recognition")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=80)

app_mode = st.sidebar.selectbox("Navigate to", ["Home", "About", "Disease Recognition", "Model Performance"],
index = ["Home", "About", "Disease Recognition", "Model Performance"].index(st.session_state.current_page))

# Update session state if user selects different page
if app_mode != st.session_state.current_page:
    st.session_state.current_page = app_mode

# Footer
st.markdown(
    '<div class="footer">Plant Disease Recognition System | Developed with TensorFlow and Streamlit</div>',
    unsafe_allow_html=True
)

# Home Page
if st.session_state.current_page == "Home":
    st.header("🌿 Plant Disease Recognition System")
    
    # Hero section
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <div class="custom-card">
            <h3>Welcome to PlantGuard!</h3>
            <p>Our mission is to help farmers and gardeners identify plant diseases efficiently. 
            Upload an image of a plant leaf, and our AI system will analyze it to detect any signs of diseases.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="custom-card">
            <h3>Why Choose PlantGuard?</h3>
            <ul>
                <li>🌱 <strong>High Accuracy:</strong> State-of-the-art deep learning models</li>
                <li>📱 <strong>Easy to Use:</strong> Simple interface for everyone</li>
                <li>⚡ <strong>Fast Results:</strong> Get diagnoses in seconds</li>
                <li>🌍 <strong>Global Coverage:</strong> 38+ plant diseases recognized</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.image("https://images.unsplash.com/photo-1516214104703-d870798883c5?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&h=600", 
                 caption="Healthy Plants, Healthy Planet", use_column_width=True)
    
    # How it works section
    st.subheader("📋 How It Works")
    steps = st.columns(3)
    with steps[0]:
        st.markdown("""
        <div class="custom-card">
            <h4>1. Upload Image</h4>
            <p>Capture or upload a clear photo of a plant leaf showing signs of disease</p>
            <div style="text-align:center">📸</div>
        </div>
        """, unsafe_allow_html=True)
        
    with steps[1]:
        st.markdown("""
        <div class="custom-card">
            <h4>2. AI Analysis</h4>
            <p>Our deep learning model analyzes the image to detect disease patterns</p>
            <div style="text-align:center">🤖</div>
        </div>
        """, unsafe_allow_html=True)
        
    with steps[2]:
        st.markdown("""
        <div class="custom-card">
            <h4>3. Get Results</h4>
            <p>Receive diagnosis and treatment recommendations instantly</p>
            <div style="text-align:center">📊</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Get started button
    st.markdown("""
    <div style="text-align: center; margin: 40px 0;">
        <a href="#disease-recognition" style="text-decoration: none;">
            <button style="
                background: linear-gradient(to right, #4CAF50, #2E7D32);
                color: white;
                border: none;
                padding: 15px 40px;
                font-size: 18px;
                border-radius: 50px;
                cursor: pointer;
                font-weight: bold;
                box-shadow: 0 6px 15px rgba(76, 175, 80, 0.4);
                transition: all 0.3s ease;
            ">
                🚀 Get Started
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

# About Page
elif st.session_state.current_page == "About":
    st.header("📚 About PlantGuard")
    
    st.markdown("""
    <div class="custom-card">
        <h3>Our Mission</h3>
        <p>PlantGuard aims to democratize plant disease diagnosis, making it accessible to everyone from 
        home gardeners to commercial farmers. By leveraging artificial intelligence, we help detect plant 
        diseases early, potentially saving crops and reducing pesticide use through targeted treatments.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🌱 About the Dataset")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="custom-card">
            <p>The dataset used to train our model was created using offline augmentation from the 
            <a href="https://github.com/spMohanty/PlantVillage-Dataset" target="_blank">original PlantVillage dataset</a>. 
            It contains approximately 87,000 RGB images of healthy and diseased crop leaves categorized into 38 different classes.</p>
            
            <h4>Dataset Composition</h4>
            <ul>
                <li>📁 <strong>Training Set:</strong> 70,295 images (80%)</li>
                <li>📊 <strong>Validation Set:</strong> 17,572 images (20%)</li>
                <li>🧪 <strong>Test Set:</strong> 33 images for final evaluation</li>
            </ul>
            
            <p>The images cover 14 crop species and 26 diseases, providing comprehensive coverage of common agricultural issues.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1516214104703-d870798883c5?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&h=800", 
                 caption="Plant Disease Research", use_column_width=True)
    
    st.subheader("🧠 Model Architecture")
    st.markdown("""
    <div class="custom-card">
        <p>Our model uses a custom Convolutional Neural Network (CNN) architecture with:</p>
        <ul>
            <li>🔢 <strong>7.8 Million Parameters</strong></li>
            <li>🧱 <strong>6 Convolutional Blocks</strong></li>
            <li>🛡️ <strong>Dropout Layers for Regularization</strong></li>
            <li>🎯 <strong>Softmax Classifier with 38 Output Classes</strong></li>
        </ul>
        <p>The model was trained for 30 epochs with data augmentation to improve generalization to real-world conditions.</p>
    </div>
    """, unsafe_allow_html=True)

    
    


# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("🔍 Disease Recognition", anchor="disease-recognition")
    
    # Initialize variables
    prediction = None
    disease_name = None
    class_name = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

    # Upload section with better UI
    st.subheader("📤 Upload a Plant Image")
    test_image = st.file_uploader("", type=["jpg", "jpeg", "png"], 
                                 help="Upload a clear image of a plant leaf showing signs of disease")
    
    if test_image:
        # Show image immediately
        st.subheader("🌿 Your Plant Image")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(test_image, caption="Uploaded Image", use_column_width=True)
        
        # Prediction button
        if st.button("🔍 Analyze Disease", key="predict_button"):
            with st.spinner("🔬 Analyzing your plant. This may take a moment..."):
                # Create progress bar
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.02)  # Simulate processing time
                    progress_bar.progress(percent_complete + 1)
                
                try:
                    prediction = model_prediction(test_image)
                    result_index = np.argmax(prediction)
                    disease_name = class_name[result_index]
                    
                    # Get confidence scores
                    confidences = prediction[0]
                    top_indices = np.argsort(confidences)[-3:][::-1]
                    
                    with col2:
                        st.success(f"Analysis Complete!")
                        st.subheader("📋 Diagnosis Results")
                        
                        # Main result card
                        st.markdown(f"""
                        <div class="custom-card">
                            <h3>Primary Diagnosis</h3>
                            <h4>{disease_name.replace('___', ': ').replace('_', ' ')}</h4>
                            <div style="font-size: 24px; color: #388e3c; margin: 10px 0;">
                                Confidence: <strong>{confidences[result_index]*100:.1f}%</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Top 3 predictions
                        st.subheader("🔍 Other Possible Diagnoses")
                        for i, idx in enumerate(top_indices):
                            if idx == result_index:
                                continue
                            confidence = confidences[idx]
                            st.markdown(f"""
                            <div class="custom-card" style="padding: 15px; margin: 10px 0;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span>{class_name[idx].replace('___', ': ').replace('_', ' ')}</span>
                                    <span style="color: #388e3c; font-weight: bold;">{confidence*100:.1f}%</span>
                                </div>
                                <div style="height: 10px; background: #e0e0e0; border-radius: 5px; margin-top: 8px;">
                                    <div style="height: 100%; width: {confidence*100}%; background: #81c784; border-radius: 5px;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Generate disease info links
                    readable_name = disease_name.replace('___', ' ').replace('_', ' ')
                    query = readable_name.replace(' ', '+')
                    
                    st.subheader("📚 Disease Information & Solutions")
                    
                    # Create tabs for different information types
                    tab1, tab2, tab3 = st.tabs(["🌱 Treatment Options", "🛡️ Prevention Methods", "🖼️ Visual Examples"])
                    
                    with tab1:
                        st.markdown(f"""
                        <div class="custom-card">
                            <h3>Treatment for {readable_name}</h3>
                            <p>While specific treatments vary, here are general approaches:</p>
                            <ul>
                                <li>Remove and destroy infected plant parts</li>
                                <li>Apply appropriate fungicides or bactericides</li>
                                <li>Improve air circulation around plants</li>
                                <li>Adjust watering practices to avoid leaf wetness</li>
                            </ul>
                            <p>For detailed, specific treatment options:</p>
                        </div>
                        """, unsafe_allow_html=True)
                        link_button(f"https://www.google.com/search?q={query}+plant+disease+treatment", 
                                   "Search Treatment Options", "#4CAF50", "🔍")
                    
                    with tab2:
                        st.markdown(f"""
                        <div class="custom-card">
                            <h3>Preventing {readable_name}</h3>
                            <p>Prevention strategies include:</p>
                            <ul>
                                <li>Plant disease-resistant varieties</li>
                                <li>Practice crop rotation</li>
                                <li>Maintain proper plant spacing</li>
                                <li>Water at the base of plants</li>
                                <li>Regularly inspect plants for early signs</li>
                            </ul>
                            <p>Learn more about prevention methods:</p>
                        </div>
                        """, unsafe_allow_html=True)
                        link_button(f"https://www.google.com/search?q={query}+plant+disease+prevention", 
                                   "Search Prevention Methods", "#2196F3", "🔍")
                    
                    with tab3:
                        st.markdown(f"""
                        <div class="custom-card">
                            <h3>Visual Examples of {readable_name}</h3>
                            <p>View images to help confirm diagnosis:</p>
                            <ul>
                                <li>Compare symptoms with verified examples</li>
                                <li>Identify disease progression stages</li>
                                <li>See how disease affects different plant varieties</li>
                            </ul>
                            <p>Browse example images:</p>
                        </div>
                        """, unsafe_allow_html=True)
                        link_button(f"https://www.google.com/search?tbm=isch&q={query}+plant+disease", 
                                   "View Example Images", "#9C27B0", "🔍")
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")

# Model Performance Page
elif app_mode == "Model Performance":
    st.header("📊 Model Performance Metrics")
    
    # Classification report data
    classification_data = [
        {'class': 'Apple___Apple_scab', 'precision': 1.00, 'recall': 0.82, 'f1-score': 0.90, 'support': 504},
        {'class': 'Apple___Black_rot', 'precision': 0.96, 'recall': 1.00, 'f1-score': 0.98, 'support': 497},
        {'class': 'Apple___Cedar_apple_rust', 'precision': 0.97, 'recall': 0.97, 'f1-score': 0.97, 'support': 440},
        {'class': 'Apple___healthy', 'precision': 0.90, 'recall': 0.99, 'f1-score': 0.94, 'support': 502},
        {'class': 'Blueberry___healthy', 'precision': 0.98, 'recall': 0.97, 'f1-score': 0.97, 'support': 454},
        {'class': 'Cherry_(including_sour)___Powdery_mildew', 'precision': 1.00, 'recall': 0.93, 'f1-score': 0.96, 'support': 421},
        {'class': 'Cherry_(including_sour)___healthy', 'precision': 0.96, 'recall': 0.99, 'f1-score': 0.97, 'support': 456},
        {'class': 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'precision': 0.90, 'recall': 0.95, 'f1-score': 0.93, 'support': 410},
        {'class': 'Corn_(maize)___Common_rust_', 'precision': 0.99, 'recall': 1.00, 'f1-score': 1.00, 'support': 477},
        {'class': 'Corn_(maize)___Northern_Leaf_Blight', 'precision': 0.96, 'recall': 0.91, 'f1-score': 0.93, 'support': 477},
        {'class': 'Corn_(maize)___healthy', 'precision': 0.99, 'recall': 1.00, 'f1-score': 1.00, 'support': 465},
        {'class': 'Grape___Black_rot', 'precision': 1.00, 'recall': 0.94, 'f1-score': 0.97, 'support': 472},
        {'class': 'Grape___Esca_(Black_Measles)', 'precision': 0.96, 'recall': 1.00, 'f1-score': 0.98, 'support': 480},
        {'class': 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'precision': 1.00, 'recall': 0.99, 'f1-score': 0.99, 'support': 430},
        {'class': 'Grape___healthy', 'precision': 0.98, 'recall': 1.00, 'f1-score': 0.99, 'support': 423},
        {'class': 'Orange___Haunglongbing_(Citrus_greening)', 'precision': 0.94, 'recall': 0.99, 'f1-score': 0.97, 'support': 503},
        {'class': 'Peach___Bacterial_spot', 'precision': 0.94, 'recall': 0.96, 'f1-score': 0.95, 'support': 459},
        {'class': 'Peach___healthy', 'precision': 0.99, 'recall': 0.97, 'f1-score': 0.98, 'support': 432},
        {'class': 'Pepper,_bell___Bacterial_spot', 'precision': 0.97, 'recall': 0.94, 'f1-score': 0.95, 'support': 478},
        {'class': 'Pepper,_bell___healthy', 'precision': 0.98, 'recall': 0.95, 'f1-score': 0.96, 'support': 497},
        {'class': 'Potato___Early_blight', 'precision': 0.92, 'recall': 0.99, 'f1-score': 0.96, 'support': 485},
        {'class': 'Potato___Late_blight', 'precision': 0.96, 'recall': 0.90, 'f1-score': 0.93, 'support': 485},
        {'class': 'Potato___healthy', 'precision': 0.96, 'recall': 0.98, 'f1-score': 0.97, 'support': 456},
        {'class': 'Raspberry___healthy', 'precision': 1.00, 'recall': 0.96, 'f1-score': 0.98, 'support': 445},
        {'class': 'Soybean___healthy', 'precision': 0.97, 'recall': 0.98, 'f1-score': 0.97, 'support': 505},
        {'class': 'Squash___Powdery_mildew', 'precision': 0.97, 'recall': 0.99, 'f1-score': 0.98, 'support': 434},
        {'class': 'Strawberry___Leaf_scorch', 'precision': 0.99, 'recall': 0.95, 'f1-score': 0.97, 'support': 444},
        {'class': 'Strawberry___healthy', 'precision': 1.00, 'recall': 0.99, 'f1-score': 0.99, 'support': 456},
        {'class': 'Tomato___Bacterial_spot', 'precision': 0.88, 'recall': 0.99, 'f1-score': 0.93, 'support': 425},
        {'class': 'Tomato___Early_blight', 'precision': 0.89, 'recall': 0.95, 'f1-score': 0.92, 'support': 480},
        {'class': 'Tomato___Late_blight', 'precision': 0.91, 'recall': 0.93, 'f1-score': 0.92, 'support': 463},
        {'class': 'Tomato___Leaf_Mold', 'precision': 0.95, 'recall': 0.96, 'f1-score': 0.95, 'support': 470},
        {'class': 'Tomato___Septoria_leaf_spot', 'precision': 0.90, 'recall': 0.87, 'f1-score': 0.89, 'support': 436},
        {'class': 'Tomato___Spider_mites Two-spotted_spider_mite', 'precision': 0.97, 'recall': 0.95, 'f1-score': 0.96, 'support': 435},
        {'class': 'Tomato___Target_Spot', 'precision': 0.96, 'recall': 0.89, 'f1-score': 0.92, 'support': 457},
        {'class': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'precision': 1.00, 'recall': 0.96, 'f1-score': 0.98, 'support': 490},
        {'class': 'Tomato___Tomato_mosaic_virus', 'precision': 0.96, 'recall': 0.99, 'f1-score': 0.98, 'support': 448},
        {'class': 'Tomato___healthy', 'precision': 0.97, 'recall': 0.99, 'f1-score': 0.98, 'support': 481}
    ]
    
    # Create DataFrame
    df = pd.DataFrame(classification_data)
    
    # Overall metrics
    accuracy = 0.96
    macro_avg = {'class': 'macro avg', 'precision': 0.96, 'recall': 0.96, 'f1-score': 0.96, 'support': 17572}
    weighted_avg = {'class': 'weighted avg', 'precision': 0.96, 'recall': 0.96, 'f1-score': 0.96, 'support': 17572}
    
    # Metrics summary in cards
    st.subheader("📈 Overall Performance")
    cols = st.columns(4)
    metrics = [
        ("Accuracy", f"{accuracy*100:.2f}%", "#4CAF50"),
        ("Macro Precision", f"{macro_avg['precision']*100:.2f}%", "#2196F3"),
        ("Weighted Recall", f"{weighted_avg['recall']*100:.2f}%", "#FF9800"),
        ("F1-Score", f"{macro_avg['f1-score']*100:.2f}%", "#9C27B0")
    ]
    
    for col, (title, value, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="custom-card" style="text-align: center; background: {color}; color: white; border-left: none;">
                <div style="font-size: 16px; margin-bottom: 8px;">{title}</div>
                <div style="font-size: 24px; font-weight: bold;">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Interactive visualizations
    st.subheader("📊 Performance Distribution")
    
    tab1, tab2, tab3, tab4 = st.tabs([
    "📏 Metrics Distribution",
    "🏆 Top/Bottom Performers",
    "📊 Class Distribution",
    "📋 Full Report"
])
    
    with tab1:
        st.markdown("""
        <div class="custom-card">
            <h3>Metrics Distribution</h3>
            <p>These box plots show the distribution of precision, recall, and F1-scores across all 38 disease classes. 
            The boxes represent the interquartile range (IQR), with the line inside showing the median value.</p>
        </div>
        """, unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.style.use('ggplot')
        # Precision distribution
        sns.boxplot(y='precision', data=df, ax=axes[0], color='#4CAF50')
        axes[0].set_title('Precision Distribution', fontsize=14, fontweight='bold')
        axes[0].set_ylim(0.8, 1.02)
        axes[0].set_ylabel('Precision', fontsize=12)
        
        # Recall distribution
        sns.boxplot(y='recall', data=df, ax=axes[1], color='#2196F3')
        axes[1].set_title('Recall Distribution', fontsize=14, fontweight='bold')
        axes[1].set_ylim(0.8, 1.02)
        axes[1].set_ylabel('Recall', fontsize=12)
        
        # F1-score distribution
        sns.boxplot(y='f1-score', data=df, ax=axes[2], color='#9C27B0')
        axes[2].set_title('F1-Score Distribution', fontsize=14, fontweight='bold')
        axes[2].set_ylim(0.8, 1.02)
        axes[2].set_ylabel('F1-Score', fontsize=12)
        
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        <div class="custom-card">
            <h4>Interpretation</h4>
            <ul>
                <li><strong>Precision:</strong> Measures how many of the predicted positive cases were actually positive</li>
                <li><strong>Recall:</strong> Measures how many actual positive cases were correctly identified</li>
                <li><strong>F1-Score:</strong> Harmonic mean of precision and recall (higher is better)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="custom-card">
            <h3>Top & Bottom Performers</h3>
            <p>These charts compare the best and worst performing disease classes based on F1-Scores. 
            Understanding these differences helps identify where the model excels and where it might need improvement.</p>
        </div>
        """, unsafe_allow_html=True)
        
        top5 = df.nlargest(5, 'f1-score')
        bottom5 = df.nsmallest(5, 'f1-score')

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        plt.style.use('ggplot')
        
        # Top 5 classes
        sns.barplot(x='f1-score', y='class', data=top5, ax=axes[0], palette='Greens_d')
        axes[0].set_title('Top 5 Performing Classes', fontsize=14, fontweight='bold')
        axes[0].set_xlim(0.9, 1.0)
        axes[0].set_xlabel('F1-Score', fontsize=12)
        axes[0].set_ylabel('Disease Class', fontsize=12)
        
        # Add value annotations
        for i, v in enumerate(top5['f1-score']):
            axes[0].text(v + 0.005, i, f"{v:.2f}", color='black', ha='left', va='center', fontweight='bold')
        
        # Bottom 5 classes
        sns.barplot(x='f1-score', y='class', data=bottom5, ax=axes[1], palette='Reds_d')
        axes[1].set_title('Bottom 5 Performing Classes', fontsize=14, fontweight='bold')
        axes[1].set_xlim(0.85, 0.95)
        axes[1].set_xlabel('F1-Score', fontsize=12)
        axes[1].set_ylabel('')
        
        # Add value annotations
        for i, v in enumerate(bottom5['f1-score']):
            axes[1].text(v + 0.005, i, f"{v:.2f}", color='black', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        <div class="custom-card">
            <h4>Analysis</h4>
            <p>The model performs exceptionally well on classes with:</p>
            <ul>
                <li>Clear visual symptoms</li>
                <li>Sufficient training examples</li>
                <li>Distinctive features compared to similar diseases</li>
            </ul>
            <p>Performance is lower for classes with:</p>
            <ul>
                <li>Symptom overlap with other diseases</li>
                <li>Limited training examples</li>
                <li>Subtle visual differences</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
        <div class="custom-card">
            <h3>Class Distribution</h3>
            <p>This histogram shows the distribution of samples across different plant disease classes. 
            A balanced distribution helps ensure the model learns to recognize all diseases equally well.</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.style.use('ggplot')
        
        # Use green color palette
        sns.histplot(df['support'], bins=15, kde=True, ax=ax, color='#4CAF50')
        ax.set_title('Number of Samples per Class', fontsize=16, fontweight='bold')
        ax.set_xlabel('Number of Samples', fontsize=12)
        ax.set_ylabel('Number of Classes', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add vertical line for mean
        mean_val = df['support'].mean()
        ax.axvline(mean_val, color='#FF9800', linestyle='--', linewidth=2)
        ax.text(mean_val+10, ax.get_ylim()[1]*0.9, f'Mean: {mean_val:.1f}', 
                fontsize=12, color='#FF9800', fontweight='bold')
        
        st.pyplot(fig)
        
        # Statistics summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Classes", len(df))
        col2.metric("Average Samples", f"{df['support'].mean():.1f}")
        col3.metric("Median Samples", df['support'].median())
    
    with tab4:
        st.markdown("""
        <div class="custom-card">
            <h3>Detailed Classification Report</h3>
            <p>This comprehensive report shows performance metrics for each individual disease class. 
            You can sort by any column to identify specific strengths and weaknesses.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a styled dataframe
        styled_df = df.sort_values('f1-score', ascending=False).reset_index(drop=True)
        styled_df = styled_df.style \
            .background_gradient(subset=['precision'], cmap='Greens') \
            .background_gradient(subset=['recall'], cmap='Blues') \
            .background_gradient(subset=['f1-score'], cmap='Purples') \
            .background_gradient(subset=['support'], cmap='Oranges') \
            .format({'precision': "{:.2f}", 'recall': "{:.2f}", 'f1-score': "{:.2f}"}) \
            .set_properties(**{'text-align': 'left', 'font-size': '12px'}) \
            .set_table_styles([{
                'selector': 'thead th',
                'props': [('background-color', '#2e7d32'), ('color', 'white'), ('font-weight', 'bold')]
            }])
        
        st.dataframe(styled_df, height=800)
        
        # Export option
        st.download_button(
            label="📥 Download Full Report as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='plant_disease_classification_report.csv',
            mime='text/csv'
        )
    
    # Averages table with enhanced styling
    st.subheader("📋 Performance Averages")
    st.markdown("""
    <div class="custom-card">
        <p>These averages provide a high-level summary of model performance across all classes:</p>
        <ul>
            <li><strong>Macro Avg:</strong> Unweighted mean of all classes</li>
            <li><strong>Weighted Avg:</strong> Average weighted by number of samples per class</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    avg_df = pd.DataFrame([macro_avg, weighted_avg]).set_index('class')
    st.table(avg_df.style.format("{:.2f}")
              .background_gradient(cmap='Greens')
              .set_properties(**{'font-size': '14px', 'text-align': 'center'})
              .set_table_styles([{
                  'selector': 'thead th',
                  'props': [('background-color', '#2e7d32'), ('color', 'white'), ('font-weight', 'bold')]
              }]))