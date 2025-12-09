import streamlit as st
import joblib
import nltk
import re
from nltk.stem import PorterStemmer
from nltk import pos_tag, word_tokenize

# --- SETUP ---
# Ensure NLTK data is available in the app environment
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger")

# --- FEATURE EXTRACTION (Must match training script exactly) ---
stemmer = PorterStemmer()


def get_orthographic_features(word):
    return {
        "is_title": word.istitle(),
        "is_all_caps": word.isupper(),
        "is_lower": word.islower(),
        "is_digit": word.isdigit(),
        "is_alnum": word.isalnum(),
        "has_dash": "-" in word,
        "has_slash": "/" in word,
        "has_greek": bool(
            re.search(r"(alpha|beta|gamma|delta|I|II|III|IV)", word, re.I)
        ),
    }


def extract_features(sentence_tokens):
    """
    Takes a list of tokens (strings) and returns a list of feature dictionaries.
    """
    # 1. POS Tagging
    try:
        pos_tags_list = pos_tag(sentence_tokens)
    except:
        pos_tags_list = [(w, "NN") for w in sentence_tokens]

    pos_tags = [pos for token, pos in pos_tags_list]

    features_list = []

    for i in range(len(sentence_tokens)):
        word = sentence_tokens[i]
        postag = pos_tags[i]

        # Base Features
        features = {
            "bias": 1.0,
            "word.lower()": word.lower(),
            "word.stem": stemmer.stem(word),
            "postag": postag,
            "prefix-1": word[:1],
            "prefix-2": word[:2],
            "prefix-3": word[:3],
            "suffix-1": word[-1:],
            "suffix-2": word[-2:],
            "suffix-3": word[-3:],
            "suffix-4": word[-4:],
        }

        features.update(get_orthographic_features(word))

        # Context: Previous Word
        if i > 0:
            word1 = sentence_tokens[i - 1]
            features.update(
                {
                    "-1:word.lower()": word1.lower(),
                    "-1:postag": pos_tags[i - 1],
                    "-1:word.istitle()": word1.istitle(),
                }
            )
        else:
            features["BOS"] = True

        # Context: Next Word
        if i < len(sentence_tokens) - 1:
            word1 = sentence_tokens[i + 1]
            features.update(
                {
                    "+1:word.lower()": word1.lower(),
                    "+1:postag": pos_tags[i + 1],
                    "+1:word.istitle()": word1.istitle(),
                }
            )
        else:
            features["EOS"] = True

        features_list.append(features)

    return features_list


# --- STREAMLIT UI ---
st.set_page_config(
    page_title="Disease NER System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for modern styling
st.markdown(
    """
<style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }

    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        font-weight: 400;
    }

    /* Card styling */
    .info-card {
        background: linear-gradient(145deg, #f8f9fc 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    /* Input area styling */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }

    /* Results container */
    .results-container {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        margin-top: 1.5rem;
    }

    .results-title {
        color: #1a202c;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Disease tag styling */
    .disease-tag {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        margin: 3px;
        display: inline-block;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(238, 90, 90, 0.3);
        transition: transform 0.2s ease;
    }

    .disease-tag:hover {
        transform: scale(1.05);
    }

    .disease-label {
        font-size: 0.7rem;
        opacity: 0.9;
        margin-left: 4px;
    }

    .normal-token {
        color: #4a5568;
        padding: 4px 6px;
        margin: 2px;
        display: inline-block;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f7fafc !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    /* Stats cards */
    .stat-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fc 100%);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }

    .stat-label {
        color: #718096;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #a0aec0;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
<div class="header-container">
    <div class="header-title">🧬 Disease Named Entity Recognition</div>
    <div class="header-subtitle">Powered by Conditional Random Fields • Trained on NCBI Disease Corpus</div>
</div>
""",
    unsafe_allow_html=True,
)

# Info card
st.markdown(
    """
<div class="info-card">
    <strong>💡 How it works:</strong> This AI model analyzes medical text and automatically identifies
    disease mentions using advanced Natural Language Processing techniques. Simply enter your text below
    and click analyze!
</div>
""",
    unsafe_allow_html=True,
)


# Load Model
@st.cache_resource
def load_model():
    try:
        return joblib.load("disease_ner_model.pkl")
    except FileNotFoundError:
        st.error(
            "⚠️ Model file 'disease_ner_model.pkl' not found. Please run the training script first!"
        )
        return None


model = load_model()

# Input section
st.markdown("### 📝 Enter Medical Text")
text_input = st.text_area(
    "Medical Text Input",
    "",
    height=120,
    placeholder="Type or paste medical text here...",
    label_visibility="collapsed",
)

# Center the button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_clicked = st.button("🔍 Analyze Text", width="stretch")

if analyze_clicked:
    if model and text_input.strip():
        with st.spinner("🔬 Analyzing text..."):
            # 1. Tokenize
            tokens = word_tokenize(text_input)

            # 2. Extract Features
            features = extract_features(tokens)

            # 3. Predict (CRF expects a list of lists of features)
            prediction = model.predict([features])[0]

            # Count diseases
            disease_tokens = [
                t for t, l in zip(tokens, prediction) if l in ["B-Disease", "I-Disease"]
            ]
            disease_count = sum(1 for l in prediction if l == "B-Disease")

        # Statistics row
        st.markdown("---")
        stat_col1, stat_col2, stat_col3 = st.columns(3)

        with stat_col1:
            st.markdown(
                f"""
            <div class="stat-card">
                <div class="stat-number">{len(tokens)}</div>
                <div class="stat-label">Total Tokens</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with stat_col2:
            st.markdown(
                f"""
            <div class="stat-card">
                <div class="stat-number">{disease_count}</div>
                <div class="stat-label">Diseases Found</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with stat_col3:
            accuracy = len(disease_tokens) / len(tokens) * 100 if tokens else 0
            st.markdown(
                f"""
            <div class="stat-card">
                <div class="stat-number">{accuracy:.1f}%</div>
                <div class="stat-label">Disease Coverage</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # 4. Display Results
        st.markdown("### 🎯 Analysis Results")

        # Create fancy output with highlighting
        result_html = '<div style="line-height: 2.2; padding: 1rem; background: #fafbfc; border-radius: 12px;">'
        for token, label in zip(tokens, prediction):
            if label == "B-Disease" or label == "I-Disease":
                result_html += f'<span class="disease-tag">{token}<span class="disease-label">({label})</span></span>'
            else:
                result_html += f'<span class="normal-token">{token}</span>'
        result_html += "</div>"

        st.markdown(result_html, unsafe_allow_html=True)

        # Disease summary
        if disease_count > 0:
            st.markdown("### 🏥 Detected Diseases")

            # Extract full disease names
            diseases = []
            current_disease = []
            for token, label in zip(tokens, prediction):
                if label == "B-Disease":
                    if current_disease:
                        diseases.append(" ".join(current_disease))
                    current_disease = [token]
                elif label == "I-Disease" and current_disease:
                    current_disease.append(token)
                else:
                    if current_disease:
                        diseases.append(" ".join(current_disease))
                        current_disease = []
            if current_disease:
                diseases.append(" ".join(current_disease))

            # Display as pills
            disease_pills = " ".join(
                [
                    f'<span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 8px 16px; border-radius: 20px; margin: 4px; display: inline-block; font-weight: 500;">🔴 {d}</span>'
                    for d in diseases
                ]
            )
            st.markdown(
                f'<div style="margin-top: 1rem;">{disease_pills}</div>',
                unsafe_allow_html=True,
            )

        # 5. Show Raw Data
        with st.expander("📊 View Raw Tags", expanded=False):
            st.markdown("""
            <div style="background: linear-gradient(145deg, #f0f4ff 0%, #ffffff 100%); padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
                <p style="color: #4a5568; margin: 0;">🏷️ <strong>Token-to-Label Mapping:</strong> Each token is mapped to its predicted BIO tag (B-Disease, I-Disease, or O)</p>
            </div>
            """, unsafe_allow_html=True)

            # Create a styled table for raw tags
            tag_data = []
            for i, (token, label) in enumerate(zip(tokens, prediction)):
                tag_type = "🔴 Disease Start" if label == "B-Disease" else "🟠 Disease Cont." if label == "I-Disease" else "⚪ Outside"
                tag_data.append({"#": i + 1, "Token": token, "Tag": label, "Type": tag_type})

            import pandas as pd
            df_tags = pd.DataFrame(tag_data)

            # Custom styling for the dataframe
            st.markdown("""
            <style>
            .tag-table {
                font-family: 'Segoe UI', sans-serif;
            }
            </style>
            """, unsafe_allow_html=True)

            st.dataframe(
                df_tags,
                width='stretch',
                hide_index=True,
                column_config={
                    "#": st.column_config.NumberColumn("#", width="small"),
                    "Token": st.column_config.TextColumn("Token", width="medium"),
                    "Tag": st.column_config.TextColumn("Tag", width="medium"),
                    "Type": st.column_config.TextColumn("Type", width="medium"),
                }
            )

            # Tag distribution summary
            tag_counts = {}
            for label in prediction:
                tag_counts[label] = tag_counts.get(label, 0) + 1

            st.markdown("#### 📊 Tag Distribution")
            dist_cols = st.columns(len(tag_counts))
            colors = {"B-Disease": "#ff6b6b", "I-Disease": "#ffa502", "O": "#a4b0be"}
            for idx, (tag, count) in enumerate(tag_counts.items()):
                with dist_cols[idx]:
                    color = colors.get(tag, "#667eea")
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}20 0%, {color}10 100%);
                                padding: 1rem; border-radius: 10px; text-align: center;
                                border: 2px solid {color};">
                        <div style="font-size: 1.5rem; font-weight: 700; color: {color};">{count}</div>
                        <div style="color: #4a5568; font-size: 0.85rem; font-weight: 500;">{tag}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # 6. Show Probability Marginals
        with st.expander("📈 View Confidence Scores", expanded=False):
            st.markdown("""
            <div style="background: linear-gradient(145deg, #fff5f5 0%, #ffffff 100%); padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
                <p style="color: #4a5568; margin: 0;">🎯 <strong>Prediction Confidence:</strong> Probability scores for each possible label. Higher values indicate stronger model confidence.</p>
            </div>
            """, unsafe_allow_html=True)

            marginals = model.predict_marginals([features])[0]

            # Create enhanced confidence dataframe
            import pandas as pd
            confidence_data = []
            for i, (token, probs) in enumerate(zip(tokens, marginals)):
                row = {
                    "#": i + 1,
                    "Token": token,
                    "Predicted": prediction[i],
                }
                # Add probability columns as percentages (0-100)
                for label in sorted(probs.keys()):
                    row[f"P({label})"] = probs.get(label, 0.0) * 100
                # Add confidence indicator as percentage
                max_prob = max(probs.values()) * 100
                row["Confidence"] = max_prob
                confidence_data.append(row)

            df_conf = pd.DataFrame(confidence_data)

            # Display with conditional formatting
            st.dataframe(
                df_conf,
                width='stretch',
                hide_index=True,
                column_config={
                    "#": st.column_config.NumberColumn("#", width="small"),
                    "Token": st.column_config.TextColumn("Token", width="medium"),
                    "Predicted": st.column_config.TextColumn("Predicted", width="small"),
                    "P(B-Disease)": st.column_config.ProgressColumn(
                        "P(B-Disease) %",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "P(I-Disease)": st.column_config.ProgressColumn(
                        "P(I-Disease) %",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "P(O)": st.column_config.ProgressColumn(
                        "P(O) %",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence %",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                }
            )

            # Confidence summary
            st.markdown("#### 📉 Confidence Analysis")
            avg_confidence = sum(max(m.values()) for m in marginals) / len(marginals)
            low_conf_tokens = [(t, max(m.values())) for t, m in zip(tokens, marginals) if max(m.values()) < 0.7]

            conf_col1, conf_col2 = st.columns(2)
            with conf_col1:
                conf_color = "#27ae60" if avg_confidence > 0.8 else "#f39c12" if avg_confidence > 0.6 else "#e74c3c"
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {conf_color}20 0%, {conf_color}10 100%);
                            padding: 1.2rem; border-radius: 12px; text-align: center;
                            border: 2px solid {conf_color};">
                    <div style="font-size: 2rem; font-weight: 700; color: {conf_color};">{avg_confidence:.1%}</div>
                    <div style="color: #4a5568; font-size: 0.9rem; font-weight: 500;">Average Confidence</div>
                </div>
                """, unsafe_allow_html=True)

            with conf_col2:
                uncertain_count = len(low_conf_tokens)
                unc_color = "#27ae60" if uncertain_count == 0 else "#f39c12" if uncertain_count < 3 else "#e74c3c"
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {unc_color}20 0%, {unc_color}10 100%);
                            padding: 1.2rem; border-radius: 12px; text-align: center;
                            border: 2px solid {unc_color};">
                    <div style="font-size: 2rem; font-weight: 700; color: {unc_color};">{uncertain_count}</div>
                    <div style="color: #4a5568; font-size: 0.9rem; font-weight: 500;">Low Confidence Tokens (&lt;70%)</div>
                </div>
                """, unsafe_allow_html=True)

            if low_conf_tokens:
                st.markdown("##### ⚠️ Tokens with Low Confidence:")
                for token, conf in low_conf_tokens:
                    st.markdown(f"- **{token}**: {conf:.1%}")
    elif not text_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown(
    """
<div class="footer">
    Made with ❤️ using Streamlit • CRF Model trained on NCBI Disease Corpus
</div>
""",
    unsafe_allow_html=True,
)
