import streamlit as st
from PIL import Image

from backend.classifier import classify_image
from backend.rag import get_ingredients, get_healthier_alternatives, seed_collection


@st.cache_resource
def _init():
    seed_collection()


_init()

st.set_page_config(
    page_title="PlateMate",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* ---------- Base ---------- */
    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2E7D32, #66BB6A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #777;
        font-size: 1.1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #333;
        border-bottom: 2px solid #E8F5E9;
        padding-bottom: 0.4rem;
        margin-top: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: #F1F8E9;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* ---------- Mobile (<640px) ---------- */
    @media (max-width: 640px) {
        .main-title { font-size: 2rem; }
        .subtitle   { font-size: 0.95rem; }

        /* Full-width buttons & inputs */
        button, .stButton > button,
        [data-testid="stFileUploader"],
        [data-testid="stCameraInput"] {
            width: 100% !important;
        }

        /* Prevent iOS zoom on input focus */
        input, select, textarea { font-size: 16px !important; }

        /* Larger tap targets */
        .stButton > button {
            min-height: 3rem;
            font-size: 1.05rem !important;
        }

        /* Stack metric cards vertically */
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">🍽️ PlateMate</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Snap or upload a food photo for AI-powered nutritional insights</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About")
    st.write(
        "**PlateMate** uses computer vision to identify foods, "
        "then retrieves nutritional advice from a curated knowledge base "
        "using Retrieval-Augmented Generation (RAG)."
    )
    st.divider()
    st.markdown(
        "**Tech stack:** FastAPI · ChromaDB · HuggingFace Transformers · "
        "Google Gemini · Streamlit"
    )

# --------------- Image input ---------------
tab_upload, tab_camera = st.tabs(["📁 Upload", "📷 Camera"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a food image",
        type=["jpg", "jpeg", "png", "webp"],
    )

with tab_camera:
    camera_photo = st.camera_input("Take a photo of your food")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif camera_photo is not None:
    image = Image.open(camera_photo).convert("RGB")

if image is not None:
    st.image(image, caption="Your image", use_container_width=True)

    if st.button("🔍  Analyze", type="primary", use_container_width=True):
        with st.spinner("Classifying image..."):
            predictions = classify_image(image)

        if not predictions:
            st.error("Classification returned no results.")
        else:
            top_food = predictions[0]["label"]
            confidence = predictions[0]["confidence"]

            st.markdown(
                '<p class="section-header">🎯 Classification</p>',
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)
            col1.metric("Detected Food", top_food.replace("_", " ").title())
            col2.metric("Confidence", f"{confidence * 100:.1f}%")

            if len(predictions) > 1:
                with st.expander("Other possibilities"):
                    for p in predictions[1:]:
                        st.write(
                            f"- **{p['label'].replace('_', ' ').title()}** "
                            f"— {p['confidence'] * 100:.1f}%"
                        )

            with st.spinner("Fetching ingredients..."):
                ingredients = get_ingredients(top_food)
            st.markdown(
                '<p class="section-header">📝 Ingredients</p>',
                unsafe_allow_html=True,
            )
            st.write(ingredients)

            with st.spinner("Generating healthier alternatives via RAG..."):
                alternatives, sources_used = get_healthier_alternatives(top_food)
            st.markdown(
                '<p class="section-header">💡 Healthier Alternatives</p>',
                unsafe_allow_html=True,
            )
            st.info(
                f"Retrieved from {sources_used} knowledge-base documents via RAG"
            )
            st.write(alternatives)
else:
    st.info("👆 Upload a food image or snap a photo to get started.")
