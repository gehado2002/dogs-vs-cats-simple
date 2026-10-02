import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Dogs vs Cats", page_icon="🐶")
st.title("🐶🐱 Dogs vs Cats Classifier")

BASE = Path(__file__).parent / "models"

# cat = 0, dog = 1  (sigmoid output = probability of dog)
MODELS = {
    "VGG16 (fine-tuned)": {"file": "vgg16_best_model.keras", "size": (224, 224), "threshold": 0.5},
    "Custom CNN":         {"file": "cnn_best_model.keras",   "size": (150, 150), "threshold": 0.4},
}

available = {n: c for n, c in MODELS.items() if (BASE / c["file"]).exists()}
if not available:
    st.error("Put cnn_best_model.keras and/or vgg16_best_model.keras inside the models/ folder.")
    st.stop()


@st.cache_resource
def get_model(file):
    return load_model(BASE / file, compile=False)


name = st.sidebar.selectbox("Model", list(available))
cfg = available[name]
threshold = st.sidebar.slider("Dog threshold", 0.1, 0.9, cfg["threshold"], 0.05)
model = get_model(cfg["file"])

files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

for f in files:
    img = Image.open(f)
    x = ImageOps.exif_transpose(img).convert("RGB").resize(cfg["size"], Image.NEAREST)
    x = np.expand_dims(np.asarray(x, dtype="float32") / 255.0, 0)

    p_dog = float(model.predict(x, verbose=0)[0][0])
    is_dog = p_dog > threshold

    c1, c2 = st.columns(2)
    c1.image(img, use_container_width=True)
    c2.subheader("Dog 🐶" if is_dog else "Cat 🐱")
    c2.metric("Confidence", f"{(p_dog if is_dog else 1 - p_dog) * 100:.1f}%")
    st.divider()
