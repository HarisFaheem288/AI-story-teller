import streamlit as st
import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
import pyttsx3
import torch
import os
import tempfile
import gdown
import zipfile

# -------- CONFIG --------
# Local FAISS files - already present locally
FAISS_INDEX_PATH = "stories_index.faiss"
FAISS_METADATA_PATH = "stories_metadata.json"

# Google Drive file IDs for zipped models
GDRIVE_ID_STORY_MODEL = "1B7mRxf7djcTV4zLwT-iq0OPH5xXfhn9k"
GDRIVE_ID_IMAGE_MODEL = "YOUR_GOOGLE_DRIVE_FILE_ID_LOCAL_MISTRAL2_ZIP"

# Where to extract models
MODEL_DIR = "models"
STORY_MODEL_DIR = os.path.join(MODEL_DIR, "local_mistral_model1")
IMAGE_MODEL_DIR = os.path.join(MODEL_DIR, "local_mistral_model2")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.title("📖 RAG Story Generator with Image & Audio")

def download_and_extract_gdrive_file(gdrive_id, extract_to):
    os.makedirs(MODEL_DIR, exist_ok=True)
    zip_path = os.path.join(MODEL_DIR, f"{gdrive_id}.zip")
    if not os.path.exists(extract_to):
        if not os.path.exists(zip_path):
            url = f"https://drive.google.com/uc?id={gdrive_id}"
            st.write(f"Downloading model from Google Drive (ID: {gdrive_id})...")
            gdown.download(url, zip_path, quiet=False)
        st.write(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        st.write(f"Model folder {extract_to} already exists, skipping download.")

@st.cache_resource(show_spinner=True)
def load_models():
    # Download & extract models from Drive if needed
    download_and_extract_gdrive_file(GDRIVE_ID_STORY_MODEL, STORY_MODEL_DIR)
    download_and_extract_gdrive_file(GDRIVE_ID_IMAGE_MODEL, IMAGE_MODEL_DIR)

    # Load FAISS index & metadata from local disk
    st.write("Loading FAISS index and metadata from local files...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load story generation model and tokenizer
    st.write("Loading story generation model...")
    tokenizer = AutoTokenizer.from_pretrained(STORY_MODEL_DIR)
    story_model = AutoModelForCausalLM.from_pretrained(
        STORY_MODEL_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    # Load image generation pipeline
    st.write("Loading image generation model...")
    image_pipe = StableDiffusionPipeline.from_pretrained(
        IMAGE_MODEL_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    return index, metadata, embed_model, tokenizer, story_model, image_pipe

# Load all models once
index, metadata, embed_model, tokenizer, story_model, image_pipe = load_models()

def retrieve_chunks(query, top_k=3):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(query_vec, top_k)
    return [metadata['texts'][i] for i in indices[0]]

def generate_story_with_context(query):
    context_chunks = retrieve_chunks(query)
    context_text = "\n".join(context_chunks)

    final_prompt = (
        f"The following text contains background information for writing a story. "
        f"Do NOT copy or list this background directly; just use it to inspire the writing.\n\n"
        f"Background:\n{context_text}\n\n"
        f"Now write a creative, original story based on: {query}\n"
        f"Only output the story itself, without mentioning the background."
    )

    inputs = tokenizer(final_prompt, return_tensors="pt").to(device)
    outputs = story_model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.8
    )

    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Background:" in story:
        story = story.split("Background:")[-1].strip()
    return story

def generate_image(prompt):
    image = image_pipe(prompt).images[0]
    img_path = os.path.join(tempfile.gettempdir(), "generated_image.png")
    image.save(img_path)
    return img_path

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)
    audio_path = os.path.join(tempfile.gettempdir(), "story_audio.wav")
    engine.save_to_file(text, audio_path)
    engine.runAndWait()
    return audio_path

# Streamlit UI
user_prompt = st.text_area("Enter your story idea:", height=150)

if st.button("Generate Story, Image & Audio") and user_prompt.strip():
    with st.spinner("Generating story..."):
        story = generate_story_with_context(user_prompt)
    st.subheader("📜 Generated Story")
    st.write(story)

    with st.spinner("Generating image..."):
        img_path = generate_image(user_prompt)
    st.image(img_path, caption="Generated Image", use_column_width=True)

    with st.spinner("Generating audio..."):
        audio_path = text_to_speech(story)

    if st.button("▶ Play Audio"):
        audio_bytes = open(audio_path, 'rb').read()
        st.audio(audio_bytes, format='audio/wav')
