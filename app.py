import streamlit as st
import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
import pyttsx3
import torch
from PIL import Image

# ======== SETTINGS ========
story_model_path = "microsoft/Phi-3-mini-4k-instruct"  # Your Phi-3 model path
image_model_id = "CompVis/stable-diffusion-v1-4"    # Lightweight image generation model
faiss_index_path = "stories_index.faiss"
faiss_metadata_path = "stories_metadata.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

st.title("📚 Story + 🎨 Image + 🔊 Voice Generator")

@st.cache_resource
def load_resources():
    # Load FAISS index
    index = faiss.read_index(faiss_index_path)
    with open(faiss_metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Embedding model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Story model
    tokenizer = AutoTokenizer.from_pretrained(story_model_path)
    story_model = AutoModelForCausalLM.from_pretrained(
        story_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    # Image model
    image_pipe = StableDiffusionPipeline.from_pretrained(
        image_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    return index, metadata, embed_model, tokenizer, story_model, image_pipe

index, metadata, embed_model, tokenizer, story_model, image_pipe = load_resources()

# ======== Retrieval Function ========
def retrieve_chunks(query, top_k=3):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(query_vec, top_k)
    return [metadata['texts'][i] for i in indices[0]]

# ======== Story Generation Function ========
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

# ======== Image Generation Function ========
def generate_image(prompt):
    image = image_pipe(prompt).images[0]
    return image

# ======== Text-to-Speech Function ========
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.setProperty('volume', 1.0)
    audio_path = "story_audio.wav"
    engine.save_to_file(text, audio_path)
    engine.runAndWait()
    return audio_path

# ======== Streamlit UI ========
prompt = st.text_input("Enter your story idea:")

if st.button("Generate Story & Image"):
    if prompt:
        with st.spinner("Generating story..."):
            story = generate_story_with_context(prompt)
        st.subheader("📖 Generated Story")
        st.write(story)

        with st.spinner("Generating image..."):
            img = generate_image(prompt)
        st.image(img, caption="Generated Image", use_column_width=True)

        with st.spinner("Generating audio..."):
            audio_path = text_to_speech(story)
        audio_file = open(audio_path, "rb")
        st.audio(audio_file.read(), format="audio/wav")

        st.success("✅ Done!")
    else:
        st.warning("Please enter a prompt.")
