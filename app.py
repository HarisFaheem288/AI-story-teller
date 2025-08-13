import streamlit as st
import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
import pyttsx3
import torch
from PIL import Image
import re

# =========================
# Project Settings
# =========================
PROJECT_NAME = "FableForge: AI Story Studio" 
story_model_path = "local_mistral_model1"    # your local LLM folder (looks like a Phi-3 model from the stack)
image_model_id  = "local_mistral_model2"     # fast, light SD
faiss_index_path = "stories_index.faiss"
faiss_metadata_path = "stories_metadata.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Page Config & Styling
# =========================
st.set_page_config(page_title="FableForge", page_icon="🪄", layout="wide")
st.markdown("""
<style>
section.main > div { padding-top: 1rem; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
h1, h2, h3 { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
.ff-card {
  background: linear-gradient(180deg, rgba(250,250,250,1) 0%, rgba(245,245,245,1) 100%);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}
div.stButton>button { border-radius: 14px; padding: 0.6rem 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title(PROJECT_NAME)
st.caption("Create immersive tales with image art and optional voice narration — all in one elegant studio.")

# =========================
# Cache Lightweight Stuff
# =========================
@st.cache_resource
def load_faiss_and_embedder():
    index = faiss.read_index(faiss_index_path)
    with open(faiss_metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return index, metadata, embed_model

index, metadata, embed_model = load_faiss_and_embedder()

# =========================
# Cache Heavy Models Separately (loaded only when used)
# =========================
@st.cache_resource
def load_story_model():
    tokenizer = AutoTokenizer.from_pretrained(story_model_path)
    # ensure pad token exists to avoid generation warnings/issues
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        story_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",                    # accelerate offload
        low_cpu_mem_usage=True
    )
    return tokenizer, model

@st.cache_resource
def load_image_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(
        image_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    if device == "cuda":
        pipe.enable_model_cpu_offload()  # safer VRAM usage
    else:
        pipe = pipe.to(device)
    return pipe

# =========================
# Utilities
# =========================
def retrieve_chunks(query, top_k=3):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(query_vec, top_k)
    return [metadata['texts'][i] for i in indices[0]]

def clean_story_output(text: str) -> str:
    # Remove any accidental leakage like 'Background:', etc.
    # Keep only the story content.
    # Strip anything before well-known markers if echoed
    for m in ["Secret inspiration:", "Background:", "Story request:", "Inspiration:"]:
        if m in text:
            text = text.split(m)[-1]
    # Remove any trailing prompt echoes
    text = re.sub(r"(Secret inspiration:|Background:|Story request:).*", "", text, flags=re.IGNORECASE|re.DOTALL)
    return text.strip()

def generate_story_with_context(query, max_new_tokens=450, temperature=0.85):
    tokenizer, story_model = load_story_model()
    context_chunks = retrieve_chunks(query, top_k=3)
    context_text = "\n".join(context_chunks)

    # Safer prompt that discourages echoing the background
    final_prompt = (
        "You are a creative author. You have private notes to inspire your writing. "
        "NEVER reveal or repeat those notes verbatim. Write a vivid, cohesive story.\n\n"
        f"Private notes:\n{context_text}\n\n"
        f"Story request: {query}\n\n"
        "Output ONLY the story text. Do not mention private notes, background, or instructions."
    )

    # IMPORTANT: Do NOT move inputs to device when using device_map='auto'
    inputs = tokenizer(
        final_prompt,
        return_tensors="pt",
        padding=True
    )
    # Let Accelerate handle placement; don't .to(device)
    with torch.no_grad():
        outputs = story_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id
        )
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    story = clean_story_output(raw)
    return story

def generate_image(prompt, width=384, height=384):
    pipe = load_image_pipe()
    image = pipe(prompt, width=width, height=height).images[0]
    return image

def text_to_speech(text, rate=160, volume=1.0, out_path="story_audio.wav"):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    return out_path

# =========================
# Sidebar Controls (Dashboard feel)
# =========================
with st.sidebar:
    st.header(" Controls") 
    st.write("Tune your generation settings.")
    story_len = st.slider("Story length (tokens)", 200, 800, 450, 50)
    temp = st.slider("Creativity (temperature)", 0.3, 1.2, 0.85, 0.05)
    img_size = st.selectbox("Image size", ["384×384", "512×512", "640×640"], index=0)
    size_map = {"384×384": (384,384), "512×512": (512,512), "640×640": (640,640)}
    w, h = size_map[img_size]
    st.divider()
    st.caption("Tip: Start smaller sizes if your GPU RAM is limited.") 

# =========================
# Main UI
# =========================
with st.container():
    st.markdown('<div class="ff-card">', unsafe_allow_html=True)
    prompt = st.text_input("Your Story Idea", placeholder="Once upon a time in a magical forest, a tiny fox found a glowing blue crystal...") 
    c1, c2 = st.columns([2, 1])  # story wide (left), image narrow (right)

    # Session state to persist outputs across reruns
    if "story" not in st.session_state: st.session_state.story = ""
    if "image" not in st.session_state: st.session_state.image = None
    if "audio_path" not in st.session_state: st.session_state.audio_path = None

    gen = st.button("🪄 Generate Story & Image", use_container_width=True)

    if gen:
        if not prompt.strip():
            st.warning("Please enter a story idea first.")
        else:
            with st.spinner("Writing your story..."):
                st.session_state.story = generate_story_with_context(prompt, max_new_tokens=story_len, temperature=temp)

            with st.spinner("Painting your image..."):
                st.session_state.image = generate_image(prompt, width=w, height=h)

            st.session_state.audio_path = None  # reset audio until user asks
            st.success("Done! Scroll to view your results.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Display Results
    with c1:
        st.subheader("Your Story 📖")
        if st.session_state.story:
            st.markdown(f'<div class="ff-card">{st.session_state.story}</div>', unsafe_allow_html=True)
        else:
            st.info("Your story will appear here after generation.")

    with c2:
        st.subheader("Artwork 🎨") 
        if st.session_state.image is not None:
            st.image(st.session_state.image, caption="Generated Image", use_column_width=True)
        else:
            st.info("Your image will appear here after generation.")

    st.divider()

    # Narration controls (separate button)
    st.subheader(" Narration🔊")
    colA, colB = st.columns([1, 3])
    with colA:
        speak = st.button(" Create & Play Audio", use_container_width=True, disabled=not bool(st.session_state.story)) 
    with colB:
        st.caption("Click to synthesize voice **after** the story is ready.")

    if speak and st.session_state.story:
        with st.spinner("Synthesizing narration..."):
            st.session_state.audio_path = text_to_speech(st.session_state.story)
        with open(st.session_state.audio_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")
        st.success("Narration ready!")

    # If audio already exists from a prior click, show the player
    if st.session_state.audio_path and not speak:
        with open(st.session_state.audio_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")
