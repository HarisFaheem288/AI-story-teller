
**StoryCraft AI** is a Streamlit-based interactive dashboard that brings your creative ideas to life!  
Enter a short story idea, and the app will:

1. **Generate an original story** (without copying background text).
2. **Create an AI-generated image** to match the story.
3. **Narrate the story** using realistic text-to-speech — playback starts only when you click a button.


---

## 🚀 Features

- 📝 **AI Story Generation** — Create engaging, original tales from simple prompts.
- 🎨 **Image Generation** — Visualize your story with Stable Diffusion Turbo.
- 🔊 **Voice Narration** — Listen to your story with smooth text-to-speech.
- ⚡ **Optimized Performance** — Models load only when needed to save memory.
- 🎛️ **Interactive Dashboard** — Modern UI built with Streamlit.

---

## 🛠️ Tech Stack

- **[Python 3.10+]**
- **[Streamlit]
- **[FAISS]
- **[Sentence Transformers]
- **[Transformers]
- **[Diffusers]
- **[Pyttsx3]
- **[Torch]

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/storycraft-ai.git
   cd storycraft-ai
````

2. **Create a virtual environment & activate it**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download or place your local models**

   * Story generation model → Place in `local_mistral_model1`
   * FAISS index → `stories_index.faiss`
   * Metadata JSON → `stories_metadata.json`

---

## ▶️ Usage

Run the app:

```bash
streamlit run app.py
```

Then:

1. Enter your story idea in the input box.
2. Click **Generate** — wait for story + image to appear.
3. Click the **Play Audio** button to listen.

---

## 📂 Project Structure

```
storycraft-ai/
│
├── app.py                     # Main Streamlit app
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── local_mistral_model1/      # Local LLM for story generation
├── stories_index.faiss        # FAISS index file
├── stories_metadata.json      # Metadata for retrieval
└── assets/                    # Optional images, logos
```

---

## 📸 Screenshots

> Replace these placeholders with your screenshots after running the app.

**Main Dashboard**
![Main Dashboard](assets/screenshot_main.png)

**Generated Story & Image**
![Generated Story](assets/screenshot_story.png)

---

## 💡 Tips

* For best performance, run on a machine with a GPU (CUDA supported).
* The `sd-turbo` model is chosen for speed; you can swap with any SD model on Hugging Face.
* You can tweak **temperature** and **max\_new\_tokens** in the code for more creative control.

---

---

## 🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss what you would like to change.

---

**Made with ❤️ using Python, Streamlit & Hugging Face**

```
