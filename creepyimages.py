import streamlit as st
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import time
import os
from PIL import Image
import io

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(page_title="Strange valley Generator", layout="centered")
st.title("Strange valley Generator")

# ------------------------------
# PROMPT LIBRARY
# ------------------------------
PROMPTS = {
    "Surreal Glitch World (Original)": 
        "A surreal, glitchy environment that reflects the player’s perception. Rooms bend and ripple, mirrors reflect impossible scenes, lights flicker. Objects shift shape, textures glitch like corrupted digital data. Colors change with emotion: warm tones for curiosity, cold for unease. Faint static or digital noise distorts surfaces. The world feels alive, dreamlike, slightly unstable.",

    "Liminal Backrooms Hallway":
        "An endless yellow hallway lit by flickering fluorescent lights, stained carpet, peeling wallpaper, low ceiling, the feeling of being watched, silence so loud it hurts, uncanny emptiness, liminal space, unsettling atmosphere",

    "Abandoned Mall at Night":
        "A dead shopping mall frozen in time, dim emergency lights, empty storefronts, dusty floors, abandoned food court, quiet echoes, eerie stillness, early 2000s aesthetic, unsettling liminality",

    "Uncanny Human Face":
        "A hyper-realistic human face that looks almost right but slightly wrong, unnatural proportions, unsettling stare, waxy skin, subtle distortion, analog horror aesthetic, uncanny valley effect",

    "Endless Office Reality":
        "A sprawling infinite office building, buzzing fluorescent lights, repeating hallways, beige cubicles, ancient computers still on, no people, quiet hum, forgotten corporate labyrinth",

    "Poolrooms Nightmare":
        "An infinite indoor pool structure, pale yellow tiles, warm stagnant water, echoing drips, endless corners, underground atmosphere, dreamlike architecture, surreal horror",

    "Entity in the Distance":
        "A dark humanoid figure barely visible at the end of a hallway, too tall, wrong shape, blurred edges, watching silently, reality distortion around it, dread in the air",

    "Glitched Childhood Memory":
        "A playground from a forgotten childhood memory, broken swings, gray sky, VHS distortion, melted shapes, eerie nostalgia, soft colors, corrupted dream logic",

    "Impossible Staircase":
        "A staircase that bends through space, stairs leading into darkness, upside-down rooms, shadow figures in corners, surreal geometry, psychological horror",

    "Lost VHS Broadcast":
        "A corrupted television broadcast, static-filled screen, warped faces, emergency message repeating, analog interference, unknown language, creepy late-night horror vibe"
}

# ------------------------------
# LOAD MODEL ONCE
# ------------------------------
@st.cache_resource
def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    pipe.enable_attention_slicing()
    pipe.enable_cpu_offload()

    return pipe


pipe = load_pipeline()

# ------------------------------
# UI - PROMPT PICKER
# ------------------------------
st.subheader("🧩 Prompt Library")

selected_prompt = st.selectbox("Choose a prompt style:", list(PROMPTS.keys()))

# editable prompt box
prompt = st.text_area(
    "Prompt",
    height=150,
    value=PROMPTS[selected_prompt]
)

# ------------------------------
# SETTINGS
# ------------------------------
col1, col2 = st.columns(2)
with col1:
    width = st.number_input("Width", min_value=256, max_value=1024, value=512, step=64)
    steps = st.slider("Steps", 10, 50, 20)

with col2:
    height = st.number_input("Height", min_value=256, max_value=1024, value=512, step=64)
    guidance = st.slider("Guidance Scale", 1.0, 15.0, 7.5)

# ------------------------------
# SAVE OPTIONS
# ------------------------------
st.subheader("💾 Save Options")

save_mode = st.radio(
    "When downloading:",
    ["Browser download", "Save to folder path"]
)

folder_path = None
if save_mode == "Save to folder path":
    folder_path = st.text_input("Folder path (example: C:/Users/You/Pictures)")

filename = st.text_input("Filename", value="neuroreality.png")

# ------------------------------
# GENERATION
# ------------------------------
if st.button("🎨 Generate Image"):
    with st.spinner("Generating..."):
        image, gen_time = None, None
        image, gen_time = (
            pipe(
                prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance
            ).images[0],
            "Done"  # Display-only; pipeline handles time internally
        )

    st.image(image, caption="Generated Image", use_container_width=True)
    st.success("Generation complete")

    # Convert image for download
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_data = buf.getvalue()

    # ------------------------------
    # SAVE LOGIC
    # ------------------------------
    if save_mode == "Save to folder path" and folder_path:

        if st.button("📁 Save to Folder"):
            try:
                os.makedirs(folder_path, exist_ok=True)
                save_path = os.path.join(folder_path, filename)
                with open(save_path, "wb") as f:
                    f.write(img_data)
                st.success(f"Saved to: {save_path}")
            except Exception as e:
                st.error(f"Save failed: {e}")

    else:
        st.download_button(
            "⬇ Download Image",
            data=img_data,
            file_name=filename,
            mime="image/png"
        )


