
# this is with styles using dropdown
from langsmith.wrappers import wrap_openai
from langsmith import traceable
import streamlit as st
import openai
import os
import streamlit as st
from PIL import Image
import torch
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
# from utils import image_to_canny
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image

#canny function for model
def image_to_canny(pil_img):
    image = np.array(pil_img)
    image = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    canny_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(canny_image)

#st.set_page_config(page_title="AI Art Generator", layout="centered")
load_dotenv()
st.image(image="aibg.jpeg")
st.title("Welcome to image converter")
# Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = st.secrets['OPENAI_API_KEY']

mode=st.selectbox('What you want',['Select Mode','text to image','image to image'])

if mode == 'text to image':
        # Function to generate image
    def generate_image(prompt: str, style: str, image_size: str):
        styled_prompt = f"{prompt}, in {style} style" if style else prompt
        response = openai.images.generate(
            prompt=styled_prompt,
            n=1,
            size=image_size,
            response_format="url"
        )
        return response.data[0].url

    # Streamlit UI
    st.title(" Text to Image Generator")

    prompt = st.text_input("Enter your image description:")
    style = st.selectbox(
        "Choose an image style",
        ["", "realistic", "cartoon", "anime", "cyberpunk", "pixel art", "watercolor", "oil painting", "sketch", "3D render"]
    )
    image_size = st.selectbox("Choose image size", ["256x256", "512x512", "1024x1024"])
    submit = st.button("Generate Image")

    if submit and prompt:
        try:
            with st.spinner("Generating image..."):
                image_url = generate_image(prompt, style, image_size)
                st.image(image_url, caption=f"Generated Image ({style or 'default'} style)", use_column_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
            
        
elif mode == 'image to image':
    #st.set_page_config(page_title="Image to Anime using ControlNet", layout="centered")

    st.title("Image to image generator")
    st.markdown("Upload an image and describe the style you want (e.g., 'anime cat with big eyes').")

    # Upload image and input prompt
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    prompt = st.text_input("Enter your prompt", ) #value="anime style cat with big eyes"
    submit_button=st.button('generate image')
    if submit_button and uploaded_file and prompt:
        with st.spinner("Generating..."):

            # Load and display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.subheader("📸 Uploaded Image")
            st.image(image, caption="Original Uploaded Image", use_column_width=True)

            # Convert to Canny edges
            canny_image = image_to_canny(image)

            # Load ControlNet model
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny", 
                torch_dtype=torch.float32  # For CPU use
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float32
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

            # Move model to CPU
            pipe = pipe.to("cpu")

            # Generate the image
            result = pipe(prompt, image=canny_image, num_inference_steps=5)
            generated_image = result.images[0]

            # Save and display output
            os.makedirs("output", exist_ok=True)
            out_path = os.path.join("output", "generated.png")
            generated_image.save(out_path)

            # Show result
            st.subheader("🎨 Generated Image")
            st.image(generated_image, caption="Generated Image", use_column_width=True)
            st.success("Done!")

            # Download option
            with open(out_path, "rb") as file:
                st.download_button("Download Image", file, "anime_output.png", "image/png")
            
            
 

















