
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
# st.image(image="aibg.jpeg")
st.title("Welcome to image converter")
# Set your OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = st.secrets['OPENAI_API_KEY']

mode=st.selectbox('What you want',['Select Mode','text to image','image to image'])

if mode == 'text to image':
        # Function to generate image
    def generate_image(prompt: str, style: str,image_size: str):
        styled_prompt = f"{prompt}, in {style} style" if style else prompt
        response = openai.images.generate(
            prompt=styled_prompt,
            n=3, #num of img
            size="256x256",
            model="dall-e-2",
            response_format="url"
        )
        return [img_data.url for img_data in response.data]

    # Streamlit UI
    st.title(" Text to Image Generator")
    prompt = st.text_input("Enter your image description:")
    
    styles = ["Ghibli art","realistic","3D render", "sketch","cartoon","anime","oil painting"]
    st.write("Choose image styles:")
    selected_styles = []

    for style in styles:
        if st.checkbox(style, key=style):
            selected_styles.append(style)

    submit = st.button("Generate Images")

    if submit and prompt:
        if len(selected_styles) >= 3:
            st.error("Please select 3 styles to generate images.")
        else:
            try:
                with st.spinner("Generating images..."):
                    image_urls = []
                    for style in selected_styles:
                        image_urls += generate_image(prompt, style, "256x256")
                    
                    # Display side by side
                    cols = st.columns(3)
                    for i in range(3):
                        with cols[i]:
                            st.image(image_urls[i], caption=f"{selected_styles[i]} style")
            except Exception as e:
                st.error(f"Error: {e}")
elif mode == 'image to image':
    
    st.title("Image to Image generator")
    st.markdown("Upload an image and describe the style you want (e.g., 'anime cat with big eyes').")
 
    # Upload image and input prompt
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    prompt = st.text_input("Enter your prompt", ) #value="anime style cat with big eyes"
    submit_button=st.button('generate image')
    if submit_button and uploaded_file and prompt:
        with st.spinner("Generating..."):
 
            # Load and display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            uploaded_resized = image.resize((256, 256))
            # st.image(image, caption="Original Uploaded Image")
 
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
            generated_resized = generated_image.resize((256, 256))

            st.subheader("📸 Uploaded vs 🎨 Generated Image")
            col1, col2 = st.columns(2)
 
            with col1:
                st.image(uploaded_resized, caption="Original Uploaded")
            with col2:
                st.image(generated_resized, caption="Generated Image")
            
            
 

















