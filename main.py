
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
    
       
    # Style options
    styles = ["realistic", "cartoon", "anime", "cyberpunk", "pixel art", "watercolor", "oil painting", "sketch", "3D render"]
    style = st.selectbox("Choose an image style", [""] + styles)

    # Prompt templates for styles
    style_prompts = {
    "realistic": (
        "An ultra high-resolution, photo-realistic image of {}, captured with a full-frame DSLR camera "
        "using a 50mm lens. Natural lighting, cinematic composition, soft shadows, realistic textures, "
        "depth of field, and lifelike environmental details. True-to-life color balance and reflections."
    ),

    "cartoon": (
        "A vibrant and expressive cartoon-style illustration of {}, featuring bold outlines, exaggerated features, "
        "and a playful color palette. Clear character design, smooth shading, and a clean, flat aesthetic suitable "
        "for animated series or comic books."
    ),

    "anime": (
        "A high-quality anime-style artwork of {}, with dynamic posing, dramatic lighting, and expressive facial features. "
        "Includes a detailed background in the style of modern anime, vibrant colors, cel-shading, and subtle glow effects. "
        "Inspired by Studio Ghibli and Makoto Shinkai aesthetics."
    ),

    "cyberpunk": (
        "A futuristic cyberpunk-themed depiction of {}, set in a neon-lit urban environment. Rich in sci-fi elements like "
        "holograms, cybernetic enhancements, rainy streets, reflective surfaces, and glowing signs. A moody, high-contrast "
        "color palette with deep purples, blues, and electric neon pinks."
    ),

    "pixel art": (
        "A retro 8-bit pixel art rendition of {}, with carefully crafted low-resolution details. Classic video game aesthetic, "
        "blocky textures, simplified shading, and a nostalgic color palette. Pixel-perfect outlines and sprite-style design."
    ),

    "watercolor": (
        "A delicate and expressive watercolor painting of {}, with fluid brush strokes, soft gradients, and subtle color blending. "
        "The image should feature paper texture, muted tones, and natural imperfections, evoking a hand-painted look "
        "with an airy, artistic feel."
    ),

    "oil painting": (
        "A traditional oil painting of {}, in the style of Renaissance or Baroque fine art. Rich, layered brushwork, "
        "realistic shadows and highlights, and a warm, earthy color palette. Classical composition and texture detail, "
        "with visible canvas strokes and chiaroscuro lighting."
    ),

    "sketch": (
        "A highly detailed pencil sketch of {}, drawn on textured paper. Includes fine linework, crosshatching, and shading to "
        "emphasize depth and contrast. Monochrome tones with subtle smudging, resembling a technical or anatomical illustration."
    ),

    "3D render": (
        "A professional 3D render of {}, modeled with intricate geometry and high-resolution textures. Realistic lighting setup with "
        "HDRI reflections, soft shadows, and physically accurate materials. Rendered in a photorealistic engine such as Blender Cycles "
        "or Unreal Engine."
    )
    }

    # Submit button
    submit = st.button("Generate Image")

    if submit and prompt:
        try:
            with st.spinner("Generating image..."):
                # Apply style-specific prompt
                full_prompt = style_prompts.get(style, "{}").format(prompt)
                
                image_urls = generate_image(full_prompt, style, "256x256")
                cols = st.columns(3)
                for i in range(3):
                    with cols[i]:
                        st.image(image_urls[i], caption=f"Image {i+1}")
                        
         # st.image(image_url, caption=f"Generated Image ({style or 'default'} style)", use_column_width=True)
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
            
            
 

















