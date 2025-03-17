import json
import os
import time
import uuid
import tempfile
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import base64
import mimetypes

from google import genai
from google.genai import types

def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)

def generate(text, file_name, api_key, model="gemini-2.0-flash-exp"):
    # Initialize client using provided api_key (or fallback to env variable)
    client = genai.Client(api_key=(api_key.strip() if api_key and api_key.strip() != ""
                                     else os.environ.get("GEMINI_API_KEY")))
    
    files = [ client.files.upload(file=file_name) ]
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_text(text=text),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_modalities=["image", "text"],
        response_mime_type="text/plain",
    )

    text_response = ""
    image_path = None
    # Create a temporary file to potentially store image data.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            candidate = chunk.candidates[0].content.parts[0]
            # Check for inline image data
            if candidate.inline_data:
                save_binary_file(temp_path, candidate.inline_data.data)
                print(f"File of mime type {candidate.inline_data.mime_type} saved to: {temp_path} and prompt input: {text}")
                image_path = temp_path
                # If an image is found, we assume that is the desired output.
                break
            else:
                # Accumulate text response if no inline_data is present.
                text_response += chunk.text + "\n"
    
    del files
    return image_path, text_response

def process_image_and_prompt(composite_pil, prompt, gemini_api_key):
    try:
        # Save the composite image to a temporary file.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            composite_path = tmp.name
            composite_pil.save(composite_path)
        
        file_name = composite_path  
        input_text = prompt 
        model = "gemini-2.0-flash-exp" 

        image_path, text_response = generate(text=input_text, file_name=file_name, api_key=gemini_api_key, model=model)
        
        if image_path:
            # Load and convert the image if needed.
            result_img = Image.open(image_path)
            if result_img.mode == "RGBA":
                result_img = result_img.convert("RGB")
            return [result_img], ""  # Return image in gallery and empty text output.
        else:
            # Return no image and the text response.
            return None, text_response
    except Exception as e:
        raise gr.Error(f"Error Getting {e}", duration=5)

# Build a Blocks-based interface with a custom HTML header.
with gr.Blocks() as demo:
    gr.HTML(
    """
    <div style='display: flex; align-items: center; justify-content: center; gap: 20px'>
      <div style="background-color: var(--block-background-fill); border-radius: 8px">
          <img src="https://www.gstatic.com/lamda/images/gemini_favicon_f069958c85030456e93de685481c559f160ea06b.png" style="width: 100px; height: 100px;">
      </div>
      <div>
          <h1>Gen AI Image Editing</h1>
          <p>Gemini for Image Editing</p>
          <p>Powered by <a href="https://gradio.app/">Gradio</a> ⚡️</p>
          <p>Duplicate Repo <a href="https://huggingface.co/spaces/ameerazam08/Gemini-Image-Edit?duplicate=true">Duplicate</a></p>
          <p>Get an API Key <a href="https://aistudio.google.com/apikey">here</a></p>
          <p>Follow me on Twitter: <a href="https://x.com/Ameerazam18">Ameerazam18</a></p>
      </div>
    </div>
    """
    )

    gr.Markdown("""
## ⚠️ API Configuration ⚠️
- **Issue:** ❗ Sometimes the model returns text instead of an image, causing failures when saving as an image.  
### 🔧 Steps to Address:
1. **🛠️ Duplicate the Repository**  
   - Create a separate copy for modifications.  
2. **🔑 Use Your Own Gemini API Key**  
   - You **must** configure your own Gemini key for generation!  
---
### 📌 Usage  
- Upload an image and enter a prompt to generate outputs.
- If text is returned instead of an image, it will appear in the text output.
- Upload Only PNG Image
- ❌ **Do not use NSFW images!**
""")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                image_mode="RGBA"
            )
            gemini_api_key = gr.Textbox(
                lines=1,
                placeholder="Enter Gemini API Key (optional)",
                label="Gemini API Key (optional)"
            )
            prompt_input = gr.Textbox(
                lines=2,
                placeholder="Enter prompt here...",
                label="Prompt"
            )
            submit_btn = gr.Button("Generate")
        with gr.Column():
            output_gallery = gr.Gallery(label="Generated Outputs")
            output_text = gr.Textbox(label="Gemini Output", placeholder="Text response will appear here if no image is generated.")

    # Set up the interaction with two outputs.
    submit_btn.click(
        fn=process_image_and_prompt,
        inputs=[image_input, prompt_input, gemini_api_key],
        outputs=[output_gallery, output_text],
    )
    
    examples = [
        ["data/1.webp", 'change text to "AMEER"', ""],
        ["data/2.webp", "remove the spoon from hand only", ""],
        ["data/3.webp", 'change text to "Make it "', ""],
        ["data/1.jpg", "add joker style only on face", ""],
        ["data/1777043.jpg", "add joker style only on face", ""],
        ["data/2807615.jpg", "add lipstick on lip only", ""],
        ["data/76860.jpg", "add lipstick on lip only", ""],
        ["data/2807615.jpg", "make it happy looking face only", ""],
    ]
    
    gr.Examples(
        examples=examples,
        inputs=[image_input, prompt_input, gemini_api_key],
        label="Try these examples"
    )

demo.queue(max_size=500).launch()
