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
    if not api_key and not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("Please provide a Gemini API key")
    
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
        if not composite_pil:
            return None, "Please upload an image first!"
            
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
        return None, f"Error: {str(e)}"

# Custom CSS for loading animation
loading_css = """
.animate-pulse {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: .5;
    }
}

.loading-spinner {
    width: 40px;
    height: 40px;
    border: 4px solid rgba(255, 105, 180, 0.1);
    border-radius: 50%;
    border-top-color: #FF69B4;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}

/* Kawaii stars animation */
@keyframes twinkle {
    0%, 100% { opacity: 0.2; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.2); }
}

.star {
    position: absolute;
    background: #FFD700;
    clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%);
    animation: twinkle 2s infinite;
}
"""

# Build a Blocks-based interface with a custom HTML header and CSS
with gr.Blocks(css=loading_css, theme=gr.themes.Soft(primary_hue="pink")) as demo:
    # Custom HTML header with proper class for styling
    gr.HTML(
    """
    <div class="header-container" style="display: flex; align-items: center; padding: 0 0 15px 10px; margin-top: -5px;">
      <div style="margin-right: 15px;">
          <img src="https://s3-eu-west-1.amazonaws.com/tpd/logos/613b1914c1f260001e077ffd/0x0.png" alt="VYBE logo" style="max-width: 90px; display: block; margin: 0;">
      </div>
      <div>
          <h1 style="margin: 0; padding: 0; font-size: 24px;"> Instant Fashion </h1>
          <p style="margin: 2px 0 0 0; font-size: 14px;">Transform your images with magical AI powers! </p>
      </div>
    </div>
    """
    )
    
    with gr.Accordion(" Configuration ", open=False, elem_classes="config-accordion"):
        gr.Markdown("""
        ###  Settings
        - **Model**: Gemini 2.0 Flash
        - **Temperature**: 1.0
        - **Max Tokens**: 8192
        
        ###  Important Notes
        - Use your own Gemini API key for best results
        - Upload PNG images for optimal performance
        - Processing time may vary based on image size
        """)

    with gr.Row(elem_classes="main-content"):
        with gr.Column(elem_classes="input-column"):
            image_input = gr.Image(
                type="pil",
                label=" Upload Your Image",
                image_mode="RGBA",
                elem_id="image-input",
                elem_classes="upload-box"
            )
            gemini_api_key = gr.Textbox(
                lines=1,
                placeholder="🔑 Enter your Gemini API Key here...",
                label="API Key",
                elem_classes="api-key-input"
            )
            prompt_input = gr.Textbox(
                lines=2,
                placeholder="✨ Describe the magical changes you want to make...",
                label="Edit Prompt",
                elem_classes="prompt-input"
            )
            submit_btn = gr.Button(" Generate Image ", elem_classes="generate-btn")
        
        with gr.Column(elem_classes="output-column"):
            output_gallery = gr.Gallery(
                label=" Generated Output", 
                elem_classes="output-gallery",
                show_label=True
            )
            output_text = gr.Textbox(
                label=" Status", 
                placeholder="✨ Processing status will appear here...",
                elem_classes="output-text"
            )

    # Set up the interaction with loading animation
    submit_btn.click(
        fn=process_image_and_prompt,
        inputs=[image_input, prompt_input, gemini_api_key],
        outputs=[output_gallery, output_text],
    )
    
    #gr.Markdown("## 💖 Example Prompts", elem_classes="gr-examples-header")
    
    # Remove examples section since it's causing issues with caching
    # examples = [
    #     ["data/1.webp", 'change text to "AMEER"', ""],
    #     ["data/2.webp", "remove the spoon from hand only", ""],
    #     ["data/3.webp", 'change text to "Make it "', ""],
    #     ["data/1.jpg", "add joker style only on face", ""],
    #     ["data/1777043.jpg", "add joker style only on face", ""],
    #     ["data/2807615.jpg", "add lipstick on lip only", ""],
    #     ["data/76860.jpg", "add lipstick on lip only", ""],
    #     ["data/2807615.jpg", "make it happy looking face only", ""],
    # ]
    
    # gr.Examples(
    #     examples=examples,
    #     inputs=[image_input, prompt_input],
    #     outputs=[output_gallery, output_text],
    #     fn=process_image_and_prompt,
    #     cache_examples=True,
    #     elem_id="examples-grid"
    # )

demo.queue(max_size=50).launch()