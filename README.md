# Gemini Image Edit

A magical image editing application powered by Google's Gemini AI and Gradio. Transform your images with AI-powered edits using natural language prompts.

## 🚀 Features

- Upload and edit images using natural language prompts
- Powered by Google's Gemini 2.0 Flash model
- Beautiful and intuitive Gradio interface
- Real-time image processing
- Support for various image formats

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Required Python packages (see Installation)

## 💻 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/gemini-image-edit.git
cd gemini-image-edit
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## 🔑 Setup

1. Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. You can either:
   - Set the API key as an environment variable: `export GEMINI_API_KEY=your_api_key`
   - Or enter it directly in the application interface

## 🎮 Usage

1. Run the application:
```bash
python app.py
```

2. Open your web browser and navigate to the local URL shown in the terminal (typically http://127.0.0.1:7860)

3. Upload an image and enter your edit prompt in natural language

4. Click "Generate Magic" to process your image

## 🎨 Example Prompts

- "Add a magical sparkle effect"
- "Make the background more vibrant"
- "Add a cute kawaii style"
- "Remove the background"
- "Add a dreamy filter"

## ⚠️ Important Notes

- The application uses the Gemini 2.0 Flash model
- Processing time may vary based on image size and complexity
- For best results, use PNG images
- Keep your API key secure and never share it publicly

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
