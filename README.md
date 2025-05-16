# Text-to-Image Generator with Stable Diffusion

A FastAPI application that generates images from text descriptions using Stable Diffusion models.

## Features

- Text to image generation using Stable Diffusion
- Web interface for easy interaction
- API endpoint for programmatic access
- Customizable generation parameters (steps, guidance scale, dimensions)
- Returns images as files or base64-encoded strings

## Requirements

- Python 3.8+
- CUDA-compatible GPU recommended (but will work on CPU)
- 8+ GB RAM (16+ GB recommended for better performance)

## Installation

1. Clone this repository:
   ```
   git clone <your-repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

Start the application with:

```
python main.py
```

The server will start on `http://localhost:8000`. You can access the web interface by opening this URL in your browser.

### API Endpoints

- `GET /`: Web interface for the application
- `POST /generate`: Generate an image from text
  - Parameters (form data):
    - `prompt` (required): Text description of the image to generate
    - `negative_prompt` (optional): Things to avoid in the image
    - `num_inference_steps` (optional, default=50): Number of denoising steps
    - `guidance_scale` (optional, default=7.5): How closely to follow the prompt
    - `width` (optional, default=512): Image width
    - `height` (optional, default=512): Image height
    - `return_format` (optional, default="file"): "file" or "base64"
- `GET /health`: Health check endpoint

### Example API Usage

Using curl:

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "prompt=a beautiful landscape with mountains and a lake" \
  -F "num_inference_steps=50" \
  -F "guidance_scale=7.5" \
  -F "width=512" \
  -F "height=512" \
  -F "return_format=base64" \
  -o image.json
```

## Customization

### Changing the Model

You can change the Stable Diffusion model by modifying the `model_id` variable in the `startup_event` function in `main.py`. Available models can be found on [Hugging Face](https://huggingface.co/models?pipeline_tag=text-to-image).

### Performance Optimization

For better performance:

1. Use a CUDA-compatible GPU
2. Reduce image dimensions (width and height)
3. Reduce the number of inference steps (30-40 often gives good results)

## License

[MIT License](LICENSE)

## Acknowledgements

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) by CompVis
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- [FastAPI](https://fastapi.tiangolo.com/) 