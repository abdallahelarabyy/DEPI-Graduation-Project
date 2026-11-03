# Text-to-Image Generator with Stable Diffusion

[![tests](https://github.com/abdallahelarabyy/DEPI-Graduation-Project/actions/workflows/tests.yml/badge.svg)](https://github.com/abdallahelarabyy/DEPI-Graduation-Project/actions/workflows/tests.yml)

A FastAPI application that generates images from text descriptions using Stable Diffusion models.

> Graduation project for the **Digital Egypt Pioneers Initiative (DEPI)**.

## Features

- Text to image generation using Stable Diffusion
- Web interface for easy interaction
- API endpoint for programmatic access
- Customizable generation parameters (steps, guidance scale, dimensions, negative prompt)
- Reproducible results: pass a `seed`, or reuse the seed returned with every image
- Input validation with clear error messages (for example, dimensions must be multiples of 8)
- Non-blocking server: generation runs in a worker thread, one image at a time
- Health endpoint that reports the model, the device (CUDA / Apple MPS / CPU) and any load error
- Docker and docker-compose setup with GPU support
- Automated API tests with a mocked pipeline (GitHub Actions)
- Returns images as files or base64-encoded strings

## Requirements

- Python 3.8+
- CUDA-compatible GPU recommended (but will work on CPU)
- 8+ GB RAM (16+ GB recommended for better performance)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/abdallahelarabyy/DEPI-Graduation-Project.git
   cd DEPI-Graduation-Project
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
    - `width` / `height` (optional, default=512): image size, a multiple of 8 between 256 and 1024
    - `seed` (optional): random seed. The same seed and parameters give the same image
    - `return_format` (optional, default="file"): "file" or "base64"
  - The seed used is returned in the JSON body (`seed`) or in the `X-Seed` response header
  - Invalid parameters return `422` with a list of problems. If the model is not loaded, the endpoint returns `503`
- `GET /health`: model status, model ID, device and load error (if any)

### Example API Usage

Using curl:

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "prompt=a beautiful landscape with mountains and a lake" \
  -F "num_inference_steps=50" \
  -F "guidance_scale=7.5" \
  -F "width=512" \
  -F "height=512" \
  -F "seed=42" \
  -F "return_format=base64" \
  -o image.json
```

Using the included Python client:

```bash
python client.py "a beautiful landscape with mountains and a lake" --steps 30 --seed 42
```

### Running with Docker

```bash
docker compose up --build
```

The compose file reserves an NVIDIA GPU. Remove the `deploy` section to run on CPU.

### Running the Tests

The tests replace Stable Diffusion with a fake pipeline, so they run in about a second without a GPU or a model download:

```bash
pip install pytest httpx
pytest
```

## Customization

### Changing the Model

The app uses [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) by default. To use a different Stable Diffusion model, set the `MODEL_ID` environment variable:

```bash
MODEL_ID=<huggingface-model-id> python main.py
```

Available models can be found on [Hugging Face](https://huggingface.co/models?pipeline_tag=text-to-image).

### Performance Optimization

For better performance:

1. Use a CUDA-compatible GPU
2. Reduce image dimensions (width and height)
3. Reduce the number of inference steps (30-40 often gives good results)

## Acknowledgements

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) by CompVis
- [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- [FastAPI](https://fastapi.tiangolo.com/) 
