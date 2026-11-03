import base64
import io
import os
import random
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

MODEL_ID = os.environ.get("MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
OUTPUT_DIR = "output"

# Generation limits (Stable Diffusion needs dimensions divisible by 8)
MIN_SIZE, MAX_SIZE = 256, 1024
MAX_STEPS = 150
MAX_GUIDANCE = 30.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

state = {"pipe": None, "device": None, "error": None}
# One generation at a time: the pipeline is not thread-safe and GPU memory is limited
generation_lock = threading.Lock()


def select_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline():
    """Load the Stable Diffusion pipeline. Returns (pipe, device)."""
    import torch
    from diffusers import StableDiffusionPipeline

    device = select_device(torch)
    print(f"Loading {MODEL_ID} on {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    if device != "cpu":
        pipe.enable_attention_slicing()
    return pipe, device


def make_generator(device: str, seed: int):
    import torch
    # MPS generators are not supported for every op; seed on CPU in that case
    return torch.Generator(device="cpu" if device == "mps" else device).manual_seed(seed)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["error"] = None
    try:
        state["pipe"], state["device"] = load_pipeline()
    except Exception as e:  # keep the server up so /health can report the problem
        state["error"] = f"{type(e).__name__}: {e}"
        print(f"Model failed to load: {state['error']}")
    yield
    state["pipe"] = None


app = FastAPI(title="Text to Image API",
              description="Generate images from text prompts using Stable Diffusion",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


def validate(prompt, num_inference_steps, guidance_scale, width, height, return_format):
    errors = []
    if not prompt.strip():
        errors.append("prompt must not be empty")
    if not 1 <= num_inference_steps <= MAX_STEPS:
        errors.append(f"num_inference_steps must be between 1 and {MAX_STEPS}")
    if not 0 <= guidance_scale <= MAX_GUIDANCE:
        errors.append(f"guidance_scale must be between 0 and {MAX_GUIDANCE}")
    for name, value in (("width", width), ("height", height)):
        if not MIN_SIZE <= value <= MAX_SIZE or value % 8:
            errors.append(f"{name} must be a multiple of 8 between {MIN_SIZE} and {MAX_SIZE}")
    if return_format.lower() not in ("file", "base64"):
        errors.append("return_format must be 'file' or 'base64'")
    if errors:
        raise HTTPException(status_code=422, detail=errors)


# A plain `def` (not `async def`): FastAPI runs it in a worker thread, so a long
# generation does not block the event loop and /health keeps responding.
@app.post("/generate")
def generate_image(
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(""),
    num_inference_steps: int = Form(50),
    guidance_scale: float = Form(7.5),
    width: int = Form(512),
    height: int = Form(512),
    seed: Optional[int] = Form(None),
    return_format: str = Form("file"),  # 'file' or 'base64'
):
    validate(prompt, num_inference_steps, guidance_scale, width, height, return_format)

    pipe = state["pipe"]
    if pipe is None:
        raise HTTPException(status_code=503,
                            detail=f"Model is not loaded. {state['error'] or 'Still starting up.'}")

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    try:
        with generation_lock:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=make_generator(state["device"], seed),
            ).images[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

    image_filename = os.path.join(OUTPUT_DIR, f"generated_{uuid.uuid4()}.png")
    image.save(image_filename)

    if return_format.lower() == "base64":
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return JSONResponse(content={
            "status": "success",
            "prompt": prompt,
            "seed": seed,
            "base64_image": f"data:image/png;base64,{img_str}",
        })

    return FileResponse(
        path=image_filename,
        media_type="image/png",
        filename=os.path.basename(image_filename),
        headers={"X-Seed": str(seed)},
    )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if state["pipe"] is not None else "degraded",
        "model_loaded": state["pipe"] is not None,
        "model_id": MODEL_ID,
        "device": state["device"],
        "error": state["error"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
