from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
from diffusers import StableDiffusionPipeline
import uuid
import os
from PIL import Image
import io
import base64
from typing import Optional

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)
# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

app = FastAPI(title="Text to Image API", description="Generate images from text prompts using Stable Diffusion")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the model
@app.on_event("startup")
async def startup_event():
    global pipe
    model_id = "runwayml/stable-diffusion-v1-5"  # You can change this to any other Stable Diffusion model
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)
    
    # Enable memory efficient attention if using CUDA
    if device == "cuda":
        pipe.enable_attention_slicing()

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/generate")
async def generate_image(
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(""),
    num_inference_steps: Optional[int] = Form(50),
    guidance_scale: Optional[float] = Form(7.5),
    width: Optional[int] = Form(512),
    height: Optional[int] = Form(512),
    return_format: Optional[str] = Form("file")  # 'file' or 'base64'
):
    try:
        # Generate a unique filename
        image_filename = f"output/generated_{uuid.uuid4()}.png"
        
        # Generate the image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        ).images[0]
        
        # Save the image
        image.save(image_filename)
        
        if return_format.lower() == "base64":
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return JSONResponse(content={
                "status": "success",
                "prompt": prompt,
                "base64_image": f"data:image/png;base64,{img_str}"
            })
        else:
            # Return the file
            return FileResponse(
                path=image_filename,
                media_type="image/png",
                filename=os.path.basename(image_filename)
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 