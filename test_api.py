#!/usr/bin/env python3
"""
Test script for the Text-to-Image API.
This script demonstrates how to use the API programmatically.
"""

import requests
import json
import base64
import os
from PIL import Image
import io
import argparse

def generate_image(
    prompt,
    negative_prompt="",
    num_inference_steps=50,
    guidance_scale=7.5,
    width=512,
    height=512,
    return_format="base64",
    api_url="http://localhost:8000/generate"
):
    """
    Generate an image using the Text-to-Image API.
    
    Args:
        prompt (str): Text description of the image to generate
        negative_prompt (str, optional): Things to avoid in the image
        num_inference_steps (int, optional): Number of denoising steps
        guidance_scale (float, optional): How closely to follow the prompt
        width (int, optional): Image width
        height (int, optional): Image height
        return_format (str, optional): "file" or "base64"
        api_url (str, optional): URL of the API endpoint
        
    Returns:
        PIL.Image.Image: The generated image
    """
    # Prepare the form data
    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "width": width,
        "height": height,
        "return_format": return_format
    }
    
    print(f"Generating image with prompt: '{prompt}'")
    print(f"Using parameters: {json.dumps({k: v for k, v in data.items() if k != 'prompt'})}")
    
    # Send the request
    response = requests.post(api_url, data=data)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    # Process the response
    if return_format == "base64":
        # Parse the JSON response
        response_data = response.json()
        # Extract the base64 image data
        base64_image = response_data["base64_image"].split(",")[1]
        # Convert base64 to image
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
    else:
        # Convert the binary response to an image
        image = Image.open(io.BytesIO(response.content))
    
    return image

def main():
    parser = argparse.ArgumentParser(description="Test the Text-to-Image API")
    parser.add_argument("prompt", help="Text description of the image to generate")
    parser.add_argument("--negative", help="Negative prompt (things to avoid)", default="")
    parser.add_argument("--steps", type=int, help="Number of inference steps", default=50)
    parser.add_argument("--guidance", type=float, help="Guidance scale", default=7.5)
    parser.add_argument("--width", type=int, help="Image width", default=512)
    parser.add_argument("--height", type=int, help="Image height", default=512)
    parser.add_argument("--format", choices=["base64", "file"], help="Return format", default="base64")
    parser.add_argument("--url", help="API URL", default="http://localhost:8000/generate")
    parser.add_argument("--output", help="Output file path", default="generated_image.png")
    
    args = parser.parse_args()
    
    image = generate_image(
        prompt=args.prompt,
        negative_prompt=args.negative,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        return_format=args.format,
        api_url=args.url
    )
    
    if image:
        # Save the image
        image.save(args.output)
        print(f"Image saved to {args.output}")
        
        # Display the image if running in an environment with a display
        try:
            image.show()
        except:
            print("Could not display the image. It has been saved to disk.")

if __name__ == "__main__":
    main() 