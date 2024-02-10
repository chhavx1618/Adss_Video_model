import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np

app = FastAPI()

def detect_objects(image_path, weights_path):
    # Run YOLOv7 detection on the input image
    command = f"python detect.py --weights {weights_path} --source {image_path}"

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Object detection failed")

    # Read the original image
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise HTTPException(status_code=500, detail=f"Failed to read the original image at path: {image_path}")

    # Find the latest output image in runs/detect
    output_folder = Path("runs/detect")
    exp_folders = [f for f in output_folder.iterdir() if f.is_dir()]
    if not exp_folders:
        raise HTTPException(status_code=500, detail="No experiment folders found in runs/detect")

    latest_exp_folder = max(exp_folders, key=os.path.getmtime)
    output_files = list(latest_exp_folder.glob("*.jpg"))
    if not output_files:
        raise HTTPException(status_code=500, detail="Output image not found")
    
    output_path = output_files[-1]  # Get the latest output image
    output_image = cv2.imread(str(output_path))
    if output_image is None:
        raise HTTPException(status_code=500, detail=f"Failed to read the output image at path: {output_path}")

    # Combine the original image and the output image with bounding boxes
    result_image = np.concatenate((original_image, output_image), axis=1)

    # Save the result image
    result_path = "result.jpg"
    cv2.imwrite(result_path, result_image)

    # Remove the images from the runs/detect directory
    for file in output_files:
        os.remove(file)

    return result_path

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    # Save the uploaded image file
    upload_folder = Path("uploads")
    upload_folder.mkdir(parents=True, exist_ok=True)  # Ensure the 'uploads' directory exists
    image_path = upload_folder / file.filename
    with image_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Specify the path to the YOLOv7 weights file
    weights_path = "runs/train/exp4/weights/epoch_829.pt"

    # Perform object detection and get the result image path
    result_path = detect_objects(image_path, weights_path)

    # Return the result image as a streaming response
    return StreamingResponse(open(result_path, "rb"), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
