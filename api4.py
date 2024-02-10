import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
import cloudinary.uploader
import subprocess
from PIL import Image

# Cloudinary configuration
cloudinary_config = {
    "cloud_name": "dllhhra4b",  # Replace with your Cloudinary cloud name
    "api_key": "752488391172751",  # Replace with your Cloudinary API key
    "api_secret": "bEJ5HZtlritkzln4TwMcpwpMwPQ",  # Replace with your Cloudinary API secret
}

app = FastAPI()


def convert_to_jpg(image_path):
    image = Image.open(image_path)
    jpg_image_path = image_path.with_suffix(".jpg")
    image.convert("RGB").save(jpg_image_path)
    return jpg_image_path


def detect_objects(image_path, weights_path):
    # Run YOLOv7 detection on the input image
    command = f"python detect.py --weights {weights_path} --source {image_path}"

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Object detection failed: {e}")

    # Find the latest output image in runs/detect
    output_folder = Path("runs/detect")
    exp_folders = [f for f in output_folder.iterdir() if f.is_dir()]
    if not exp_folders:
        raise HTTPException(
            status_code=500, detail="No experiment folders found in runs/detect"
        )

    latest_exp_folder = max(exp_folders, key=os.path.getmtime)
    output_files = list(latest_exp_folder.glob("*.jpg"))
    if not output_files:
        raise HTTPException(status_code=500, detail="Output image not found")

    output_path = output_files[-1]  # Get the latest output image
    return output_path


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded image file locally
        upload_folder = Path("uploads")
        upload_folder.mkdir(parents=True, exist_ok=True)
        image_path = upload_folder / file.filename
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Specify the path to the YOLOv7 weights file
        weights_path = "runs/train/exp4/weights/epoch_829.pt"

        # Perform object detection using YOLOv7
        output_image_path = detect_objects(image_path, weights_path)

        # Upload the result image to Cloudinary
        upload_response = cloudinary.uploader.upload(
            str(output_image_path),
            public_id=f"detection_result_{file.filename}",
            folder="my_images",  # Create a folder if needed
            resource_type="image",  # Specify the resource type as image
            **cloudinary_config,
        )

        # Delete the local uploaded image
        os.remove(image_path)

        # Return the image URL and other useful information
        return JSONResponse(
            {
                "message": "Object detection successful!",
                "image_url": upload_response["url"],
                "public_id": upload_response["public_id"],
                "secure_url": upload_response["secure_url"],
            }
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
