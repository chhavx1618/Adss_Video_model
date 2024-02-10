import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
import cloudinary.uploader
import subprocess
from PIL import Image
from flask import Flask, request, jsonify
from model_predict import predict_class

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

def convert_to_webm(input_path, output_path):
    try:
        # Execute FFmpeg command to convert video to WebM format
        command = f"ffmpeg -i {input_path} -c:v libvpx -crf 10 -b:v 1M -c:a libvorbis {output_path}"
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Video conversion failed: {e}")



def detect_objectsv(video_path, weights_path):
    # Run YOLOv7 detection on the input video
    command = f"python detect.py --weights {weights_path} --source {video_path}"

    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="Object detection failed")

    # Find the latest output video in runs/detect
    output_folder = Path("runs/detect")
    exp_folders = [f for f in output_folder.iterdir() if f.is_dir()]
    if not exp_folders:
        raise HTTPException(status_code=500, detail="No experiment folders found in runs/detect")

    latest_exp_folder = max(exp_folders, key=os.path.getmtime)
    output_files = list(latest_exp_folder.glob("*.mp4"))
    if not output_files:
        raise HTTPException(status_code=500, detail="Output video not found")

    output_path = output_files[-1]  # Get the latest output video
    return output_path

def convert_to_webm(input_path, output_path):
    try:
        # Execute FFmpeg command to convert video to WebM format
        command = f"ffmpeg -i {input_path} -c:v libvpx -crf 10 -b:v 1M -c:a libvorbis {output_path}"
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Video conversion failed: {e}")
    
    
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


@app.post("/detecti")
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


@app.post("/detectv")
async def detect_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded video file locally
        upload_folder = Path("uploads")
        upload_folder.mkdir(parents=True, exist_ok=True)
        video_path = upload_folder / file.filename
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Specify the path to the YOLOv7 weights file
        weights_path = "runs/train/exp4/weights/epoch_829.pt"  # Replace with your actual path

        # Perform object detection using your existing logic to get the result video path
        result_path = detect_objects(video_path, weights_path)

        # Convert the output video to WebM format
        output_webm_path = upload_folder / f"{Path(file.filename).stem}.webm"
        convert_to_webm(result_path, output_webm_path)

        # Upload the converted video to Cloudinary
        upload_response = cloudinary.uploader.upload(
            str(output_webm_path),
            public_id=f"detection_result_{Path(file.filename).stem}",  # Remove extension from filename
            folder="my_videos",  # Create a folder if needed
            resource_type="video",  # Specify the resource type as video
            **cloudinary_config
        )

        # Delete the local files (optional)
        os.remove(video_path)
        os.remove(result_path)
        os.remove(output_webm_path)

        # Return the video URL and other useful information
        return JSONResponse({
            "message": "Video conversion and upload successful!",
            "video_url": upload_response["url"],
            "public_id": upload_response["public_id"],
            "secure_url": upload_response["secure_url"],
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")



@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    prediction = predict_class(text)
    return jsonify({'class': prediction})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)