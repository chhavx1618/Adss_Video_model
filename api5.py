import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse  # Add JSONResponse import
import shutil
import subprocess
from pathlib import Path
import shutil
import cloudinary.uploader
import cv2
import numpy as np
# Cloudinary configuration
cloudinary_config = {
    "cloud_name": "dllhhra4b",  # Replace with your Cloudinary cloud name
    "api_key": "752488391172751",  # Replace with your Cloudinary API key
    "api_secret": "bEJ5HZtlritkzln4TwMcpwpMwPQ",  # Replace with your Cloudinary API secret
}

app = FastAPI()




def detect_objects(video_path, weights_path):
    # Run YOLOv7 detection on the input video
    command = f"python detect.py --weights {weights_path} --source {video_path} --save-txt"

    try:
        # Execute the detection command
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Object detection failed: {e}")

    # Read the original video
    original_video = cv2.VideoCapture(str(video_path))
    if original_video is None:
        raise HTTPException(status_code=500, detail=f"Failed to read the original video at path: {video_path}")

    # Find the latest experiment folder in runs/detect
    output_folder = Path("runs/detect")
    exp_folders = [f for f in output_folder.iterdir() if f.is_dir()]
    if not exp_folders:
        raise HTTPException(status_code=500, detail="No experiment folders found in runs/detect")
    
    latest_exp_folder = max(exp_folders, key=os.path.getmtime)

    # Get the output video file from the latest experiment folder
    output_files = list(latest_exp_folder.glob("*.mp4"))
    if not output_files:
        raise HTTPException(status_code=500, detail="Output video not found in the latest experiment folder")
    
    output_path = output_files[-1]  # Get the output video file path
    output_video = cv2.VideoCapture(str(output_path))
    if output_video is None:
        raise HTTPException(status_code=502, detail=f"Failed to read the output video at path: {output_path}")

    # Get video properties
    frame_width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(original_video.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    result_path = "result.mp4"
    out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width * 2, frame_height))

    # Combine the original and output videos into a single video
    while True:
        ret_orig, frame_orig = original_video.read()
        ret_out, frame_out = output_video.read()
        if not ret_orig or not ret_out:
            break  # Break the loop if either video reaches the end
        result_frame = np.concatenate((frame_orig, frame_out), axis=1)
        out.write(result_frame)

    # Release the video capture objects
    original_video.release()
    output_video.release()
    out.release()

    return result_path
# @app.post("/detect")
# async def detect_video(file: UploadFile = File(...)):
#     try:
#         # Save the uploaded video file locally (optional)
#         upload_folder = Path("uploads")
#         upload_folder.mkdir(parents=True, exist_ok=True)
#         video_path = upload_folder / file.filename
#         with video_path.open("wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         # Specify the path to the YOLOv7 weights file
#         weights_path = "runs/train/exp4/weights/epoch_829.pt"  # Replace with your actual path

#         # Perform object detection using your existing logic to get the result video path
#         result_path = detect_objects(video_path, weights_path)

#         # Upload the result video to Cloudinary
#         upload_response = cloudinary.uploader.upload(
#             result_path,
#             public_id=f"detection_result_{file.filename}",
#             folder="my_videos",  # Create a folder if needed
#             resource_type="video",  # Specify the resource type as video
#             **cloudinary_config
#         )

#         # Delete the local result video (optional)
#         os.remove(result_path)

#         # Return the video URL and other useful information
#         return JSONResponse({
#             "message": "Object detection successful!",
#             "video_url": upload_response["url"],
#             "public_id": upload_response["public_id"],
#             "secure_url": upload_response["secure_url"],
#         })

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {e}")



@app.post("/detect")
async def detect_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded video file locally (optional)
        upload_folder = Path("uploads")
        upload_folder.mkdir(parents=True, exist_ok=True)
        video_path = upload_folder / file.filename
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Specify the path to the YOLOv7 weights file
        weights_path = "runs/train/exp4/weights/epoch_829.pt"  # Replace with your actual path

        # Perform object detection using your existing logic to get the result video path
        result_path = detect_objects(video_path, weights_path)

        # Upload the result video to Cloudinary
        upload_response = cloudinary.uploader.upload(
            result_path,
            public_id=f"detection_result_{Path(file.filename).stem}",  # Remove extension from filename
            folder="my_videos",  # Create a folder if needed
            resource_type="video",  # Specify the resource type as video
            **cloudinary_config
        )

        # Delete the local result video (optional)
        os.remove(result_path)

        # Return the video URL and other useful information
        return JSONResponse({
            "message": "Object detection successful!",
            "video_url": upload_response["url"],
            "public_id": upload_response["public_id"],
            "secure_url": upload_response["secure_url"],
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
