import cv2
import pandas as pd
import os

# Replace 'input.mp4' with your video file and 'output_folder' with the directory where you want to save frames.
input_video = '1_720p.mkv'
output_folder = 'output_folder'
output_csv = 'frame_details.csv'  # Specify the CSV file for storing frame details
input_video_1="2_720p.mkv"
output_folder_1 = 'output_folder_1'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_folder_1, exist_ok=True)

# Create a DataFrame to store frame details
frame_details = pd.DataFrame(columns=['half','Timestamp', 'Frame_ID'])

# Open the video file
cap = cv2.VideoCapture(input_video)

frame_id = 0
timestamp = 0  # Initialize the timestamp

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save the frame to the output folder
    frame_filename = os.path.join(output_folder, f'frame_{frame_id:04d}.png')
    cv2.imwrite(frame_filename, frame)

    frame_details = frame_details._append({'half':1,'Timestamp': timestamp/1000, 'Frame_ID': frame_id}, ignore_index=True)

    print(timestamp,frame_id)
    # Increment timestamp by 1 second (1000 milliseconds)
    timestamp += 1000

    # Move to the next second in the video
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp)

    frame_id += 1

cap.release()
frame_details=frame_details._append({'half':"",'Timestamp': "", 'Frame_ID': ""}, ignore_index=True)
# Open the video file
cap = cv2.VideoCapture(input_video_1)

frame_id = 0
timestamp = 0  # Initialize the timestamp

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save the frame to the output folder
    frame_filename = os.path.join(output_folder_1, f'frame_{frame_id:04d}.png')
    cv2.imwrite(frame_filename, frame)

    frame_details = frame_details._append({'half':2,'Timestamp': timestamp/1000, 'Frame_ID': frame_id}, ignore_index=True)

    print(timestamp,frame_id)
    # Increment timestamp by 1 second (1000 milliseconds)
    timestamp += 1000

    # Move to the next second in the video
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp)

    frame_id += 1

cap.release()

# Save frame details to a CSV file
frame_details.to_csv(output_csv, index=False)
