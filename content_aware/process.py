import cv2
import math
from collections import deque
from moviepy.editor import VideoFileClip, AudioFileClip
from detect1 import detect as detect1

detect = detect1()

def process_video(input_video_path, output_video_path, arr):
    # Open input video
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object to write output video
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (720, 1280))

    # Process each frame
    total =0
    i=0
    num = 0
    prevx = frame_width//2
    prevy = frame_height//2
    qx = deque(maxlen=10)
    qy = deque(maxlen=10)
    while cap.isOpened():
        # if(num>150):
        #     break
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects and get bounding boxes
        boxes = detect.detect_objects(frame)
        j=0
        new_height = 1280
        new_width = 720
        if boxes is not None:# Crop and resize frame based on bounding box
            max_area = 0
            max_area_box = None
            for box in boxes:
                x, y, w, h = box
                area = w * h
                if area > max_area:
                    max_area = area
                    max_area_box = box
            
            
            x1 = x + w // 2
            y1 = y + h // 2
            if(len(qy)!=0):
                curry = sum(qy)//len(qy)
                currx = sum(qx)//len(qx)
            else:
                curry = prevy
                currx = prevx
            
            disx = abs(x1 - currx)
            disx *= disx
            disy = abs(y1 - curry)
            disy *= disy
            
            # prevx = x1
            # prevy = y1
            
            dis = math.sqrt(disx+disy)
            arr.append(dis)
                    
            if max_area_box is not None:
                if(dis<17):
                    x3,y3,w3,h3 = max_area_box
                    center_x = prevx
                    center_y = prevy
                    x3 += h3 // 2
                    y3 += w3 // 2
                    qy.append(y3)
                    qx.append(x3)
                    # prevx = (prevx + x3)/2 
                    # prevy = (prevy + y3)/2
                else:
                    # If distance greater than threshold(15) came with hit and trail then empty the queue
                    x, y, w, h = max_area_box
                    center_x = x + w // 2
                    center_y = y + h // 2
                    prevx = x
                    prevy = y
                    qy.clear()
                    qx.clear()

                # Crop and resize frame based on bounding box
                cropped_frame = frame[max(0, center_y - new_height // 2):min(frame_height, center_y + new_height // 2),
                                      max(0, center_x - new_width // 2):min(frame_width, center_x + new_width // 2)]
                
                # Check if cropped frame is empty
                if cropped_frame.size == 0:
                    print("Cropped frame is empty for box:", box)
                    continue
                
                resized_frame = cv2.resize(cropped_frame, (720,1280))
            else:
                cropped_frame = frame[max(0, prevy - new_height // 2):min(frame_height, prevy + new_height // 2),
                                      max(0, prevx - new_width // 2):min(frame_width, prevx + new_width // 2)]
                resized_frame = cv2.resize(cropped_frame, (720, 1280))
                    
                # Write resized frame to output video
            j+=1
            total+=1
            out.write(resized_frame)
            print(f'{i} {j}')
            num += 1
            i+=1

    # Release video capture and writer objects
    print(total)
    cap.release()
    out.release()
    
    video_with_audio = VideoFileClip(input_video_path)

    # Load the video without audio (v2)
    video_without_audio = VideoFileClip("output.mp4")

    # Extract the audio from the first video
    audio = video_with_audio.audio

    # Set the extracted audio to the second video
    final_video = video_without_audio.set_audio(audio)

    # Export the final video
    final_video.write_videofile("music.mp4", codec="libx264", audio_codec="aac")