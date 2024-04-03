# -*- coding: utf-8 -*-
"""
Created on Sun Mar 07 19:48:35 2024

@author: MaxGr
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from datetime import datetime

current_datetime = datetime.now()
date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")


import torch
print('torch.version: ',torch. __version__)
print('torch.version.cuda: ',torch.version.cuda)
print('torch.cuda.is_available: ',torch.cuda.is_available())
print('torch.cuda.device_count: ',torch.cuda.device_count())
print('torch.cuda.current_device: ',torch.cuda.current_device())
device_default = torch.cuda.current_device()
torch.cuda.device(device_default)
print('torch.cuda.get_device_name: ',torch.cuda.get_device_name(device_default))
device = torch.device("cuda")




import yaml

# File path
file_path = "OPENAI_API_KEY.yaml"

# Write the API key to the YAML file
# with open(file_path, "w") as f:
#     yaml.dump({"OPENAI_API_KEY": OPENAI_API_KEY}, f)
    
# Read the API key
with open(file_path, "r") as f:
    data = yaml.safe_load(f)
    OPENAI_API_KEY = data["OPENAI_API_KEY"]
    
    

import os
import time
import threading
from func_timeout import func_set_timeout

import openai
from openai import OpenAI

client = OpenAI(
  api_key=OPENAI_API_KEY,
)

system_sensitivity = 'normal'

prompt_background = "{You are an voice assistant for blind person, \
the input is the actual data collected by a phone camera, the phone is always facing front, \
please provide the key information for the blind user to help him navigate and avoid potential danger. \
Please note that the xloc and yloc represent the object location (proportional to the image), \
object height and width are also a proportion.}"
    
prompt_location = "{The location information (center_x, center_y, height, width) of objects is the proportion to the image, \
the detected objects are categorized into 4 type based on the image region.\
Left and Right: objects located on left 25% or right 25% of the image, these objects are usually moving and has large proportion.\
Front: objects that are still far away, can be used to discriminate the current situation.\
Ground: objects that are nearby, need to be cautioned.}"

prompt_motion = "{Analyze the movement (speed and direction) \
and location (xloc and yloc) of each object to determine its trajectory relative to the user.\
Use this information to assess whether an object is moving towards the user and if so, \
how quickly a potential collision might occur based on the object's speed and direction of movement.}"
    
prompt_sensitivity = '{System sensitivity: Incorporate the sensitivity setting in your response. \
For a low sensitivity setting, identify and report only imminent and direct threats to safety. \
For normal sensitivity, include potential hazards that could pose a risk if not avoided. \
For high sensitivity, report all detected objects that could potentially cause any level of inconvenience or danger.\
More focus on pedestrians and less focus on cars, as users are mostly walking on the sidewalk. \
Please more focus on the left,right,and ground area, as they are usually very close,\
but when you evaluate the emergency, consider the size and type of objects.\
Current sensitivity: ' + str(system_sensitivity) + '}'
   
    
    
instruction = prompt_background + prompt_location + prompt_sensitivity


@func_set_timeout(10)
def GPT_response(model, prompt):
    # model="gpt-3.5-turbo",
    # model="gpt-4-turbo",
    # model="gpt-4-0125-preview",
    completion = client.chat.completions.create(
        model=model,
      messages=[
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
      ]
    )
    return completion




prompt_format_full = 'Please organize your output into this format: \
{ "scene": …, quickly describe the current situation for blind user; \
  "key_objects" …, quickly and roughly locate the key objects for blind user; \
  "danger_checker": …, quickly diagnose if there is potential danger for a blind person; \
  "danger_label": …, output 1 if there is an emergency, output 0 if not; \
  "danger_index": [object_id, danger_index], estimate a score from 0 to 100 about each objects that may cause danger; \
  "voice_guide": …, the main output to instant alert the blind person for emergency.}'
    

prompt_format_turbo = 'Please organize your output into this format: \
{ "danger_score": output 1 for immediate threat, output 0 if not; \
  "reason": the main output to instant alert the blind person for emergency.}'

    
prompt_format_benchmark = 'Please organize your output into this format: \
{ "danger_score": output 1 for immediate threat, output 0 if not; \
  "reason": explain your annotation reason within 10 words.}'
    
# { "danger_score": predict a score from 0 to 1 to evaluate the emergency level, non-emergency shoule below 0.5; \


prompt_word_limiter = 'Limit your answer into 20 words'



GPT_list = []
def GPT_annotation(frame_info_i):
    global GPT_list
    # frame_info_i = categorized_detections
    object_info = str(frame_info_i)
    prompt = object_info + prompt_format_benchmark

    GPT_start_time = time.time()
    try:
        completion = GPT_response("gpt-3.5-turbo-0125", prompt)
        response = completion.choices[0].message.content
        usage = completion.usage
    except:
        print('GPT time running out...')
        return
    GPT_end_time = time.time()
    GPT_time_cost = round(GPT_end_time-GPT_start_time, 4)
    
    # completion_tokens = usage.completion_tokens
    # prompt_tokens = usage.prompt_tokens
    # total_tokens = usage.total_tokens
    
    GPT_list.append([response, GPT_time_cost, usage])
    print(response)
    # print(usage)
    # print(time_cost)
        
    # return [response, [completion_tokens, prompt_tokens, total_tokens], GPT_time_cost]

# save_file = np.array([response_list, tokens_list, time_list,
#                       detection_info, instruction, prompt], dtype=object)

# file_name = f'{video_name}_{weight_file}_{model}_{system_sensitivity}_.npy'
# # eval(response)










'''
import models
'''

from typing import List
from dataclasses import dataclass
from onemetric.cv.utils.iou import box_iou_batch

import supervision as sv
box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()

from ultralytics.trackers.byte_tracker import BYTETracker, STrack 
from ultralytics import YOLOWorld, YOLO


weight_file = 'yolov8x-worldv2.pt'
# model = YOLOWorld('yolov8l-world.pt') # bbox
# model = YOLO('yolov8x.pt')  # bbox
# model = YOLO('yolov8x-seg.pt')  # seg
# model = YOLO('yolov8l-world.pt') # bbox
# model = YOLO('yolov8x-worldv2.pt')  # bbox, latest yolo
model = YOLO(weight_file)


'''
Define custom classes

VIN = Visually impaired navigation
'''
VIN = [
    'car', 'person', 'bus', 'bicycle', 'motorcycle', 'traffic light', 'stop sign',
    'fountain',
    'crosswalk', 'sidewalk', 'door', 'stair', 'escalator', 'elevator', 'ramp',
    'bench', 'trash can', 'pole', 'fence', 'tree', 'dog', 'cat', 'bird', 'parking meter',
    'mailbox', 'manhole', 'puddle', 'construction sign', 'construction barrier',
    'scaffolding', 'hole', 'crack', 'speed bump', 'curb', 'guardrail', 'traffic cone',
    'traffic barrel', 'pedestrian signal', 'street sign', 'fire hydrant', 'lamp post',
    'bench', 'picnic table', 'public restroom', 'fountain', 'statue', 'monument',
    'directional sign', 'information sign', 'map', 'emergency exit', 'no smoking sign',
    'wet floor sign', 'closed sign', 'open sign', 'entrance sign', 'exit sign',
    'stairs sign', 'escalator sign', 'elevator sign', 'restroom sign', 'men restroom sign',
    'women restroom sign', 'unisex restroom sign', 'baby changing station',
    'wheelchair accessible sign', 'braille sign', 'audio signal device', 'tactile paving',
    'detectable warning surface', 'guide rail', 'handrail', 'turnstile', 'gate',
    'ticket barrier', 'security checkpoint', 'metal detector', 'baggage claim',
    'lost and found', 'information desk', 'meeting point', 'waiting area', 'seating area',
    'boarding area', 'disembarking area', 'charging station', 'water dispenser',
    'vending machine', 'ATM', 'kiosk', 'public telephone', 'public Wi-Fi hotspot',
    'emergency phone', 'first aid station', 'defibrillator',
    'tree', 'pole', 'lamp post', 'staff', 'road hazard'
]

URBAN_WALKING = [
    'pedestrian', 'cyclist', 'car', 'bus', 'motorcycle', 'scooter', 'electric scooter',
    'traffic light', 'stop sign', 'crosswalk', 'sidewalk', 'curb', 'ramp', 'stair', 'escalator', 
    'elevator', 'bench', 'trash can', 'pole', 'fence', 'tree', 'fire hydrant', 'lamp post',
    'construction barrier', 'construction sign', 'scaffolding', 'hole', 'crack', 'speed bump', 
    'puddle', 'manhole', 'drain', 'grate', 'loose gravel', 'ice patch', 'snow pile', 'leaf pile',
    'standing water', 'mud', 'sand', 'street sign', 'directional sign', 'information sign',
    'parking meter', 'mailbox', 'bicycle rack', 'outdoor seating', 'planter box', 'bollard', 
    'guardrail', 'traffic cone', 'traffic barrel', 'pedestrian signal', 'crowd', 'animal', 'dog', 
    'bird', 'cat', 'public restroom', 'fountain', 'statue', 'monument', 'picnic table', 
    'outdoor advertisement', 'vendor cart', 'food truck', 'emergency exit', 'no smoking sign', 
    'wet floor sign', 'closed sign', 'open sign', 'entrance sign', 'exit sign', 'stairs sign', 
    'escalator sign', 'elevator sign', 'restroom sign', 'braille sign', 'audio signal device', 
    'tactile paving', 'detectable warning surface', 'guide rail', 'handrail', 'turnstile', 
    'gate', 'security checkpoint', 'water dispenser', 'vending machine', 'ATM', 'kiosk',
    'public telephone', 'public Wi-Fi hotspot', 'emergency phone', 'charging station',
    'first aid station', 'defibrillator', 'tree', 'pole', 'lamp post', 'staff', 'road hazard'
]


URBAN_WALKING_GENERALIZED = [
    'vehicles',  # Including cars, buses, motorcycles, bicycles, electric scooters
    'pedestrians',  # Including people walking, children playing
    'traffic signs and signals',  # Stop signs, traffic lights, pedestrian signals
    'roadway features',  # Crosswalks, sidewalks, curbs, ramps, stairs
    'surface conditions',  # Puddles, ice, snow, cracks, holes
    'street furniture',  # Benches, trash cans, lamp posts, bollards, bike racks
    'construction areas',  # Construction barriers, signs, scaffolding
    'vegetation',  # Trees, planters, grass areas that may intrude on pathways
    'animals',  # Pets like dogs and cats, as well as birds that may cause distractions
    'public amenities',  # Restrooms, water dispensers, vending machines, seating areas
    'navigation aids',  # Directional signs, informational signs, tactile paving, braille signs
    'temporary obstacles',  # Street vendors, outdoor seating, crowds, parked bicycles
    'emergency facilities',  # Fire hydrants, emergency phones, first aid stations
    'transportation hubs',  # Bus stops, metro entrances, taxi stands
    'electronic devices',  # ATMs, public Wi-Fi hotspots, charging stations
    'safety features',  # Guardrails, handrails, detectable warning surfaces
]


URBAN_WALKING_HAZARDS = [
    'person', 'cyclist', 'car', 'bus', 'motorcycle', 'scooter', 'fountain', 'bench', 
    'traffic light', 'stop sign', 'curb', 'ramp', 'stair', 'escalator','charging station',
    'elevator', 'trash can', 'pole', 'tree', 'fire hydrant', 'lamp post','ATM', 'kiosk',
    'construction barrier', 'construction sign', 'scaffolding', 'hole', 'crack', 'speed bump',
    'puddle', 'manhole', 'drain', 'grate', 'loose gravel', 'ice patch', 'snow pile', 'leaf pile',
    'standing water', 'mud', 'sand', 'street sign', 'directional sign', 'information sign',
    'parking meter', 'mailbox', 'bicycle rack', 'outdoor seating', 'planter box', 'bollard',
    'guardrail', 'traffic cone', 'traffic barrel', 'pedestrian signal', 'crowd', 'animal', 'dog',
    'bird', 'cat', 'public restroom', 'fountain', 'statue', 'monument', 'picnic table',
    'outdoor advertisement', 'vendor cart', 'food truck', 'emergency exit', 'no smoking sign',
    'wet floor sign', 'closed sign', 'open sign', 'entrance sign', 'exit sign', 'stairs sign',
    'escalator sign', 'elevator sign', 'restroom sign', 'braille sign', 'audio signal device',
    'tactile paving', 'detectable warning surface', 'guide rail', 'handrail', 'turnstile',
    'gate', 'security checkpoint', 'water dispenser', 'vending machine', 
    'public telephone', 'emergency phone', 
    'first aid station', 'defibrillator', 
    # Additional road hazards
    # 'uneven pavement', 
    'recently paved asphalt', 'oil spill', 'road debris', 'overhanging branches',
    'low-hanging signage', 'temporary road signs', 'roadworks', 'excavation sites', 'utility works',
    'fallen objects', 'spilled cargo', 'flood', 'ice', 'snowdrift', 'landslide debris',
    'erosion damage', 'parked vehicles', 'moving equipment',
    'street performers', 'demonstrations', 'large gatherings', 'parade', 'marathon', 'street fair',
    # 'crowded sidewalk', 'narrow sidewalk', 'blocked sidewalk', 
    'temporary scaffolding',
    'electrical hazards', 'wire tangle', 'unsecured manhole covers', 'improperly installed street elements',
    'visual distractions', 'audio distractions', 'smell hazards', 'toxic spill', 'biohazard materials',
    'wildlife crossings', 'stray animals', 'pets without leashes', 'flying debris', 'air pollution',
    'smoke plumes', 'dust storms', 'sandstorms', 'flash floods', 'earthquake damage', 'volcanic ash'
]

URBAN_WALKING_HAZARDS_GENERAL = [
    'vehicle',  # Generalizing cars, buses, motorcycles, scooters, etc.
    'pedestrian', 
    'cyclist',
    'traffic signal',  # Includes traffic lights and pedestrian signals
    'street sign',  # Generalizing all types of street signs
    'crosswalk',
    'sidewalk',
    'curb',
    'ramp',
    'stair',
    'escalator',
    'elevator',
    'public seating',  # Generalizing benches, picnic tables, outdoor seating
    'trash receptacle',  # Generalizing trash cans, recycling bins
    'street furniture',  # Generalizing poles, fences, lamp posts, bollards, etc.
    'tree',
    'construction site',  # Generalizing barriers, signs, scaffolding
    'road obstruction',  # Generalizing holes, cracks, speed bumps, puddles, manholes
    'loose materials',  # Generalizing loose gravel, sand, leaves
    'slick surface',  # Generalizing ice patches, wet floors
    'animal',  # Generalizing dogs, cats, birds
    'outdoor advertisement',
    'vendor',  # Generalizing vendor carts, food trucks
    'water feature',  # Generalizing fountains, water dispensers
    'monument',  # Generalizing statues, monuments
    'information point',  # Generalizing information signs, kiosks, information desks
    'access point',  # Generalizing entrances, exits, emergency exits, gates
    'safety equipment',  # Generalizing first aid stations, defibrillators, fire hydrants
    'navigation aid',  # Generalizing tactile paving, audio signal devices, braille signs
    'public amenity',  # Generalizing restrooms, ATMs, vending machines, charging stations
    'transport hub',  # Generalizing turnstiles, ticket barriers, waiting areas
    'obstacle crowd',  # Recognizing crowded areas as potential obstacles
]


MASK = ['people', 'human face', 'car license plate', 'license plate', 'plate']



CLASSES = URBAN_WALKING_HAZARDS
model.set_classes(CLASSES)



'''
utils
'''
def calculate_movements(data_previous, tracker_id_previous, data_current, tracker_id_current):
    # Initialize a dictionary to hold the movements
    movements = {}
    
    # Create mappings from tracker_id to bbox for both previous and current frames
    bbox_map_previous = {tid: bbox for tid, bbox in zip(tracker_id_previous, data_previous) if tid is not None}
    bbox_map_current = {tid: bbox for tid, bbox in zip(tracker_id_current, data_current) if tid is not None}
    
    # Iterate over the tracker IDs in the previous frame
    for tid_previous, bbox_previous in bbox_map_previous.items():
        if tid_previous in bbox_map_current:
            # Calculate the center of the bbox in the previous frame
            center_previous = ((bbox_previous[0] + bbox_previous[2]) / 2, (bbox_previous[1] + bbox_previous[3]) / 2)
            
            # Find the corresponding bbox in the current frame
            bbox_current = bbox_map_current[tid_previous]
            # Calculate the center of the bbox in the current frame
            center_current = ((bbox_current[0] + bbox_current[2]) / 2, (bbox_current[1] + bbox_current[3]) / 2)
            
            # Calculate the movement (difference in centers)
            dx = center_current[0] - center_previous[0]
            dy = center_current[1] - center_previous[1]
            
            # Store the movement
            movements[tid_previous] = (dx, dy)
    
    return movements





'''
Video setting
'''
text_color = [(0, 255, 0),(0, 0, 255)]
mark_danger = False

fps = 5
display_start_frame = 0
display_until_frame = 10000

# Open the video file
video_path = './Video/'
# video_capture = cv2.VideoCapture('PXL_20240222_174115817.mp4')  
# video_capture = cv2.VideoCapture('PXL_20240222_175043394.TS.mp4')
# video_capture = cv2.VideoCapture('citywalk_NJ.mp4')
# video_capture = cv2.VideoCapture('Scene_BikePath.mp4')  
# video_capture = cv2.VideoCapture('IMG_5701.MOV')  

# video_capture = cv2.VideoCapture('./Video/VID_20240313_132440_00_001_02_08.MOV')  
# video_name = 'NJ_1_2.mp4'
# video_name = 'NJ_1_3.mp4'
# video_name = 'NJ_1_5.mp4'

video_name = 'JP_1.MOV'
# video_name = 'NJ_3_1.mp4'
# video_name = 'SC_8.MOV'
# video_name = 'SC_7.MOV'




# Check if the video file opened successfully
video_capture = cv2.VideoCapture(video_path + video_name)  
if not video_capture.isOpened():
    print("Error opening video file.")
    exit()  # Stop execution if there's an error


for i in range(1):
    ret, frame = video_capture.read()
    
# Get frame dimensions
img_height, img_width = frame.shape[:2]

# Calculate 'H' segmentation lines
left_line_x = img_width // 4
right_line_x = img_width * 3 // 4
top_line_y = img_height // 2
bottom_line_y = img_height // 2
   

output_video_filename = f"./output/output_video_{date_time_string}.mp4"


fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_filename, fourcc, 10.0, (img_width, img_height))

 


'''
detection
'''
point_list = deque()
detection_info = []
bbox_list = []
key_frame = []
last_frame = 0
skipped_frame = 6
motion_factor = 10 #60 // skipped_frame


object_list = []
object_alert = []

response_list = []
tokens_list = []
time_list = []

frame_list = []
frame_id = 0
start_time = time.time()
while True:
    frame_id += 1

    print('Current frame: ', frame_id)
    ret, frame = video_capture.read()
    if not ret:  # End of the video
        break
    
    if frame_id < display_start_frame:
        continue
    if frame_id > display_until_frame:
        break
    
    cv2.putText(frame,f'{frame_id-i}',(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color[mark_danger], 2, cv2.LINE_AA)
    if frame_id % 5 in [1,2,3,4]:
    # if frame_id % 10 in [1,2,3,4,5,6,7,8,9]:
        # cv2.imshow('Video Frame', frame)
        continue
    
    '''
    yolo inference
    '''
    results = model.predict(frame)
    # results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()


    # image = results[0].orig_img
    boxes = results[0].boxes # xyxy
    xywh = results[0].boxes.xywh # xywh
    mask = results[0].masks
    h,w = frame.shape[0:2]

    predictions = boxes.data.cpu().numpy()
    
    if len(predictions) > 0:
        if len(predictions[0]) == 7:
            boxes = predictions[:,0:4]
            tracker_id = predictions[:,4].astype(int) 
            classes = predictions[:,6].astype(int)
            scores = predictions[:,5]
        else:
            boxes = predictions[:,0:4]
            tracker_id = np.zeros(len(predictions)).astype(int)
            classes = predictions[:,5].astype(int)
            scores = predictions[:,4]
    else:
        continue
    
    
    # annotate image with detections
    scene=frame.copy()
    # scene = np.zeros(frame.shape)
    detections = sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=classes,
        tracker_id = tracker_id
    )
    
    detections = detections[detections.confidence > 0.6]

    labels = [f'{tracker_id} {model.names[class_id]} {confidence:0.2f}'
            for confidence, class_id, tracker_id in zip(detections.confidence, 
                                                        detections.class_id, 
                                                        detections.tracker_id)]
    
    # annotated_frame = box_annotator.annotate(scene=scene, 
    #                                           detections=detections, 
    #                                           labels=labels)
    
    '''
    movements
    '''
    current_frame = [tracker_id, boxes, classes, scores]
    
    if frame_id > display_start_frame and frame_id % skipped_frame == 0 and skipped_frame>1:
        # tracker_id_previous = last_frame[0]
        # data_previous = last_frame[1]
        # tracker_id_current = current_frame[0]
        # data_current = current_frame[1]
        
        # movements = calculate_movements(data_previous, tracker_id_previous, data_current, tracker_id_current)
        # print(movements)
        
# ==============================================================
        '''
        video process
        '''
        frame_info = []
        categorized_detections = {'frame_id':frame_id, 'left': [], 'right': [], 'front': [], 'ground': []}

        for pid, box, label, score in zip(tracker_id, boxes, classes, scores):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(label)]  # Replace with your own label mapping
            
            if class_name not in object_list:
                object_list.append(class_name)
    
            height = y2-y1
            width = x2-x1
            center_x = x1+(width)//2
            center_y = y1+(height)//2
    
            height = int(height/h *100)
            width = int(width/w *100)
            x_loc = int(center_x/w *100)
            y_loc = int(center_y/h *100)
            
            size = int(height * width / 100)
            
            # Categorize based on the 'H' segmentation
            if x_loc < 25:
                location = 'left'
            elif x_loc > 75:
                location = 'right'
            elif y_loc < 50:
                location = 'front'
            else:
                location = 'ground'

            # if pid not in movements:
            #     dx, dy = 0, 0
            # else:
            #     dx = int((movements[pid][0] / w *100) *motion_factor )
            #     dy = int((movements[pid][1] / h *100) *motion_factor )
            

            info = f'ID:{pid}, \
class:{class_name}, \
confidence:{score:.2f}, \
center_x:{x_loc}%, \
center_y:{y_loc}%, \
object_height:{height}%, object_width:{width}%,\
size: {size}%'
            # movement: {(dx,dy)}%, \
            categorized_detections[location].append(info)
            frame_info.append(info)
            
            # offset = (center_x+dx, center_y+dy)
            # annotated_frame = cv2.arrowedLine(annotated_frame, (center_x, center_y), offset, (0,0,255),2)
            
            if location in ['ground']:
                object_alert.append([frame_id, location, pid, class_name, score, size])
            if location in ['left', 'right']:
                if size > 20:
                    object_alert.append([frame_id, location, pid, class_name, score, size])
                

        detection_info.append(categorized_detections)
        # key_frame.append(annotated_frame)
        
        gpt_response_thread = threading.Thread(target=GPT_annotation, args=(categorized_detections,))
        gpt_response_thread.start()
        
        if len(GPT_list) > 0:
            [response, GPT_time_cost, usage] = GPT_list[-1]
        
            response_list.append(response)
            # tokens_list.append([completion_tokens, prompt_tokens, total_tokens])
            time_list.append(GPT_time_cost)
                
        try:
            GPT_data = eval(response)
            level = GPT_data['danger_score']
            content = GPT_data['reason']
        except: pass

# ==============================================================

    last_frame = [tracker_id, boxes, classes, scores]

    # if frame_id % 10 == 0:
    #     for box in boxes:
    #         x1, y1, x2, y2 = map(int, box)
    #         center = (x1+(x2-x1)//2, y1+(y2-y1)//2)
    #         point_list.append(center)
    #         if len(point_list) > 20:
    #             point_list.popleft()
            
    # for center in point_list:
    #     cv2.circle(annotated_frame, center, radius=5, color=(0,0,255), thickness=10)


    '''
    Display the frame
    '''
    # Draw lines for the 'H'
    cv2.line(annotated_frame, (left_line_x, 0), (left_line_x, img_height), (0, 255, 0), 10)  # Left vertical line
    cv2.line(annotated_frame, (right_line_x, 0), (right_line_x, img_height), (0, 255, 0), 10)  # Right vertical line
    cv2.line(annotated_frame, (left_line_x, top_line_y), (right_line_x, bottom_line_y), (0, 255, 0), 10)  # Horizontal line
    
    if 'level' and 'content' in locals():
        text_1 = f"Emergency level: {level}"
        text_2 = content
        color = (0,255*(1-level),255*level)
        text_x = left_line_x + 10
        text_y = img_height - 100
        cv2.putText(annotated_frame, text_1, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        cv2.putText(annotated_frame, text_2, (text_x, text_y+50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    cv2.imshow('Video Frame', annotated_frame)
    output_video.write(annotated_frame)

    time.sleep(1/fps)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('last.jpg', annotated_frame)
        break


video_capture.release()
output_video.release()
# cv2.destroyAllWindows()

end_time = time.time()
total_time = end_time-start_time
FPS = (frame_id-display_start_frame)/total_time
print('FPS: ', FPS)


# cv2.imshow('Video Frame', annotated_frame)
# cv2.imwrite('test.jpg', annotated_frame)


print('Unique Objects: ', len(object_list))
print('Danger Labels: ', len(object_alert))


    





    












































