import os
import sys
import argparse
import glob
import cv2

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker


###################################



import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))



def get_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image



import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)






####################################












tracker_name = 'mixformer_vit_online'
tracker_param = 'baseline'


tracker_params = {'model': 'mixformer_vit_base_online.pth.tar', 'update_interval': 10, 'online_sizes': 5, 'search_area_scale': 4.5, 'max_score_decay': 1.0, 'vis_attn': 0}


tracker = Tracker(tracker_name, tracker_param, "lasot", tracker_params=tracker_params)



params = tracker.params



params.tracker_name = tracker_name
params.tracker_param  = tracker_param


multiobj_mode = getattr(params, 'multiobj_mode', getattr(tracker.tracker_class, 'multiobj_mode', 'default'))


tr = tracker.create_tracker(params)



images_path = '/usr/mvl2/esdft/cat/'
frames = glob.glob(f'{images_path}/*.jpg')

#bbox = [85.0000,294.0000,330.0000,420.0000]
bbox = [1120.0000, 605.0000, 145.0000, 105.0000]

def _read_image(image_file: str):
    im = cv2.imread(image_file)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)



def _build_init_info(box):
    return {'init_bbox': box}


img = _read_image(frames[0])


tr.initialize(img, _build_init_info(bbox))



for frame in frames:

    image = _read_image(frame)

    #info = OrderedDict()
    #info['previous_output'] = prev_output


    out = tr.track(image)
    #prev_output = OrderedDict(out)

    #state = out['target_bbox']

    #mask = out['segmentation']

    state = out['target_bbox']


    start_point = (int(state[0]), int(state[1]))
    end_point = (int(state[0]+state[2]), int(state[1]+state[3]))


    predictor.set_image(image)
    input_box = np.array([start_point[0], start_point[1],  end_point[0],  end_point[1]])

    masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
    )

    mask = get_mask(masks)

    img = cv2.imread(frame, 0)
    cv2.rectangle(img, start_point, end_point, color=(255,255,255), thickness=5)
    cv2.imshow('image', img)

    cv2.imshow('image', mask)
    cv2.waitKey(1)



