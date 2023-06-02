import os
import sys
import argparse
import glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import torch


prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker
from lib.test.evaluation.environment import env_settings

from vot_data_preprocessing import get_bbox

tracker_name = 'mixformer_vit_online'
tracker_param = 'baseline'


tracker_params = {'model': 'mixformer_vit_base_online.pth.tar', 'update_interval': 10, 'online_sizes': 5, 'search_area_scale': 4.5, 'max_score_decay': 1.0, 'vis_attn': 0}


def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)



def get_tracker(run_id):

    tracker = Tracker(tracker_name, tracker_param, "lasot",run_id=1, tracker_params=tracker_params)
    params = tracker.params
    params.tracker_name = tracker_name
    params.tracker_param  = tracker_param
    tr = tracker.create_tracker(params)

    return tr








images_path = '/usr/mvl2/esdft/cat/'
frames = glob.glob(f'{images_path}/*.jpg')



file_path = '/usr/mvl2/esdft/development_data/sequences/cat-18/'
gt_files = glob.glob(f'{file_path}/groundtruth*.txt')


bboxes = []

for gt_file in gt_files:
    bboxes.append(get_bbox(gt_file))



def _read_image(image_file: str):
    im = cv2.imread(image_file)
    #return im
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)



def _build_init_info(box):
    return {'init_bbox': box}


img = _read_image(frames[0])



trackers = []

for ind in range(len(bboxes)):

    tr = get_tracker(run_id=ind)
    tr.initialize(img, _build_init_info(bboxes[ind]))
    trackers.append(tr)


colors = []

for ind in range(len(bboxes)):

    colors.append(random_color())



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



import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)



for frame in frames:

    image = _read_image(frame)
    predictor.set_image(image)


    outs = []

    for tr in trackers:
        outs.append(tr.track(image))



    #img = cv2.imread(frame)

    #plt.imshow(image)


    for ind in range(len(outs)):
        state = outs[ind]['target_bbox']


        start_point = (int(state[0]), int(state[1]))
        end_point = (int(state[0]+state[2]), int(state[1]+state[3]))
        input_box = np.array([start_point[0], start_point[1], end_point[0], end_point[1]])
 
        masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
        )

        print(masks)

        #cv2.rectangle(image, start_point, end_point, color=colors[ind], thickness=5)
        #show_mask(masks, plt.gca())

    #cv2.imshow('image', image)
    #plt.axis('off')
    #plt.show()
    #plt.pause(0.0000000000001)
    #plt.draw()

    #cv2.imshow('image', mask*255)
    #cv2.waitKey(1)



