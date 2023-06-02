import os
import sys
import argparse
import glob
import cv2
import random

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



#tracker1 = Tracker(tracker_name, tracker_param, "lasot",run_id=1, tracker_params=tracker_params)



#params1 = tracker1.params


#params1.tracker_name = tracker_name
#params1.tracker_param  = tracker_param


#tr1 = tracker1.create_tracker(params1)

#tr1 = get_tracker(run_id=1)
#tr2 = get_tracker(run_id=2)




images_path = '/usr/mvl2/esdft/cat/'
frames = glob.glob(f'{images_path}/*.jpg')



file_path = '/usr/mvl2/esdft/development_data/sequences/cat-18/'
gt_files = glob.glob(f'{file_path}/groundtruth*.txt')


bboxes = []

for gt_file in gt_files:
    bboxes.append(get_bbox(gt_file))



def _read_image(image_file: str):
    im = cv2.imread(image_file)
    return im
    #return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)



def _build_init_info(box):
    return {'init_bbox': box}


img = _read_image(frames[0])


#tr1.initialize(img, _build_init_info(bbox1))
#tr2.initialize(img, _build_init_info(bbox2))


trackers = []

for ind in range(len(bboxes)):

    tr = get_tracker(run_id=ind)
    tr.initialize(img, _build_init_info(bboxes[ind]))
    trackers.append(tr)


colors = []

for ind in range(len(bboxes)):

    colors.append(random_color())


for frame in frames:

    image = _read_image(frame)

    #info = OrderedDict()
    #info['previous_output'] = prev_output

    outs = []

    for tr in trackers:
        outs.append(tr.track(image))


    #out1 = outs[0]
    #out2 = outs[1]
    #prev_output = OrderedDict(out)

    #state = out['target_bbox']

    #mask = out['segmentation']

    img = cv2.imread(frame)

    for ind in range(len(outs)):
        state = outs[ind]['target_bbox']


        start_point = (int(state[0]), int(state[1]))
        end_point = (int(state[0]+state[2]), int(state[1]+state[3]))
        cv2.rectangle(img, start_point, end_point, color=colors[ind], thickness=5)

    #img = cv2.imread(frame)
    #cv2.rectangle(img, start_point, end_point, color=(255,0,0), thickness=5)
    #cv2.rectangle(img, start_point1, end_point1, color=(0,0,255), thickness=5)
    cv2.imshow('image', img)

    #cv2.imshow('image', mask*255)
    cv2.waitKey(1)



