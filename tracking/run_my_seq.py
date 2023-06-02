import os
import sys
import argparse
import glob
import cv2

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker



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

    img = cv2.imread(frame, 0)
    cv2.rectangle(img, start_point, end_point, color=(255,255,255), thickness=5)
    cv2.imshow('image', img)

    #cv2.imshow('image', mask*255)
    cv2.waitKey(1)



