import numpy as np
import cv2



def rle_to_mask(rle, width, height):
    """
    rle: input rle mask encoding
    each evenly-indexed element represents number of consecutive 0s
    each oddly indexed element represents number of consecutive 1s
    width and height are dimensions of the mask
    output: 2-D binary mask
    """
    # allocate list of zeros
    v = [0] * (width * height)

    # set id of the last different element to the beginning of the vector
    idx_ = 0
    for i in range(len(rle)):
        if i % 2 != 0:
            # write as many 1s as RLE says (zeros are already in the vector)
            for j in range(rle[i]):
                v[idx_+j] = 1
                
        idx_ += rle[i]
    
    
    # reshape vector into 2-D mask
    # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
    return np.array(v, dtype=np.uint8).reshape((height, width))

def create_mask_from_string(mask_encoding):
    """
    mask_encoding: a string in the following format: x0, y0, w, h, RLE
    output: mask, offset
    mask: 2-D binary mask, size defined in the mask encoding
    offset: (x, y) offset of the mask in the image coordinates
    """
    elements = [int(el) for el in mask_encoding]
    tl_x, tl_y, region_w, region_h = elements[:4]
    rle = np.array([el for el in elements[4:]], dtype=np.int32)

    # create mask from RLE within target region
    mask = rle_to_mask(rle, region_w, region_h)
    

    return mask, (tl_x, tl_y)




def parse_region(string: str):

    if string[0] == 'm':
        # input is a mask - decode it
        m_, offset_ = create_mask_from_string(string[1:].split(','))
        return m_




def _mask_to_bbox(mask: np.ndarray):

    mask = mask.astype(int)
    xs = mask.sum(axis=-2).nonzero()[0].tolist()
    ys = mask.sum(axis=-1).nonzero()[0].tolist()

    if len(ys) > 0 and len(xs) > 0:
        x, y, w, h = xs[0], ys[0], xs[-1] - xs[0], ys[-1] - ys[0]
    else:
        x, y, w, h = 0, 0, 0, 0

    return [x, y, w, h]




def get_bbox(gt_path):

   with open(gt_path, 'r') as f:
      lines = f.readlines()
    
   mask = parse_region(lines[0])
    
   return _mask_to_bbox(mask)




