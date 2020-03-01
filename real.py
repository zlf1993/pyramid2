from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import CenterNet as net
import os
from coco.coco import CenterNetTestConfig, CocoDataset
from utils import utils as cocoutils
from PIL import Image
from utils import visualize
from utils.utils import resize_mask
import matplotlib.pyplot as plt
import cv2
import time

import os
import sys
import random
import itertools
import colorsys
from skimage.measure import find_contours

import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon


def ccc(config, num_select, index, select_bbox, exist_i, class_seg, num_i, pic_preg):
    if num_i == 0:
        masks = np.zeros([int(config.IMAGE_MAX_DIM//config.STRIDE), int(config.IMAGE_MAX_DIM//config.STRIDE), num_select], np.float32)
    else:
        masks = seg_instance(config, num_select, index, select_bbox, exist_i, class_seg, num_i, pic_preg)
    return masks


def seg_instance(config, num_select, index, select_bbox, exist_i, class_seg, num_i, pic_preg):
    # [num_g_of_i]
    box_i = select_bbox[exist_i, ...]
    # plt.imshow(class_seg)
    # plt.show()
    # shape of coordinate equals [h_y_num, w_x_mun]
    h = range(int(config.IMAGE_MAX_DIM//config.STRIDE))
    [meshgrid_x, meshgrid_y] = np.meshgrid(h, h)
    meshgrid_y = np.expand_dims(meshgrid_y, axis=-1) + 0.5  # [Y, X, -1]
    meshgrid_x = np.expand_dims(meshgrid_x, axis=-1) + 0.5

    box_y1 = np.reshape(box_i[..., 0], [1, 1, -1])
    box_x1 = np.reshape(box_i[..., 1], [1, 1, -1])
    box_y2 = np.reshape(box_i[..., 2], [1, 1, -1])
    box_x2 = np.reshape(box_i[..., 3], [1, 1, -1])
    dist_l = meshgrid_x - box_x1  # (y, x, num_g)
    dist_r = box_x2 - meshgrid_x
    dist_t = meshgrid_y - box_y1
    dist_b = box_y2 - meshgrid_y

    # [y, x, num]
    grid_y_mask = (dist_t > 0.).astype(np.float32) * (dist_b > 0.).astype(np.float32)
    grid_x_mask = (dist_l > 0.).astype(np.float32) * (dist_r > 0.).astype(np.float32)

    class_seg = np.expand_dims(class_seg, axis=-1)
    n_seg = np.tile(class_seg, [1, 1, num_i])  # (y, x, num_g)
    rect_mask = grid_y_mask * grid_x_mask
    # for i in range(num_i):
    #     plt.imshow(rect_mask[..., i])
    #     plt.show()

    dependent = np.expand_dims((np.sum(rect_mask, axis=2)>1).astype(np.float32), -1)
    common_mask_n = rect_mask * dependent * n_seg
    # print(np.shape(common_mask_n))
    # for i in range(num_i):
    #     plt.imshow(common_mask_n[..., i])
    #     plt.show()

    seperate_mask = rect_mask * n_seg - common_mask_n * n_seg
    # print(np.shape(seperate_mask))
    # for i in range(num_i):
    #     plt.imshow(seperate_mask[..., i])
    #     plt.show()
    # select_tlbr
    p_x1 = meshgrid_x - np.expand_dims(pic_preg[..., 1], -1)
    p_x2 = meshgrid_x + np.expand_dims(pic_preg[..., 3], -1)
    p_y1 = meshgrid_y - np.expand_dims(pic_preg[..., 0], -1)
    p_y2 = meshgrid_y + np.expand_dims(pic_preg[..., 2], -1)

    inter_width = np.minimum(box_x2, p_x2) - np.maximum(box_x1, p_x1)
    inter_height = np.minimum(box_y2, p_y2) - np.maximum(box_y1, p_y1)
    inter_area = inter_width * inter_height
    union_area = (box_y2 - box_y1) * (box_x2 - box_x1) + (p_y2 - p_y1) * (p_x2 - p_x1) - inter_area
    iou = inter_area / (union_area + 1e-12)

    iou_mask = iou * common_mask_n
    # print(np.shape(iou_mask))
    # for i in range(num_i):
    #     plt.imshow(iou_mask[..., i])
    #     plt.show()
    iou_max = np.expand_dims(np.amax(iou_mask, axis=-1), -1)
    # print(np.shape(iou_max))

    divide_mask = (np.equal(iou_mask, iou_max)).astype(np.float32)*common_mask_n
    masks = seperate_mask + divide_mask
    mask_score = masks*iou
    # for i in range(num_i):
    #     plt.imshow(mask_score[..., i])
    #     plt.show()
    masks = (mask_score > config.BOX_THRESHOLD).astype(np.float32)
    temp_masks = np.zeros([int(config.IMAGE_MAX_DIM//config.STRIDE), int(config.IMAGE_MAX_DIM//config.STRIDE), num_select], np.float32)
    temp_masks[..., index] = masks
    return temp_masks

# Set Coco
ROOT_DIR = os.path.abspath("../")

Config = CenterNetTestConfig()
Config.display()

# dataset = CocoDataset()
# COCO_DIR = ROOT_DIR + "/coco2014"
# dataset.load_coco(Config, COCO_DIR, "train", class_ids=[1, 17, 18])
# dataset.prepare()
class_names = {0:"BG",1:"person",2:"bicycle",3:"car",4:"motorcycle",5:"airplane",6:"bus",7:"train",8:"truck",9:"boat",10:"traffic light",11:"fire hydrant",
               12:"stop sign",13:"parking meter",14:"bench",15:"bird",16:"cat",17:"dog",18:"horse",19:"sheep",20:"cow",21:"elephant",22:"bear",
               23:"zebra",24:"giraffe",25:"backpack",26:"umbrella",27:"handbag",28:"tie",29:"suitcase",30:"frisbee",31:"skis",32:"snowboard",
               33:"sports ball",34:"kite",35:"baseball bat",36:"baseball glove",37:"skateboard",38:"surfboard",39:"tennis racket",40:"bottle",
               41:"wine glass",42:"cup",43:"fork",44:"knife",45:"spoon",46:"bowl",47:"banana",48:"apple",49:"sandwich",50:"orange",51:"broccoli",
               52:"carrot",53:"hot dog",54:"pizza",55:"donut",56:"cake",57:"chair",58:"couch",59:"potted plant",60:"bed",61:"dining table",
               62:"toilet",63:"tv",64:"laptop",65:"mouse",66:"remote",67:"keyboard",68:"cell phone",69:"microwave",70:"oven",71:"toaster",
               72:"sink",73:"refrigerator",74:"book",75:"clock",76:"vase",77:"scissors",78:"teddy bear",79:"hair drier",80:"toothbrush"}


centernet = net.CenterNet(Config, "pc")

image = Image.open('COCO_val2014_000000000761.jpg')
#image = Image.open('COCO_val2014_000000000241.jpg')
#image = Image.open('COCO_val2014_000000000474.jpg')
# image = Image.open('COCO_val2014_000000034372.jpg')
#image = Image.open('people.jpeg')
# image = Image.open('children.jpg')
#image = Image.open('car1.jpeg')
#image = Image.open('car2.jpg')
#image = Image.open('car3.jpg')



# image = np.array(image)
# image, window, scale, padding, crop = cocoutils.resize_image(
#             image,
#             min_dim=Config.IMAGE_MIN_DIM,
#             min_scale=Config.IMAGE_MIN_SCALE,
#             max_dim=Config.IMAGE_MAX_DIM,
#             mode=Config.IMAGE_RESIZE_MODE)


# [select_center, select_scores, select_bbox, select_class_id, class_seg, pic_preg] = centernet.test_one_image(image)

# select_center = select_center[0]
# select_class_id = select_class_id[0]
# select_scores = select_scores[0]
# select_bbox = select_bbox[0]

# class_seg = class_seg[0]

# if np.shape(select_center)[0] > Config.DETECTION_MAX_INSTANCES:
#     num_select = Config.DETECTION_MAX_INSTANCES
# else:
#     num_select = np.shape(select_center)[0]
# select_scores = select_scores[0:num_select, ...]
# select_center = select_center[0:num_select, ...]
# select_class_id = select_class_id[0:num_select, ...]
# select_bbox = select_bbox[0:num_select, ...]


# final_masks = np.zeros([int(Config.IMAGE_MAX_DIM//Config.STRIDE), int(Config.IMAGE_MAX_DIM//Config.STRIDE), num_select], np.float32)

# for i in range(Config.NUM_CLASSES):
#     exist_i = np.equal(select_class_id, i)  # [0,1,...]
#     exist_int = exist_i.astype(int)
#     index = np.where(exist_int>0)[0]  # [a, b, 5, 8..]
#     num_i = np.sum(exist_int)
#     masks = ccc(Config, num_select, index, select_bbox, exist_i, class_seg[..., i], num_i, pic_preg)
#     final_masks = final_masks + masks

# # TODO: resize masks
# padding = [(0, 0), (0, 0), (0, 0)]
# stride_mask = resize_mask(final_masks, Config.STRIDE, padding, 0)
# stride_mask = cv2.medianBlur(stride_mask, 5)
# masks = stride_mask.astype(np.uint8).astype(np.float)
# if len(np.shape(masks)) is 2:
#     masks = np.expand_dims(masks, -1)
# class_names = {0:"bg", 1:'person', 2:"car"}
# visualize.display_instances(image, select_center*int(Config.STRIDE)+1, select_bbox*int(Config.STRIDE), masks, select_class_id + 1, class_names, select_scores, show_mask=True)

time.sleep(2)

#cap = cv2.VideoCapture("./source.gif")

cap = cv2.VideoCapture(0)

width = 512  #定义摄像头获取图像宽度
height = 512   #定义摄像头获取图像长度

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  #设置宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  #设置长度


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.3):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1.0,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, centers, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=False, show_bbox=True, show_centers=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    centers: [num_instance, (y1, x1)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = centers.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == class_ids.shape[0]

    colors = colors or random_colors(N)
    masked_image = image.astype(np.int8).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i]*100 if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.0f}%".format(label, score) if score else label
        else:
            caption = captions[i]
        cv2.putText(image, caption, (x1, y1), fontFace=0, fontScale=0.5, color=(0, 0, 255))
        cy, cx = centers[i]
        if show_centers:
            cv2.circle(image, (cx, cy), 3, color, 0)
        if show_mask:
            # Mask
            mask = masks[:, :, i]
            image = apply_mask(image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            # padded_mask = np.zeros(
            #     (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            # padded_mask[1:-1, 1:-1] = mask
            # contours = find_contours(padded_mask, 0.5)
            # for verts in contours:
            #     # Subtract the padding and flip (y, x) to (x, y)
            #     verts = np.fliplr(verts) - 1
            #     p = Polygon(verts, facecolor="none", edgecolor=color)
            #     ax.add_patch(p)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            # padded_mask = np.zeros(
            #     (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            # padded_mask[1:-1, 1:-1] = mask
            # contours = find_contours(padded_mask, 0.5)
            # print(contours)
            # cv2.polylines(masked_image, contours, 1, color)
    cv2.imshow('result', image)


_, ax = plt.subplots(1, figsize=(16, 16))
while cap.isOpened():

    try:
        # clear_output(wait=True)
        ret, frame = cap.read()
        if not ret:
            break  
        frame, window, scale, padding, crop = cocoutils.resize_image(
            frame,
            min_dim=Config.IMAGE_MIN_DIM,
            min_scale=Config.IMAGE_MIN_SCALE,
            max_dim=Config.IMAGE_MAX_DIM,
            mode=Config.IMAGE_RESIZE_MODE)

        start = time.clock()

        [select_center, select_scores, select_bbox, select_class_id, class_seg, pic_preg] = centernet.test_one_image(frame)
        elapsed = (time.clock() - start)
        fps = 1.0/elapsed
        print(fps)
        start = time.clock()
        if select_center is None:
            continue
        select_center = select_center[0]
        select_class_id = select_class_id[0]
        select_scores = select_scores[0]
        select_bbox = select_bbox[0]
        class_seg = class_seg[0]
        num_select = np.shape(select_center)[0]
        
        final_masks = np.zeros([int(Config.IMAGE_MAX_DIM//Config.STRIDE), int(Config.IMAGE_MAX_DIM//Config.STRIDE), num_select], np.float32)

        for i in range(Config.NUM_CLASSES):
            exist_i = np.equal(select_class_id, i)  # [0,1,...]
            exist_int = exist_i.astype(int)
            index = np.where(exist_int>0)[0]  # [a, b, 5, 8..]
            num_i = np.sum(exist_int)
            masks = ccc(Config, num_select, index, select_bbox, exist_i, class_seg[..., i], num_i, pic_preg)
            final_masks = final_masks + masks

        # TODO: resize masks
        padding = [(0, 0), (0, 0), (0, 0)]
        stride_mask = resize_mask(final_masks, 8, padding, 0)

        # stride_mask = cv2.medianBlur(stride_mask, 5)
        masks = stride_mask.astype(np.int8).astype(np.float32)
        # print(np.amax(masks))
        if len(np.shape(masks)) is 2:
            masks = np.expand_dims(masks, -1)
        class_names = {0:"bg", 1:'person', 2:"car"}
        display_instances(frame, select_center*8+3, select_bbox*8, masks, select_class_id + 1, class_names, select_scores, show_mask=True)


        # cv2.imshow('result', frame)
        elapsed = (time.clock() - start)
        fps = 1.0 / elapsed
        print(fps)
        # time.sleep(0.02)
        c = cv2.waitKey(1)
        if c == 27:
            break
    except KeyboardInterrupt:
        video.release()

    
cap.release()
cv2.destroyAllWindows()






