#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import pdb
import math
from PIL import Image

node_positions = {0: (469, 775), 1: (469, 647), 2: (467, 526), 3: (159, 466), 4: (168, 460), 5: (114, 430), 6: (224, 408), 7: (96, 405), 8: (468, 405), 9: (133, 368), 10: (31, 347), 11: (285, 348), 12: (348, 285), 13: (466, 284), 14: (297, 206), 15: (466, 163), 16: (468, 39)}

node_neighbors = {0: [3], 1: [6], 2: [11], 3: [4, 5, 0], 4: [6, 3], 5: [7, 3], 6: [11, 1, 4], 7: [9, 10, 5], 8: [12], 9: [14, 7], 10: [7], 11: [12, 2, 6], 12: [13, 15, 8, 11], 13: [12], 14: [16, 9], 15: [12], 16: [14]}

def grab_endpoints(node_neighbors, node_positions):
    endpoints = []
    for key, value in node_neighbors.iteritems():
        if len(value) == 1:
            endpoints.append(node_positions[key])
    return endpoints 

def get_text(img,x,y,w,h):
    cropped = img[y:y+h, x:x+w]
    cv2.imwrite('~/Downloads/temp.png', cropped)
    text = pytesseract.image_to_string(Image.open('~/Downloads/temp.png'),lang='lat')
    return text

def get_text_positions(contours, grays):
    storage = {}
    for index, contour in enumerate(contours):
        [x,y,w,h] = cv2.boundingRect(contour)
        if h < 10 or w < 10: continue
        if h > grays.shape[0] * 0.5 or w > grays.shape[1] * 0.5: continue
        words = get_text(image,x,y,w,h)
        storage[(x,y)] = words
        # pdb.set_trace()
    return storage

def combine_separate_words(text_positions, end_matches):
    # keyed_on_word = {}
    end_with_words = {}
    text_pos = text_positions
    # for key, value in text_positions.iteritems():
    #     keyed_on_word[value] = key 
    for key, value in end_matches.iteritems():
        if value in end_with_words:
            end_with_words[value].append(text_pos[key])
        else:
            end_with_words[value] = [text_pos[key]]
    return end_with_words


def find_nearest(text_positions, end_positions):
    ends = end_positions
    match = {}
    small = []
    for text_pos in text_positions.keys():
        print(text_pos)
        smallest_dist = 10000000
        for pos in ends:
            dist = math.hypot(text_pos[0] - pos[0], text_pos[1] - pos[1])
            print dist
            # print(text_pos, pos)
            if dist < smallest_dist:
                print(smallest_dist, 'smallest!', dist, 'new smallest!')
                smallest_dist = dist
                small.append(pos)
        index = len(small)-1
        match[text_pos] = small[index]
    return match

def assign_text_to_leaves(opencv_img, node_neighbors, node_positions):
    node_neighbors = node_neighbors
    node_positions = node_positions
    param = 2
    grays = cv2.cvtColor(opencv_img,cv2.COLOR_BGR2GRAY)

    _,thresh = cv2.threshold(grays,150,255,cv2.THRESH_BINARY_INV) # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(thresh,kernel,iterations = param) # dilate
    im2, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours

    ends = grab_endpoints(node_neighbors, node_positions)
    text_name_pos = get_text_positions(contours, grays)
    text_match = find_nearest(text_name_pos, ends)

    for key, value in text_match.iteritems():
        color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        cv2.rectangle(img, key, (key[0]+50,key[1]+50), color, 2)
        cv2.rectangle(img, value, (value[0]+50,value[1]+50), color, 2)

    cv2.imshow('%i' % param,img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()















