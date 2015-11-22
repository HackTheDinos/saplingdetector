#!/usr/bin/env python3
import cv2
import numpy as np
import sys
from scipy.ndimage import rotate

def get_rotated_pos(x, y, rot):
    return (int(x * np.cos(rot) - y * np.sin(rot) + 0.5),
            int(x * np.sin(rot) + y * np.cos(rot) + 0.5))

def find_connected_nodes(image, i, nodes):
    out = []
    x, y = nodes[i]
    for j in range(len(nodes)):
        if i == j: continue
        distance = np.sqrt(np.sum(np.power(np.array(nodes[i]) - np.array(nodes[j]), 2)))
        dst_x, dst_y = nodes[j]

        angle = np.arctan2(dst_y - y, dst_x - x)
        
        min_x = min(x, dst_x)
        min_y = min(y, dst_y)
        max_x = max(x, dst_x)        
        max_y = max(y, dst_y)
        cutout = np.copy(image[min_y - 1  : max_y + 2, min_x - 1 : max_x + 2])

        if ((x - min_x) == 0 and (y - min_y) == 0) or ((dst_x - min_x) == 0 and (dst_y - min_y) == 0):
            start_x = image.shape[-1] // 3
            end_x = image.shape[-1]
            start_y = image.shape[-2] // 3
            end_y = image.shape[-2]
        
        contour = find_tree_contour(cutout, 1)
        other = False
        for k in range(len(nodes)):
            if k == j or k == i: continue
            k_x, k_y = nodes[k]
            k_x = k_x - min_x + 1
            k_y = k_y - min_y + 1
            if k_x >= 0 and k_x < cutout.shape[-1] and k_y >= 0 and k_y < cutout.shape[-2]:
                if contour[k_y, k_x] < 128:
                    other = True
                    break

        if other == False and contour[y - min_y + 1, x - min_x + 1] < 128 and contour[dst_y - min_y + 1, dst_x - min_x + 1] < 128:
            out.append(nodes[j])                
    return out

def find_nearest_connected_nodes(image, i, nodes):
    out = []
    x, y = nodes[i]
    min_j = i
    min_dist = np.max(image.shape)
    distances = []
    good_js = []
    for j in range(len(nodes)):
        distance = np.sqrt(np.sum(np.power(np.array(nodes[i]) - np.array(nodes[j]), 2)))
        distances.append(distance)
        if i == j: continue
        dst_x, dst_y = nodes[j]

        angle = np.arctan2(dst_y - y, dst_x - x)
        
        min_x = min(x, dst_x)
        min_y = min(y, dst_y)
        max_x = max(x, dst_x)        
        max_y = max(y, dst_y)
        cutout = np.copy(image[min_y - 1  : max_y + 2, min_x - 1 : max_x + 2])

        contours = find_tree_contours(cutout, 1)
        for contour in contours:
            if contour[y - min_y + 1, x - min_x + 1] < 128 and contour[dst_y - min_y + 1, dst_x - min_x + 1] < 128:
                if distance < min_dist:
                    min_j = j
                    min_dist = distance
                good_js.append(j)
    for j in good_js:
        if distances[j] < min_dist * 1.5:
            out.append(nodes[j])
    return out

# Nodes from corners
def nodes_from_corners(image, gray, points, max_dist=16, iterations=1):
    if iterations == 0:
        out = []
        for point in points:
            x, y = map(int, point)
            A = gray[y - 2 : y + 3, x - 2 : x + 3]
            off_x, off_y = np.array(np.unravel_index(A.argmin(), A.shape)) - 2
            new_x, new_y = x + off_x, y + off_y
            if gray[new_y, new_x] < gray[y, x]:
                out.append((new_x,new_y))
            else:
                out.append((x,y))
                
        return out
    print('iternations:', iterations)
    nodes = []
    while len(points):
        point = points[-1]
        points.pop()
        neighbors = [point]
        for i in reversed(range(len(points))):
            dist = np.sqrt(np.sum(np.power(np.array(point) - np.array(points[i]), 2)))
            if dist < max_dist:
                neighbors.append(points[i])
                points.pop(i)
        neighbors = np.array(neighbors)
        point = neighbors[np.random.randint(len(neighbors))]
        point = np.mean(neighbors, axis=0)
        print(len(points))
        if iterations == 1:
            #image[point[1], point[0], :] = (255, 0, 0)
            cv2.circle(image, tuple(map(int, point)), 4, (255,128,0), 2)
        nodes.append(point)
    return nodes_from_corners(image, gray, nodes, max_dist + 2, iterations - 1)


# Find the contour of the tree
def find_tree_contour(gray, param):
    _,thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(thresh,kernel,iterations = param) # dilate
    im2, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 

    out = np.zeros(gray.shape, dtype=np.uint8) + 255
    
    for i, contour in enumerate(contours):
        [x,y,w,h] = cv2.boundingRect(contour)
        if h < gray.shape[0] * 0.5 or w < gray.shape[1] * 0.5: continue
        #cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 1)
        cv2.drawContours(out, contours, i, 0, -1)

    out2 = np.zeros(gray.shape, dtype=np.uint8) + 255
    out2[out == 0] = gray[out == 0]
    #out2[out < 2010] = 0
    return out2

def find_tree_contours(gray, param):
    _,thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(thresh,kernel,iterations = param) # dilate
    im2, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 

    outs = []
    for i, contour in enumerate(contours):
        [x,y,w,h] = cv2.boundingRect(contour)
        if h < gray.shape[0] * 0.5 or w < gray.shape[1] * 0.5: continue
        #cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 1)
        out = np.zeros(gray.shape, dtype=np.uint8) + 255
        cv2.drawContours(out, contours, i, 0, -1)
        outs.append(out)
    #out2[out < 2010] = 0
    return outs

def main():   
    filename = sys.argv[1]
    param = int(sys.argv[2])
    
    img = cv2.imread(filename)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img[y,x,:] = np.min(img[y,x])
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    tree_contour = find_tree_contour(gray, param)
    cv2.imshow('dst2', tree_contour)
        
    dst = cv2.cornerHarris(tree_contour, 3, 3, 0.01)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    corners = dst > 0.05 * dst.max()
    #img[corners] = [0, 0, 255]

    points = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if corners[y,x]:
                points.append((x,y))

    nodes = nodes_from_corners(img, gray, points, max_dist=6, iterations=3)
    node_neighbors = {}
    for i, node in enumerate(nodes):
        node_neighbors[i] = find_connected_nodes(gray, i, nodes)
        #node_neighbors[i] = find_nearest_connected_nodes(gray, i, nodes)
        node_x, node_y = node
        for neighbor in node_neighbors[i]:
            neighbor_x, neighbor_y = neighbor
            cv2.line(img, (node_x, node_y), (neighbor_x, neighbor_y), (255,0,0), 2)
            
    cv2.imshow('dst', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
