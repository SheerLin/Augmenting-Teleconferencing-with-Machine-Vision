import math

import cv2
import numpy as np

def draw_line(src, line, color):
    for col1, row1, col2, row2 in line:
        cv2.line(src, (col1, row1), (col2, row2), color, 3, cv2.LINE_AA)

def find_longest_line(src, lines, rowMin, colMin, rowMax, colMax, vertical):
    threshold = 15

    res = None
    resL = 0

    for line in lines:
        angle = 90
        for col1, row1, col2, row2 in line:
            if row1 < rowMin or row1 > rowMax or row2 < rowMin or row2 > rowMax or col1 < colMin or col1 > colMax or col2 < colMin or col2 > colMax:
                continue
            # cv2.line(src, (col1, row1), (col2, row2), (211, 211, 211), 3, cv2.LINE_AA)
            draw_line(src, line, (211,211,211))
            if col1 != col2:
                angle = math.atan((row1 - row2) / (col1 - col2))
                angle = abs(angle * 180 / math.pi)
            if vertical:
                if abs(angle - 90) < threshold:
                    L = math.sqrt((row1 - row2)**2 + (col1 - col2)**2)
                    if L > resL:
                        res = line
                        resL = L
            else:
                if abs(angle) < threshold:
                    L = math.sqrt((row1 - row2) ** 2 + (col1 - col2) ** 2)
                    if L > resL:
                        res = line
                        resL = L
    return res

def find_intersection(line1, line2):
    # extract points
    x1, y1, x2, y2 = line1[0].astype(np.longlong)
    x3, y3, x4, y4 = line2[0].astype(np.longlong)
    # compute determinant
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    # Px = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
    # Py = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
    return Px, Py

def is_valid_point(src, point):
    col, row = point
    height = src.shape[0]
    width = src.shape[1]
    if row < 0 or row > height or col <0 or col > width:
        return False
    return True

def find_rect(src, lines):
    height = src.shape[0]
    width = src.shape[1]
    top = find_longest_line(src, lines, 0, 0, height/3, width, vertical=False)
    bottom = find_longest_line(src, lines, 2*height/3, 0, height, width, vertical=False)
    left = find_longest_line(src, lines, 0, 0, height, width/3, vertical=True)
    right = find_longest_line(src, lines, 0, 2*width/3, height, width, vertical=True)

    if top is None:
        top = np.array((0, 0, width, 0)).reshape(1,4)
    if bottom is None :
        bottom = np.array((0, height, width, height)).reshape(1,4)
    if left is None :
        left = np.array((0, 0, 0, height)).reshape(1,4)
    if right is None:
        right = np.array((width, 0, width, height)).reshape(1,4)

    draw_line(src, top, (0, 0, 255))
    draw_line(src, bottom, (0, 0, 255))
    draw_line(src, left, (0, 0, 255))
    draw_line(src, right, (0, 0, 255))

    topLeft = find_intersection(top, left)
    topRight = find_intersection(top, right)
    bottomLeft = find_intersection(bottom, left)
    bottomRight = find_intersection(bottom, right)

    if not is_valid_point(src, topLeft):
        topLeft = 0,0
    if not is_valid_point(src, topRight):
        topRight = width, 0
    if not is_valid_point(src, bottomLeft):
        bottomLeft = 0, height
    if not is_valid_point(src, bottomRight):
        bottomRight = width, height

    return np.array([topLeft, topRight, bottomLeft, bottomRight])

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    # print("rect", rect)
    return rect

def fill_frame(img, width, height):
    h, w, _ = img.shape

    # scale up the image preserving aspect ratio
    height_ratio = float(height)/h
    width_ratio = float(width)/w    
    if height_ratio < width_ratio:
        h = height
        w = int(round(w * height_ratio))
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        h = int(round(h * width_ratio))
        w = width
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    # calculate height difference and top-bottom border
    hdiff = height - h
    top = hdiff // 2
    bottom = hdiff // 2
    if hdiff % 2 != 0:
        bottom += 1
    
    # calculate width difference and left-right border
    wdiff = width - w
    left = wdiff // 2
    right = wdiff // 2
    if wdiff % 2 != 0:
        right += 1
    
    # fill borders with white
    white = [255, 255, 255]
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=white)
    return img

def four_point_transform(image, pts, width, height):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    maxWidth = min(maxWidth, width)

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    maxHeight = min(maxHeight, height)

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # fix the frame size by scaling & filling white
    im = fill_frame(warped, width, height)

    # return the warped image
    return image

def rescale_by_height(image, target_height, method=cv2.INTER_LANCZOS4):
    """Rescale `image` to `target_height` (preserving aspect ratio)."""
    w = int(round(target_height * image.shape[1] / image.shape[0]))
    return cv2.resize(image, (w, target_height), interpolation=method)

def rescale_by_width(image, target_width, method=cv2.INTER_LANCZOS4):
    """Rescale `image` to `target_width` (preserving aspect ratio)."""
    h = int(round(target_width * image.shape[0] / image.shape[1]))
    return cv2.resize(image, (target_width, h), interpolation=method)
