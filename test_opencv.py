# -*- coding: utf-8 -*-
import copy
import os
import string
import sys

import cv2
import cv
import numpy as np
from matplotlib import pyplot

import Polygon

import captcha

PROJECT_ROOT = os.path.dirname(__file__)


def _add_subplot(img, title, rows=1, cols=1, plot_number=1):
    pyplot.subplot(rows, cols, plot_number)
    pyplot.imshow(img, 'gray')
    pyplot.title(title)


def invert(image):
    return (255 - image)


def skeletonization(img):
    '''
    http://opencvpython.blogspot.ru/2012/05/skeletonization-using-opencv-python.html
    '''
    img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # ret, img = cv2.threshold(img, 127, 255, 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            break

    cv2.imwrite("skel.png", skel)
    return skel


def bbox_to_polygon(x1, y1, x2, y2):
    width = x2 - x1
    height = y2 - y1
    return Polygon.Polygon([
        [x1, y1],
        [x1, y2],
        [x2, y2],
        [x2, y1],
    ])


def intersect_bbox(bbox1, bbox2):
    rect1 = bbox_to_polygon(*bbox1)
    rect2 = bbox_to_polygon(*bbox2)
    return rect1 & rect2


def merge_bbox(bbox1, bbox2):
    bboxes = [bbox1, bbox2]
    new_bbox = [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]
    return new_bbox


def find_and_merge_intersection_bboxes(bboxes_list):
    new_bboxes_list = []
    already_merged_bboxes = []
    for bbox1 in bboxes_list:

        if bbox1 in already_merged_bboxes:
            continue

        new_bbox = bbox1
        for bbox2 in bboxes_list:
            intersect = intersect_bbox(new_bbox, bbox2)
            if intersect and intersect.area() != intersect.aspectRatio():
                new_bbox = merge_bbox(new_bbox, bbox2)
                already_merged_bboxes.append(bbox2)
        new_bboxes_list.append(new_bbox)
    return new_bboxes_list


def split_symbols(img):
    '''
    return (img with bordered, list of subimages)

    '''
    img = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = invert(img_gray)

    # thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
    img_gray = cv2.GaussianBlur(img_gray, (11, 11), 0)
    cv2.imshow("0", img_gray)
    ret, thresh = cv2.threshold(img_gray, 42, 255, 0)

    cv2.imshow("1", thresh)

    # skel = skeletonization(img)

    cv2.imshow("2", thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = map(cv2.boundingRect, contours[1:])

    max_height = [min(b[1] for b in bboxes), max(b[1] + b[-1] for b in bboxes)]
    print max_height

    collected_bboxes = []

    for i, (cnt, hie) in enumerate(zip(contours, hierarchy[0])):
        bbox = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        print i, area, hie, bbox
        if i == 0:
            continue

        if hie[-1] == 0:
            x, y, width, height = bbox
            x1 = x
            y1 = y
            x2 = x + width
            y2 = y + height
            y1, y2 = max_height

            collected_bboxes.append([x1, y1, x2, y2])

    collected_bboxes = find_and_merge_intersection_bboxes(collected_bboxes)

    subimages = []
    for i, bbox in enumerate(collected_bboxes):
        x1, y1, x2, y2 = bbox

        sub_img = img[y1:y2, x1:x2]
        sub_img = cv2.copyMakeBorder(sub_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        subimages.append(sub_img)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

        text_color = (255, 0, 0) #color as (B,G,R)
        cv2.putText(img, str(i), (x1, y1 + 20),
            cv2.FONT_HERSHEY_PLAIN, 1.0, text_color,
            thickness=1, lineType=cv2.CV_AA)

    return img, subimages


def save_char(image, char, output_dir):
    save_dir = os.path.join(output_dir, char)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in xrange(1, 10000):
        save_filepath = os.path.join(save_dir, "{:0=5}_{}.png".format(i, char))
        if not os.path.exists(save_filepath):
            cv2.imwrite(save_filepath, image)
            break
    return image


def build_study_images():
    avaiable_chars = string.digits

    captcha.get_captcha(chars=avaiable_chars).save('captcha.png')
    img = cv2.imread('captcha.png')
    new_img, sub_images = split_symbols(img)
    cv2.imshow("all_img", new_img)

    output_dir = os.path.join(PROJECT_ROOT, "training")

    avaiable_keys = map(ord, avaiable_chars)

    for sub_img in sub_images:
        cv2.imshow("char", sub_img)
        while True:
            key = cv2.waitKey(0) % 256

            if key == 27:
                sys.exit() # ESC

            elif key in avaiable_keys:
                char = chr(key)
                print key, char
                save_char(sub_img, char, output_dir)
                break


def main():
    captcha.get_captcha(u"ЙйgiÄWWW").save('captcha.png')
    img = cv2.imread('captcha.png')
    new_img, sub_images = split_symbols(img)

    _add_subplot(img, "ORIGINAL", cols=len(sub_images) + 2, plot_number=1)
    _add_subplot(new_img, "PARSE", cols=len(sub_images) + 2, plot_number=2)
    for i, sub_img in enumerate(sub_images, start=3):
        _add_subplot(sub_img, "", cols=len(sub_images) + 2, plot_number=i)

    pyplot.show()

    while (cv2.waitKey(0) % 256) != 27:
        pass

    sys.exit() # ESC


if __name__ == '__main__':
    # main()
    while True:
        build_study_images()
