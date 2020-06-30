import cv2
import numpy as np
import modules.settings as SETTINGS


def bottom(a_channel):
    _, a_channel = cv2.threshold(
        a_channel, SETTINGS.THRESH_A_CHANNEL, 255, cv2.THRESH_BINARY)
    whites = np.squeeze(np.dstack(np.where(a_channel == 255)))
    bottom_indexs = np.where(whites[:, 0] == max(whites[:, 0]))
    bottom_length = len(bottom_indexs[0])
    bottom_coord = np.mean(whites[bottom_indexs], axis=0)
    return bottom_coord[1], bottom_coord[0], bottom_length


def centroid(a_channel):
    _, a_channel = cv2.threshold(
        a_channel, SETTINGS.THRESH_A_CHANNEL, 255, cv2.THRESH_BINARY)
    whites = np.squeeze(np.dstack(np.where(a_channel == 255)))
    cy, cx = np.mean(whites, axis=0)
    return cx, cy


def direoction_of_fall(orig_h, orig_w,
                       animal, bottom_length,
                       bottom_x, centroid_x):
    orig_size = orig_h+orig_w
    real_h = orig_h*SETTINGS.WIDTH_ANIMALS[animal]/orig_w
    real_size = real_h+SETTINGS.WIDTH_ANIMALS[animal]
    dist_centroid_bottom = (bottom_x-centroid_x)*(real_size)/(orig_size)
    if (abs(dist_centroid_bottom) <= SETTINGS.UPRIGHT_DISTANCE) and\
        (bottom_length*(real_size)/(orig_size) >= SETTINGS.UPRIGHT_LENGTH) or\
            (dist_centroid_bottom == 0):
        return 'upright'
    elif (dist_centroid_bottom < 0):
        return 'right'
    elif (dist_centroid_bottom > 0):
        return 'left'
