import cv2
import numpy as np
import modules.settings as SETTINGS
from modules import img_proc


def bottom(a_channel):
    a_channel = img_proc.thresh(a_channel)
    whites = np.squeeze(np.dstack(np.where(a_channel == 255)))
    bottom_indexs = np.where(whites[:, 0] == max(whites[:, 0]))
    bottom_length = len(bottom_indexs[0])
    bottom_coord = np.mean(whites[bottom_indexs], axis=0)
    return bottom_coord[1], bottom_coord[0], bottom_length


def centroid(a_channel):
    a_channel = img_proc.thresh(a_channel)
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


def matching_pts(templ_img, query_img,
                 templ_mask=None, query_mask=None):
    '''
    if (query_mask is not None):
        img_proc.show_resized(query_mask)
    '''
    # 画像をグレースケールに変換
    gray1 = cv2.cvtColor(templ_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    # A-KAZE検出器の生成
    detector = cv2.AKAZE_create()
    templ_kp, templ_des = detector.detectAndCompute(gray1, templ_mask)

    # 特徴量の検出と特徴量ベクトルの計算
    query_kp, query_des = detector.detectAndCompute(gray2, query_mask)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # 特徴量ベクトル同士をマッチング
    matches = bf.match(query_des, templ_des)
    # 特徴量をマッチング状況に応じてソート
    matches = sorted(matches, key=lambda x: x.distance)
    min_distance = matches[0].distance
    matches_len = 0
    for match in matches:
        distance = match.distance
        if (distance > SETTINGS.MAX_DISTANCE):
            break
        # MAX_DISTANCE以下ならカウント
        matches_len += 1
    #print(min_distance, matches_len)
    return min_distance, templ_kp, query_kp, templ_img, matches, matches_len


def matrix(templ_kp, query_kp, templ_img,
           matches, matches_len, query_img, result_img, return_corners=False):
    # 位置計算
    src_pts = np.float32(
        [templ_kp[m.trainIdx].pt for m in matches[:matches_len]])
    dst_pts = np.float32(
        [query_kp[m.queryIdx].pt for m in matches[:matches_len]])
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    if return_corners:
        # 画像４隅の角座標を取得
        th = templ_img.shape[0]
        tw = templ_img.shape[1]
        pts = np.array(
            [[[0, 0], [0, th-1], [tw-1, th-1], [tw-1, 0]]],
            dtype=np.float32)
        dst = cv2.perspectiveTransform(pts, M)
        '''
        result_img = cv2.drawMatches(query_img,
                                     query_kp, templ_img, templ_kp,
                                     matches[:matches_len],
                                     result_img, flags=2)
        img_proc.show_resized(result_img)
        '''
        return M, np.int32(dst)
    # img_proc.show(result_img)
    else:
        return M


def matching_roi_area():
    # 探索範囲を探索する
    pass
