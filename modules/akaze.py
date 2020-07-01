import cv2
import numpy as np
import modules.settings as SETTINGS
from modules import img_proc
from matplotlib import pyplot as plt


def matching(templ_img, query_img,
             templ_mask=None, query_mask=None, return_corners=True):
    '''
    if (query_mask is not None):
        img_proc.show(query_mask)
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
    '''
    print("min_dist\t", min_distance)
    print("len_templ\t", len(templ_kp))
    '''
    if (len(templ_kp)/(1/3) >= 15):
        use_matchs_count = int(len(templ_kp)/(1/3))
    else:
        print("this animal doesn't have enough feature")
        use_matchs_count = len(templ_kp)
    good = matches[:use_matchs_count]
    # 位置計算
    src_pts = np.float32([templ_kp[m.trainIdx].pt for m in good])
    dst_pts = np.float32([query_kp[m.queryIdx].pt for m in good])
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
                                     query_kp, templ_img, templ_kp, good,
                                     result_img, flags=2)
                                     '''
        # img_proc.show(result_img)
        return M, np.int32(dst), min_distance
    return M, min_distance
