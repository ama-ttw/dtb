import cv2
import numpy as np


def matching(templ_img, query_img,
             templ_mask=None, query_mask=None, return_corners=True):
    '''
    if (templ_mask is not None):
        img_proc.show(templ_mask)
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
    # 位置計算
    src_pts = np.float32([templ_kp[m.trainIdx].pt for m in matches[:13]])
    dst_pts = np.float32([query_kp[m.queryIdx].pt for m in matches[:13]])
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    if return_corners:
        # 画像４隅の角座標を取得
        th = templ_img.shape[0]
        tw = templ_img.shape[1]
        pts = np.array(
            [[[0, 0], [0, th-1], [tw-1, th-1], [tw-1, 0]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pts, M)
        return M, np.int32(dst), min_distance
    return M, min_distance
