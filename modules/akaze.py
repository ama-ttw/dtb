import cv2
import numpy as np


def searchPosition(templ_img, query_img):
    # 画像をグレースケールに変換
    gray1 = cv2.cvtColor(templ_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    # A-KAZE検出器の生成
    detector = cv2.AKAZE_create()
    templ_kp, templ_des = detector.detectAndCompute(gray1, None)
    # 特徴量の検出と特徴量ベクトルの計算
    query_kp, query_des = detector.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # 特徴量ベクトル同士をマッチング
    matches = bf.match(query_des, templ_des)
    # 特徴量をマッチング状況に応じてソート
    matches = sorted(matches, key=lambda x: x.distance)
    # 位置計算
    src_pts = np.float32([templ_kp[m.trainIdx].pt for m in matches[:13]])
    dst_pts = np.float32([query_kp[m.queryIdx].pt for m in matches[:13]])
    Mx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    # 画像４隅の角座標を取得
    th = templ_img.shape[0]
    tw = templ_img.shape[1]
    pts = np.array(
        [[[0, 0], [0, th-1], [tw-1, th-1], [tw-1, 0]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pts, Mx)
    return Mx, np.int32(dst)
