import cv2
import numpy as np
import my_modules.settings as SETTINGS


def searchPosition(templ_img, query_img,
                   good_match_rate=SETTINGS.GOOD_MATCH_RATE):
    # A-KAZE検出器の生成
    detector = cv2.AKAZE_create()
    templ_kp, templ_des = detector.detectAndCompute(templ_img, None)

    # 特徴量の検出と特徴量ベクトルの計算
    query_kp, query_des = detector.detectAndCompute(query_img, None)

    # マッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 特徴量ベクトル同士をマッチング
    matches = bf.match(query_des, templ_des)

    # 特徴量をマッチング状況に応じてソート
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:int(len(matches) * good_match_rate)]

    # 位置計算
    src_pts = np.float32([templ_kp[m.trainIdx].pt for m in good])
    dst_pts = np.float32([query_kp[m.queryIdx].pt for m in good])
    Mx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    query_img = cv2.drawMatches(
        query_img, query_kp, templ_img, templ_kp, good, None, flags=2)
    cv2.imshow('image', query_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 画像４隅の角座標を取得
    th = templ_img.shape[0]
    tw = templ_img.shape[1]
    pts = np.array(
        [[[0, 0], [0, th-1], [tw-1, th-1], [tw-1, 0]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(pts, Mx)
    return Mx, np.int32(dst)
