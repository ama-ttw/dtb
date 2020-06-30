from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

imgs_directory = '../images/'
input_animals_directory = imgs_directory+'input/animals/'
templ_img = cv2.imread(input_animals_directory+'Elphant.png', 0)  # queryImage
query_img = cv2.imread(
    imgs_directory+'input/captures/test1.png', 0)  # trainImage

'''
orb = cv2.ORB_create(10000, 1.2, nlevels=8, edgeThreshold=5)
# find the keypoints and descriptors with ORB
templ_kp, templ_des = orb.detectAndCompute(img1, None)
query_kp, query_des = orb.detectAndCompute(img2, None)
'''
# A-KAZE検出器の生成
detector = cv2.AKAZE_create()
templ_kp, templ_des = detector.detectAndCompute(templ_img, None)
# 特徴量の検出と特徴量ベクトルの計算
query_kp, query_des = detector.detectAndCompute(query_img, None)


x = np.array([query_kp[0].pt])

for i in range(len(query_kp)):
    x = np.append(x, [query_kp[i].pt], axis=0)

x = x[1:len(x)]

bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
ms.fit(x)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

s = [None] * n_clusters_
for i in range(n_clusters_):
    label = ms.labels_
    d, = np.where(label == i)
    print(d.__len__())
    s[i] = list(query_kp[xx] for xx in d)

query_des_ = query_des

for i in range(n_clusters_):

    query_kp = s[i]
    label = ms.labels_
    d, = np.where(label == i)
    query_des = query_des_[d, ]

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    templ_des = np.float32(templ_des)
    query_des = np.float32(query_des)

    matches = flann.knnMatch(templ_des, query_des, 2)

    # store allabel the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > 3:
        src_pts = np.float32(
            [templ_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [query_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2)

        if M is None:
            print("No Homography")
        else:
            matchesMask = mask.ravel().tolist()

            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                              [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(
                img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)

            img3 = cv2.drawMatches(img1, templ_kp, img2, query_kp,
                                   good, None, **draw_params)

            plt.imshow(img3, 'gray'), plt.show()

    else:
        print("Not enough matches are found - %d/%d" %
              (len(good), MIN_MATCH_COUNT))
        matchesMask = None
