import cv2
import glob
from pathlib import Path
from modules import img_proc, akaze, calc, convert
import config
import imutils
import modules.settings as SETTINGS
import numpy as np


if __name__ == '__main__':
    query_img_path = config.query_img_path
    query_img = cv2.imread(query_img_path)
    query_img = imutils.resize(query_img, width=int(
        SETTINGS.QUERY_WIDTH*SETTINGS.EXPAND))
    result_img = query_img.copy()
    query_h, query_w = query_img.shape[:2]
    # 指定ディレクトリ内のpng画像パスをfor文で回す
    files = glob.glob(config.input_animals_directory +
                      config.target_animal_regexp + '.png')
    for file in files:
        # ファイル名から動物名を取得
        animal = Path(file).stem
        templ_img_path = config.input_animals_directory + animal + '.png'
        templ_img = cv2.imread(templ_img_path, -1)
        templ_img = imutils.resize(
            templ_img,
            width=int(SETTINGS.WIDTH_ANIMALS[animal]*SETTINGS.EXPAND))
        templ_a_channel = templ_img[:, :, 3]
        templ_img = img_proc.add_dtb_bg(templ_img)
        templ_mask = img_proc.thresh(templ_a_channel)
        templ_h, templ_w = templ_img.shape[:2]
        templ_centroid_x, templ_centroid_y = calc.centroid(templ_a_channel)
        norms = []
        count_animals = 0
        # 全範囲を探索する
        M_templ2query, corner_pts, min_distance = akaze.matching(
            templ_img, query_img,
            templ_mask)

        firstLoop = True
        while (min_distance <= SETTINGS.MIN_DISTANCE):
            '''
            query_centroid = convert.to_query_position(
                templ_centroid_x, templ_centroid_y, M_templ2query)
            img_proc.draw_circle(
                result_img, (query_centroid[0], query_centroid[1]))
                '''
            img_proc.draw_polyline(result_img, corner_pts)
            animal_bottom_vector = (corner_pts[0][2][0]-corner_pts[0][1][0],
                                    corner_pts[0][2][1]-corner_pts[0][1][1])
            norm = np.linalg.norm(animal_bottom_vector)
            norms.append(norm/SETTINGS.EXPAND)
            search_area_mask = cv2.warpPerspective(
                templ_a_channel, M_templ2query, (query_w, query_h))
            search_area_mask = img_proc.thresh_inv(search_area_mask)
            if firstLoop:
                search_area_mask_old = search_area_mask
                firstLoop = False
            else:
                search_area_mask = cv2.bitwise_and(
                    search_area_mask, search_area_mask_old)
            # 探索範囲を探索する
            M_templ2query, corner_pts, min_distance = akaze.matching(
                templ_img, query_img,
                templ_mask, search_area_mask)
            search_area_mask_old = search_area_mask
            count_animals += 1
        if (count_animals != 0):
            print(animal)
            print("norm:\t", np.mean(norms))
    img_proc.show(result_img)
