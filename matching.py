import cv2
import glob
from pathlib import Path
from modules import img_proc, calc, convert
import config
import imutils
import modules.settings as SETTINGS

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
    firstAnimal = True
    for file in files:
        # ファイル名から動物名を取得
        animal = Path(file).stem
        templ_img_path = config.input_animals_directory + animal + '.png'
        templ_img = cv2.imread(templ_img_path, -1)
        templ_img = imutils.resize(
            templ_img,
            width=int(SETTINGS.WIDTH_ANIMALS[animal]*SETTINGS.EXPAND))
        templ_a_channel = img_proc.extract_a_channel(templ_img)
        templ_img = img_proc.add_dtb_bg(templ_img)
        templ_mask = img_proc.thresh(templ_a_channel)
        templ_h, templ_w = templ_img.shape[:2]
        if (animal == 'Alpaca'):
            no_collision = templ_a_channel[156:, :]
            no_collision = img_proc.zerofill(no_collision)
            calc_a_channel = cv2.vconcat(
                [templ_a_channel[:156, :], no_collision])
        if (animal == 'Polar_bear'):
            no_collision = templ_a_channel[223:, :]
            no_collision = img_proc.zerofill(no_collision)
            calc_a_channel = cv2.vconcat(
                [templ_a_channel[:223, :], no_collision])
        else:
            calc_a_channel = templ_a_channel
        templ_centroid_x, templ_centroid_y = calc.centroid(calc_a_channel)
        count_animals = 0
        if firstAnimal:
            # 全範囲を探索する
            min_distance, templ_kp, query_kp, templ_img, \
                matches, matches_len = calc.matching_pts(
                    templ_img, query_img,
                    templ_mask)
            # はじめの動物を見つけられなかったら, 動物をきりかえる
            if not(((min_distance <= SETTINGS.MIN_DISTANCE)and
                    (matches_len >= SETTINGS.MATCHES_LEN))):
                continue
            # はじめの動物が見つかったら
            else:
                count_animals += 1
                # 変換マトリクスを演算し、コーナーと重心を描写する
                M_templ2query, corner_pts = calc.matrix(
                    templ_kp, query_kp, templ_img, matches, matches_len,
                    query_img, result_img,
                    return_corners=True)
                query_centroid = convert.to_query_position(
                    templ_centroid_x, templ_centroid_y, M_templ2query)
                real_h = templ_h*SETTINGS.WIDTH_ANIMALS[animal]/templ_w
                real_size = real_h+SETTINGS.WIDTH_ANIMALS[animal]
                img_proc.draw_circle(
                    result_img, (query_centroid[0], query_centroid[1]),
                    int(SETTINGS.MARKER_SIZE*real_size))
                img_proc.draw_polyline(result_img, corner_pts)
                # 探索範囲のマスクを作成
                search_area_mask = cv2.warpPerspective(
                    templ_a_channel, M_templ2query, (query_w, query_h))
                search_area_mask = img_proc.thresh_inv(search_area_mask)
                firstAnimal = False
                # 探索範囲を探索する
                min_distance, templ_kp, query_kp, templ_img,\
                    matches, matches_len = calc.matching_pts(
                        templ_img, query_img,
                        templ_mask, search_area_mask)
                search_area_mask_old = search_area_mask
                # 動物が見つかったら
                while ((min_distance <= SETTINGS.MIN_DISTANCE)and
                       (matches_len >= SETTINGS.MATCHES_LEN)):
                    count_animals += 1
                    # 変換マトリクスを演算し、コーナーと重心を描写する
                    M_templ2query, corner_pts = calc.matrix(
                        templ_kp, query_kp, templ_img, matches, matches_len,
                        query_img, result_img,
                        return_corners=True)
                    query_centroid = convert.to_query_position(
                        templ_centroid_x, templ_centroid_y, M_templ2query)
                    real_h = templ_h*SETTINGS.WIDTH_ANIMALS[animal]/templ_w
                    real_size = real_h+SETTINGS.WIDTH_ANIMALS[animal]
                    img_proc.draw_circle(
                        result_img, (query_centroid[0], query_centroid[1]),
                        int(SETTINGS.MARKER_SIZE*real_size))
                    img_proc.draw_polyline(result_img, corner_pts)
                    search_area_mask = cv2.warpPerspective(
                        templ_a_channel, M_templ2query, (query_w, query_h))
                    # マスクを作成
                    search_area_mask = img_proc.thresh_inv(search_area_mask)
                    search_area_mask = cv2.bitwise_and(
                        search_area_mask, search_area_mask_old)
                    search_area_mask_old = search_area_mask
                    # 探索範囲を探索する
                    min_distance, templ_kp, query_kp, templ_img,\
                        matches, matches_len = calc.matching_pts(
                            templ_img, query_img,
                            templ_mask, search_area_mask)
        else:
            # 探索範囲を探索する
            min_distance, templ_kp, query_kp, templ_img,\
                matches, matches_len = calc.matching_pts(
                    templ_img, query_img,
                    templ_mask, search_area_mask)
            search_area_mask_old = search_area_mask
            # 動物が見つかったら
            while ((min_distance <= SETTINGS.MIN_DISTANCE)and
                   (matches_len >= SETTINGS.MATCHES_LEN)):
                count_animals += 1
                # 変換マトリクスを演算し、コーナーと重心を描写する
                M_templ2query, corner_pts = calc.matrix(
                    templ_kp, query_kp, templ_img, matches, matches_len,
                    query_img, result_img,
                    return_corners=True)
                query_centroid = convert.to_query_position(
                    templ_centroid_x, templ_centroid_y, M_templ2query)
                real_h = templ_h*SETTINGS.WIDTH_ANIMALS[animal]/templ_w
                real_size = real_h+SETTINGS.WIDTH_ANIMALS[animal]
                img_proc.draw_circle(
                    result_img, (query_centroid[0], query_centroid[1]),
                    int(SETTINGS.MARKER_SIZE*real_size))
                img_proc.draw_polyline(result_img, corner_pts)
                search_area_mask = cv2.warpPerspective(
                    templ_a_channel, M_templ2query, (query_w, query_h))
                # マスクを作成
                search_area_mask = img_proc.thresh_inv(search_area_mask)
                search_area_mask = cv2.bitwise_and(
                    search_area_mask, search_area_mask_old)
                search_area_mask_old = search_area_mask
                # 探索範囲を探索する
                min_distance, templ_kp, query_kp, templ_img,\
                    matches, matches_len = calc.matching_pts(
                        templ_img, query_img,
                        templ_mask, search_area_mask)
        if (count_animals != 0):
            print(animal, count_animals)
    img_proc.show_resized(result_img)
    cv2.imwrite(config.output_imgs_directory+'result.png', result_img)

# norms = []
# animal_bottom_vector = (corner_pts[0][2][0]-corner_pts[0][1][0],
#                        corner_pts[0][2][1]-corner_pts[0][1][1])
# norm = np.linalg.norm(animal_bottom_vector)
# norms.append(norm/SETTINGS.EXPAND)
# norm = np.linalg.norm(animal_bottom_vector)
# norms.append(norm/SETTINGS.EXPAND)
# print("norm:\t", np.mean(norms))
