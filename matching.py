import cv2
import glob
from pathlib import Path
from modules import img_proc, akaze
import config
import imutils
import modules.settings as SETTINGS


if __name__ == '__main__':
    query_img_path = config.imgs_directory + 'input/captures/test1.png'
    query_img = cv2.imread(query_img_path)
    result_img = query_img.copy()
    query_img = imutils.resize(query_img, width=SETTINGS.QUERY_WIDTH)
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
            templ_img, width=int(SETTINGS.WIDTH_ANIMALS[animal]))
        templ_a_channel = templ_img[:, :, 3]
        templ_img = img_proc.add_dtb_bg(templ_img)
        query_mask = img_proc.thresh(templ_a_channel)
        templ_h, templ_w = templ_img.shape[:2]
        M_templ2query_1, corner_pts_1, min_distance_1 = akaze.matching(
            templ_img, query_img, query_mask)
        print(min_distance_1)
        if (min_distance_1 <= 33):
            # クエリーのテンプレコーナーを折れ線で囲う
            cv2.polylines(
                result_img, [corner_pts_1], True, (0, 255, 255), thickness=4)
            search_area_mask_1 = cv2.warpPerspective(
                templ_a_channel, M_templ2query_1, (query_w, query_h))
            search_area_mask_1 = img_proc.thresh_inv(search_area_mask_1)
            M_templ2query_2, corner_pts_2, min_distance_2 = akaze.matching(
                templ_img, query_img,
                query_mask, search_area_mask_1)
            print(min_distance_2)
            while (min_distance_2 <= 15):
                # クエリーのテンプレコーナーを折れ線で囲う
                cv2.polylines(
                    result_img, [corner_pts_2], True, (0, 255, 255), thickness=4)
                search_area_mask_2 = cv2.warpPerspective(
                    templ_a_channel, M_templ2query_2, (query_w, query_h))
                search_area_mask_2 = img_proc.thresh_inv(
                    search_area_mask_2)
                search_area_mask = cv2.bitwise_and(
                    search_area_mask_1, search_area_mask_2)
                M_templ2query_2, corner_pts_2, min_distance_2 = akaze.matching(
                    templ_img, query_img,
                    query_mask, search_area_mask)
                search_area_mask_1 = search_area_mask
            img_proc.show(result_img)
        '''
        animal_bottom_vector = (
            pts[0][2][0]-pts[0][1][0], -(pts[0][2][1]-pts[0][1][1]))
        animal_angle_rad = math.atan2(
            animal_bottom_vector[1], animal_bottom_vector[0])
        animal_size = np.linalg.norm(animal_bottom_vector)
        calc.centroid(a_channel)
        '''
