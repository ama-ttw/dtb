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
        templ_img = img_proc.add_dtb_bg(templ_img)
        templ_h, templ_w = templ_img.shape[:2]
        M_templ2query, corner_pts = akaze.matching(
            templ_img, query_img, return_corners=True)
        query_img = cv2.polylines(
            query_img, [corner_pts], True, (0, 255, 255), thickness=4)
        templ_a_channel = templ_img[:, :, 3]
        search_area_mask = cv2.warpPerspective(
            templ_a_channel, M_templ2query, (query_w, query_h))
        search_area_mask = img_proc.thresh_inv(search_area_mask)
        '''
        animal_bottom_vector = (
            pts[0][2][0]-pts[0][1][0], -(pts[0][2][1]-pts[0][1][1]))
        animal_angle_rad = math.atan2(
            animal_bottom_vector[1], animal_bottom_vector[0])
        animal_size = np.linalg.norm(animal_bottom_vector)
        calc.centroid(a_channel)
        '''
