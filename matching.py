import cv2
import glob
from pathlib import Path
from modules import img_proc, akaze, calc
import config
import imutils
import modules.settings as SETTINGS
import math
import numpy as np

if __name__ == '__main__':
    capture_img_path = config.imgs_directory + 'input/captures/3.png'
    capture_img = cv2.imread(capture_img_path)
    capture_img = imutils.resize(capture_img, width=1080)
    # 指定ディレクトリ内のpng画像パスをfor文で回す
    files = glob.glob(config.input_animals_directory +
                      config.target_animal_regexp + '.png')
    for file in files:
        # ファイル名から動物名を取得
        animal = Path(file).stem
        target_img_path = config.input_animals_directory + animal + '.png'
        target_img = cv2.imread(target_img_path, -1)
        target_img = imutils.resize(
            target_img, width=int(SETTINGS.WIDTH_ANIMALS[animal]))
        bg_added_target_img = img_proc.add_dtb_bg(target_img)
        bg_h, bg_w = bg_added_target_img.shape[:2]
        mx, pts = akaze.searchPosition(bg_added_target_img, capture_img)
        capture_img = cv2.polylines(
            capture_img, [pts], True, (0, 255, 255), thickness=4)
        distance = ((pts[0][2][0]-pts[0][1][0])**2 +
                    (pts[0][2][1]-pts[0][1][1])**2)**(1/2)
        # テンプレート画像の重心座標をクエリー重心座標にホモグラフィ変換し, 値を格納する
        a_channel = target_img[:, :, 3]
        cx, cy = calc.centroid(a_channel)
        gv_pt = cv2.perspectiveTransform(
            np.float32([cx, cy]).reshape(-1, 1, 2), mx)
        gv_pt = gv_pt.flatten()
        query_img = cv2.circle(
            capture_img, (gv_pt[0], gv_pt[1]), 10, (255, 0, 255), thickness=-1)
        vector = (pts[0][2][0]-pts[0][1][0], -(pts[0][2][1]-pts[0][1][1]))
        rad = math.atan2(vector[1], vector[0])
        print(math.degrees(rad))
        img_proc.show(capture_img)
