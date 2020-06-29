import cv2
import glob
from pathlib import Path
from fractions import Fraction
from my_modules import calc, img_proc, convert
import config
import my_modules.settings as SETTINGS


if __name__ == '__main__':
    # 指定ディレクトリ内のpng画像パスをfor文で回す
    files = glob.glob(config.input_imgs_directory +
                      config.target_animal_regexp + '.png')
    for file in files:
        # ファイル名から動物名を取得
        animal = Path(file).stem
        # OpenCVで画像読み込み
        orig_img = cv2.imread(
            config.input_imgs_directory + animal + '.png', -1)
        orig_h, orig_w = orig_img.shape[:2]
        rotation = Fraction()
        # 1回転するまでループする
        while (rotation < SETTINGS.ONE_ROTATION):
            # αチャンネルの処理
            a_channel = orig_img[:, :, 3]
            rotated_a_channel = img_proc.rotate_img_or_channel(
                a_channel, rotation)
            rotated_h, rotated_w = rotated_a_channel.shape[:2]
            centroid = calc.centroid(
                rotated_a_channel)
            bottom_x, bottom_y, bottom_length = calc.bottom(
                rotated_a_channel)
            direction = calc.direoction_of_fall(orig_h, orig_w,
                                                animal,
                                                bottom_length,
                                                bottom_x, centroid[0])
            dtb_rotation = convert.to_dtb_rot(rotation)
            # BGRチャンネルの処理
            bgr_channel = orig_img[:, :, :3]
            rotated_bgr_channel = img_proc.rotate_img_or_channel(
                bgr_channel, rotation)
            circle_added_bgr_channel = cv2.circle(
                rotated_bgr_channel,
                (int(centroid[0]), int(centroid[1])),
                int(SETTINGS.MARKER_SIZE*(rotated_h+rotated_w)),
                (0, 0, 255), -1)
            # αチャンネルとBGRチャンネルをマージする
            result_img = cv2.merge(
                (circle_added_bgr_channel, rotated_a_channel))
            # リザルト画像を保存する
            if direction == 'upright':
                cv2.imwrite(config.output_imgs_directory+animal+'_' +
                            dtb_rotation+'.png',
                            result_img)
            # 1fだけ回転させる
            rotation += Fraction(1, SETTINGS.FRAME)
