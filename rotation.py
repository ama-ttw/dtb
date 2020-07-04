import cv2
import glob
from pathlib import Path
from fractions import Fraction
from modules import calc, img_proc, convert
import config
import modules.settings as SETTINGS


if __name__ == '__main__':
    # 指定ディレクトリ内のpng画像パスをfor文で回す
    files = glob.glob(config.input_animals_directory +
                      config.target_animal_regexp + '.png')
    direction_rotation_animals = {}
    for file in files:
        # ファイル名から動物名を取得
        animal = Path(file).stem
        direction_rotation_animals[animal] = {}
        # OpenCVで画像読み込み
        orig_img = cv2.imread(
            config.input_animals_directory + animal + '.png', -1)
        orig_h, orig_w = orig_img.shape[:2]
        rotation = Fraction()
        # 1回転するまでループする
        while (rotation < SETTINGS.ONE_ROTATION):
            # αチャンネルの処理
            a_channel = img_proc.extract_a_channel(orig_img)
            if (animal == 'Alpaca'):
                no_collision = a_channel[156:, :]
                no_collision = img_proc.zerofill(no_collision)
                calc_a_channel = cv2.vconcat(
                    [a_channel[:156, :], no_collision])
            if (animal == 'Polar_bear'):
                no_collision = a_channel[223:, :]
                no_collision = img_proc.zerofill(no_collision)
                calc_a_channel = cv2.vconcat(
                    [a_channel[:223, :], no_collision])
            else:
                calc_a_channel = a_channel
            rotated_a_channel = img_proc.rotate(
                a_channel, rotation)
            rotated_calc_a_channel = img_proc.rotate(
                calc_a_channel, rotation)
            rotated_h, rotated_w = rotated_a_channel.shape[:2]
            centroid = calc.centroid(
                rotated_calc_a_channel)
            bottom_x, bottom_y, bottom_length = calc.bottom(
                rotated_calc_a_channel)
            direction = calc.direoction_of_fall(orig_h, orig_w,
                                                animal,
                                                bottom_length,
                                                bottom_x, centroid[0])
            dtb_rotation = convert.to_dtb_rot(rotation)
            direction_rotation_animals[animal][dtb_rotation] = direction
            # BGRチャンネルの処理
            bgr_channel = img_proc.extract_bgr_channel(orig_img)
            rotated_bgr_channel = img_proc.rotate(
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
                cv2.imwrite(config.output_imgs_directory+animal+'_'+dtb_rotation+'.png',
                            result_img)
            # 1fだけ回転させる
            rotation += Fraction(1, SETTINGS.FRAME)
    print(direction_rotation_animals)
