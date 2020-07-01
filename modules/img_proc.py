import numpy as np
import cv2
import modules.settings as SETTINGS
import imutils


def add_dtb_bg(templ_img):
    templ_w, templ_h = templ_img.shape[:2]
    bg_img = np.zeros((templ_w, templ_h, 3))
    bg_img += SETTINGS.RGB_BGCOLOR[::-1]  # RGBで色指定
    mask = templ_img[:, :, 3]  # アルファチャンネルだけ抜き出す。
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 3色分に増やす。
    mask = mask / 255  # 0-255だと使い勝手が悪いので、0.0-1.0に変更。
    templ_img = templ_img[:, :, :3]  # アルファチャンネルは取り出しちゃったのでもういらない。
    bg_img[0:templ_w:, 0:templ_h] *= 1 - mask  # 透過率に応じて元の画像を暗くする。
    bg_img[0:templ_w:, 0:templ_h] += templ_img * \
        mask  # 貼り付ける方の画像に透過率をかけて加算。
    bg_added_img = bg_img.astype('uint8')
    return bg_added_img


def rotate(img, i):
    h, w = img.shape[:2]
    # 回転角の指定
    angle = 360/SETTINGS.ONE_ROTATION*(-i)
    angle_rad = angle/180.0*np.pi
    w_rot = int(np.round(h*np.absolute(np.sin(angle_rad)) +
                         w*np.absolute(np.cos(angle_rad))))
    h_rot = int(np.round(h*np.absolute(np.cos(angle_rad)) +
                         w*np.absolute(np.sin(angle_rad))))
    size_rot = (w_rot, h_rot)
    # 元画像の中心を軸に回転する
    center = (w/2, h/2)
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    # 平行移動を加える (rotation + translation)
    affine_matrix = rotation_matrix.copy()
    affine_matrix[0][2] = affine_matrix[0][2] - w/2 + w_rot/2
    affine_matrix[1][2] = affine_matrix[1][2] - h/2 + h_rot/2
    img_rot = cv2.warpAffine(
        img, affine_matrix, size_rot)
    return img_rot


def show(img):
    resized_img = imutils.resize(img, height=SETTINGS.SHOW_HEIGHT)
    cv2.imshow('image', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def thresh_inv(a_channel):
    _, a_channel = cv2.threshold(
        a_channel, SETTINGS.THRESH_A_CHANNEL, 255, cv2.THRESH_BINARY_INV)
    return a_channel


def thresh(a_channel):
    _, a_channel = cv2.threshold(
        a_channel, SETTINGS.THRESH_A_CHANNEL, 255, cv2.THRESH_BINARY)
    return a_channel


def draw_polyline(img, pts):
    cv2.polylines(img, [pts], True, (0, 0, 255), thickness=4)


def draw_circle(img, pt):
    cv2.circle(img, pt, 5, (0, 255, 0), thickness=-1)
