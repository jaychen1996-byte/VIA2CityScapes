import os
import glob
import shutil
import json
from skimage import io, draw
import numpy as np

"""

通过via生成标注文件 有最外圈bg 内圈sg 和目标区域error 得到包含住目标的mask

"""


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


if __name__ == '__main__':
    images_path = "images"
    # json_path = glob.glob(os.path.join(images_path, "*.json"))
    json_path = ["images/2.json"]
    saved_path = "output"
    img_suffix = '_leftImg8bit.png'
    seg_map_suffix = '_gtFine_labelTrainIds.png'

    create_path(saved_path)

    for p in ["leftImg8bit/train", "gtFine/train"]:
        path = os.path.join(saved_path, p)
        create_path(path)

    anno = json.load(open(json_path[0], "rb"))
    anno = list(anno.values())

    for index, an in enumerate(anno):
        cur_image = io.imread(os.path.join(images_path, an['filename']))  # h,w
        io.imsave(os.path.join(saved_path, "leftImg8bit/train", str(index) + img_suffix), cur_image)
        mask = np.full([cur_image.shape[0], cur_image.shape[1]], 2, dtype=np.uint8)
        for reg in an['regions']:
            if reg['region_attributes']['name'] == "bg":
                all_x, all_y = np.array(reg['shape_attributes']['all_points_x']), np.array(
                    reg['shape_attributes']['all_points_y'])
                rr, cc = draw.polygon(all_y, all_x)
                mask[rr, cc] = 1
        for reg in an['regions']:
            if reg['region_attributes']['name'] == "sg":
                all_x, all_y = np.array(reg['shape_attributes']['all_points_x']), np.array(
                    reg['shape_attributes']['all_points_y'])
                rr, cc = draw.polygon(all_y, all_x)
                mask[rr, cc] = 2
        for reg in an['regions']:
            if reg['region_attributes']['name'] == "error":
                all_x, all_y = np.array(reg['shape_attributes']['all_points_x']), np.array(
                    reg['shape_attributes']['all_points_y'])
                rr, cc = draw.polygon(all_y, all_x)
                mask[rr, cc] = 0
        io.imsave(os.path.join(saved_path, "gtFine/train", str(index) + seg_map_suffix), mask)
        # io.imshow(mask)
        # io.show()
