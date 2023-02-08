import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_map
from ssd import SSD

if __name__ == "__main__":

    # 1: 生成test.txt列表中检测数据，2：生成标签数据 3：开始计算map
    map_mode = 1

    # MINOVERLAP用于指定想要获得的mAP0.x，
    MINOVERLAP = 0.5
    confidence = 0.02
    nms_iou = 0.5
    score_threhold = 0.5

    VOCdevkit_path = 'VOCdevkit'
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/ztest.txt")).read().strip().split()

    # 结果输出的文件夹，默认为map_out
    map_out_path = 'map_out'
    os.makedirs(os.path.join(map_out_path, 'ground-truth'), exist_ok=True)
    os.makedirs(os.path.join(map_out_path, 'detection-results'), exist_ok=True)
    os.makedirs(os.path.join(map_out_path, 'images-optional'), exist_ok=True)

    classes_path = 'conf/voc_classes.txt'
    class_names, _ = get_classes(classes_path)

    if map_mode == 1:
        print("Load model.")
        ssd = SSD(confidence=confidence, nms_iou=nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            ssd.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    if map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/" + image_id + ".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') != None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, score_threhold=score_threhold, path=map_out_path)
        print("Get map done.")
