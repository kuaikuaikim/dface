# coding: utf-8
import os
import cv2
import numpy as np
import random
import sys
import numpy.random as npr
import argparse
import dface.config as config
import dface.core.utils as utils


def gen_data(anno_file, data_dir, prefix):


    size = 48
    image_id = 0

    landmark_imgs_save_dir = os.path.join(data_dir,"48/landmark")
    if not os.path.exists(landmark_imgs_save_dir):
        os.makedirs(landmark_imgs_save_dir)

    anno_dir = config.ANNO_STORE_DIR
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)

    landmark_anno_filename = config.ONET_LANDMARK_ANNO_FILENAME
    save_landmark_anno = os.path.join(anno_dir,landmark_anno_filename)

    f = open(save_landmark_anno, 'w')
    # dstdir = "train_landmark_few"

    with open(anno_file, 'r') as f2:
        annotations = f2.readlines()

    num = len(annotations)
    print("%d total images" % num)

    l_idx =0
    idx = 0
    # image_path bbox landmark(5*2)
    for annotation in annotations:
        # print imgPath

        annotation = annotation.strip().split(' ')

        assert len(annotation)==15,"each line should have 15 element"

        im_path = os.path.join(prefix,annotation[0].replace("\\", "/"))

        gt_box = map(float, annotation[1:5])
        # gt_box = [gt_box[0], gt_box[2], gt_box[1], gt_box[3]]


        gt_box = np.array(gt_box, dtype=np.int32)

        landmark = map(float, annotation[5:])
        landmark = np.array(landmark, dtype=np.float)

        img = cv2.imread(im_path)
        assert (img is not None)

        height, width, channel = img.shape
        # crop_face = img[gt_box[1]:gt_box[3]+1, gt_box[0]:gt_box[2]+1]
        # crop_face = cv2.resize(crop_face,(size,size))

        idx = idx + 1
        if idx % 100 == 0:
            print("%d images done, landmark images: %d"%(idx,l_idx))

        x1, y1, x2, y2 = gt_box

        # gt's width
        w = x2 - x1 + 1
        # gt's height
        h = y2 - y1 + 1
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue
        # random shift
        for i in range(10):
            bbox_size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)
            nx1 = max(x1 + w / 2 - bbox_size / 2 + delta_x, 0)
            ny1 = max(y1 + h / 2 - bbox_size / 2 + delta_y, 0)

            nx2 = nx1 + bbox_size
            ny2 = ny1 + bbox_size
            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]
            resized_im = cv2.resize(cropped_im, (size, size),interpolation=cv2.INTER_LINEAR)

            offset_x1 = (x1 - nx1) / float(bbox_size)
            offset_y1 = (y1 - ny1) / float(bbox_size)
            offset_x2 = (x2 - nx2) / float(bbox_size)
            offset_y2 = (y2 - ny2) / float(bbox_size)

            offset_left_eye_x = (landmark[0] - nx1) / float(bbox_size)
            offset_left_eye_y = (landmark[1] - ny1) / float(bbox_size)

            offset_right_eye_x = (landmark[2] - nx1) / float(bbox_size)
            offset_right_eye_y = (landmark[3] - ny1) / float(bbox_size)

            offset_nose_x = (landmark[4] - nx1) / float(bbox_size)
            offset_nose_y = (landmark[5] - ny1) / float(bbox_size)

            offset_left_mouth_x = (landmark[6] - nx1) / float(bbox_size)
            offset_left_mouth_y = (landmark[7] - ny1) / float(bbox_size)

            offset_right_mouth_x = (landmark[8] - nx1) / float(bbox_size)
            offset_right_mouth_y = (landmark[9] - ny1) / float(bbox_size)


            # cal iou
            iou = utils.IoU(crop_box.astype(np.float), np.expand_dims(gt_box.astype(np.float), 0))
            if iou > 0.65:
                save_file = os.path.join(landmark_imgs_save_dir, "%s.jpg" % l_idx)
                cv2.imwrite(save_file, resized_im)

                f.write(save_file + ' -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n' % \
                (offset_x1, offset_y1, offset_x2, offset_y2, \
                offset_left_eye_x,offset_left_eye_y,offset_right_eye_x,offset_right_eye_y,offset_nose_x,offset_nose_y,offset_left_mouth_x,offset_left_mouth_y,offset_right_mouth_x,offset_right_mouth_y))

                l_idx += 1

    f.close()




def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dface_traindata_store', dest='traindata_store', help='dface train data temporary folder,include 12,24,48/postive,negative,part,landmark',
                        default='/idata/data/wider/', type=str)
    parser.add_argument('--anno_file', dest='annotation_file', help='celeba dataset original annotation file',
                        default='/idata/data/trainImageList.txt', type=str)
    parser.add_argument('--prefix_path', dest='prefix_path', help='annotation file image prefix root path',
                        default='/idata/data', type=str)


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    gen_data(args.annotation_file, args.traindata_store, args.prefix_path)


