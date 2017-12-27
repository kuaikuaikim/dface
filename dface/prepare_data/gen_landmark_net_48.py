import argparse

import cv2
import numpy as np
from core.detect import MtcnnDetector,create_mtcnn_net
from core.imagedb import ImageDB
from core.image_reader import TestImageLoader
import time
import os
import cPickle
from dface.core.utils import convert_to_square,IoU
import dface.config as config
import dface.core.vision as vision

def gen_landmark48_data(data_dir, anno_file, pnet_model_file, rnet_model_file, prefix_path='', use_cuda=True, vis=False):


    pnet, rnet, _ = create_mtcnn_net(p_model_path=pnet_model_file, r_model_path=rnet_model_file, use_cuda=use_cuda)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, min_face_size=12)

    imagedb = ImageDB(anno_file,mode="test",prefix_path=prefix_path)
    imdb = imagedb.load_imdb()
    image_reader = TestImageLoader(imdb,1,False)

    all_boxes = list()
    batch_idx = 0

    for databatch in image_reader:
        if batch_idx % 100 == 0:
            print("%d images done" % batch_idx)
        im = databatch


        if im.shape[0] >= 1200 or im.shape[1] >=1200:
            all_boxes.append(np.array([]))
            batch_idx += 1
            continue


        t = time.time()

        p_boxes, p_boxes_align = mtcnn_detector.detect_pnet(im=im)

        boxes, boxes_align = mtcnn_detector.detect_rnet(im=im, dets=p_boxes_align)

        if boxes_align is None:
            all_boxes.append(np.array([]))
            batch_idx += 1
            continue
        if vis:
            rgb_im = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
            vision.vis_two(rgb_im, boxes, boxes_align)

        t1 = time.time() - t
        t = time.time()
        all_boxes.append(boxes_align)
        batch_idx += 1

    save_path = config.MODEL_STORE_DIR

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = os.path.join(save_path, "detections_%d.pkl" % int(time.time()))
    with open(save_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)


    gen_sample_data(data_dir,anno_file,save_file, prefix_path)



def gen_sample_data(data_dir, anno_file, det_boxs_file, prefix_path =''):

    landmark_save_dir = os.path.join(data_dir, "48/landmark")

    if not os.path.exists(landmark_save_dir):
        os.makedirs(landmark_save_dir)


    # load ground truth from annotation file
    # format of each line: image/path [x1,y1,x2,y2] for each gt_box in this image

    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    image_size = 48
    net = "onet"

    im_idx_list = list()
    gt_boxes_list = list()
    gt_landmark_list = list()
    num_of_images = len(annotations)
    print("processing %d images in total" % num_of_images)

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_idx = annotation[0]

        boxes = map(float, annotation[1:5])
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        landmarks = map(float, annotation[5:])
        landmarks = np.array(landmarks, dtype=np.float32).reshape(-1, 10)

        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)
        gt_landmark_list.append(landmarks)


    save_path = config.ANNO_STORE_DIR
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    f = open(os.path.join(save_path, 'landmark_48.txt'), 'w')


    det_handle = open(det_boxs_file, 'r')

    det_boxes = cPickle.load(det_handle)
    print(len(det_boxes), num_of_images)
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"

    # index of neg, pos and part face, used as their image names
    p_idx = 0
    image_done = 0
    for im_idx, dets, gts, landmark in zip(im_idx_list, det_boxes, gt_boxes_list, gt_landmark_list):
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(os.path.join(prefix_path,im_idx))
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        for box in dets:
            x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
               continue
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                offset_left_eye_x = (landmark[0,0] - x_left) / float(width)
                offset_left_eye_y = (landmark[0,1] - y_top) / float(height)

                offset_right_eye_x = (landmark[0,2] - x_left) / float(width)
                offset_right_eye_y = (landmark[0,3] - y_top) / float(height)

                offset_nose_x = (landmark[0,4] - x_left) / float(width)
                offset_nose_y = (landmark[0,5] - y_top) / float(height)

                offset_left_mouth_x = (landmark[0,6] - x_left) / float(width)
                offset_left_mouth_y = (landmark[0,7] - y_top) / float(height)

                offset_right_mouth_x = (landmark[0,8] - x_left) / float(width)
                offset_right_mouth_y = (landmark[0,9] - y_top) / float(height)



                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(landmark_save_dir, "%s.jpg" % p_idx)

                    f.write(save_file + ' -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n' % \
                            (offset_x1, offset_y1, offset_x2, offset_y2, \
                             offset_left_eye_x, offset_left_eye_y, offset_right_eye_x, offset_right_eye_y,
                             offset_nose_x, offset_nose_y, offset_left_mouth_x, offset_left_mouth_y,
                             offset_right_mouth_x, offset_right_mouth_y))

                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

    f.close()



def model_store_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))+"/model_store"



def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        default='../data/wider/', type=str)
    parser.add_argument('--anno_file', dest='annotation_file', help='output data folder',
                        default='../data/wider/anno.txt', type=str)
    parser.add_argument('--pmodel_file', dest='pnet_model_file', help='PNet model file path',
                        default='/idata/workspace/mtcnn/model_store/pnet_epoch_5best.pt', type=str)
    parser.add_argument('--rmodel_file', dest='rnet_model_file', help='RNet model file path',
                        default='/idata/workspace/mtcnn/model_store/rnet_epoch_1.pt', type=str)
    parser.add_argument('--gpu', dest='use_cuda', help='with gpu',
                        default=config.USE_CUDA, type=bool)
    parser.add_argument('--prefix_path', dest='prefix_path', help='image prefix root path',
                        default='', type=str)

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    gen_landmark48_data(args.dataset_path, args.annotation_file, args.pnet_model_file, args.rnet_model_file, args.prefix_path, args.use_cuda)



