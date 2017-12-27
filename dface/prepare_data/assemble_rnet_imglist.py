import os
import dface.config as config
import dface.prepare_data.assemble as assemble


if __name__ == '__main__':

    anno_list = []

    # pnet_landmark_file = os.path.join(config.ANNO_STORE_DIR,config.RNET_LANDMARK_ANNO_FILENAME)
    pnet_postive_file = os.path.join(config.ANNO_STORE_DIR,config.RNET_POSTIVE_ANNO_FILENAME)
    pnet_part_file = os.path.join(config.ANNO_STORE_DIR,config.RNET_PART_ANNO_FILENAME)
    pnet_neg_file = os.path.join(config.ANNO_STORE_DIR,config.RNET_NEGATIVE_ANNO_FILENAME)

    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)
    # anno_list.append(pnet_landmark_file)

    imglist_filename = config.RNET_TRAIN_IMGLIST_FILENAME
    anno_dir = config.ANNO_STORE_DIR
    imglist_file = os.path.join(anno_dir, imglist_filename)

    chose_count = assemble.assemble_data(imglist_file ,anno_list)
    print("PNet train annotation result file path:%s" % imglist_file)
