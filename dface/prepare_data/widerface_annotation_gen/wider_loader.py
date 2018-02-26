import os
from scipy.io import loadmat

class DATA:
    def __init__(self, image_name, bboxes):
        self.image_name = image_name
        self.bboxes = bboxes


class WIDER(object):
    def __init__(self, file_to_label, path_to_image=None):
        self.file_to_label = file_to_label
        self.path_to_image = path_to_image

        self.f = loadmat(file_to_label)
        self.event_list = self.f['event_list']
        self.file_list = self.f['file_list']
        self.face_bbx_list = self.f['face_bbx_list']

    def next(self):
        for event_idx, event in enumerate(self.event_list):
            e = event[0][0].encode('utf-8')
            for file, bbx in zip(self.file_list[event_idx][0],
                                 self.face_bbx_list[event_idx][0]):
                f = file[0][0].encode('utf-8')
                path_of_image = os.path.join(self.path_to_image, e, f) + ".jpg"

                bboxes = []
                bbx0 = bbx[0]
                for i in range(bbx0.shape[0]):
                    xmin, ymin, xmax, ymax = bbx0[i]
                    bboxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
                yield DATA(path_of_image, bboxes)
                    
