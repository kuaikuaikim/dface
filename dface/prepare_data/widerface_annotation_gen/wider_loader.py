import h5py
import os


class DATA(object):
    def __init__(self, image_name, bboxes):
        self.image_name = image_name
        self.bboxes = bboxes


class WIDER(object):
    def __init__(self, file_to_label, path_to_image):
        self.file_to_label = file_to_label
        self.path_to_image = path_to_image

        self.f = h5py.File(file_to_label, 'r')
        self.event_list = self.f.get('event_list')
        self.file_list = self.f.get('file_list')
        self.face_bbx_list = self.f.get('face_bbx_list')

    def next(self):

        for event_idx, event in enumerate(self.event_list.value[0]):
            directory = self.f[event].value.tostring().decode('utf-16')
            for im_idx, im in enumerate(
                    self.f[self.file_list.value[0][event_idx]].value[0]):

                im_name = self.f[im].value.tostring().decode('utf-16')
                face_bbx = self.f[self.f[self.face_bbx_list.value
                                  [0][event_idx]].value[0][im_idx]].value

                bboxes = []

                for i in range(face_bbx.shape[1]):
                    xmin = int(face_bbx[0][i])
                    ymin = int(face_bbx[1][i])
                    xmax = int(face_bbx[0][i] + face_bbx[2][i])
                    ymax = int(face_bbx[1][i] + face_bbx[3][i])
                    bboxes.append((xmin, ymin, xmax, ymax))

                yield DATA(os.path.join(self.path_to_image, directory,
                           im_name + '.jpg'), bboxes)
