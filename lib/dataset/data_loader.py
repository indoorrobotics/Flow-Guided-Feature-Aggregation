from glob import glob
import os
import cv2
import numpy as np
from utils.image import resize, transform
import mxnet as mx

######### wrote by ron!!! ############3333

class DataLoader:
    def __init__(self, input_dir, cfg, sort = True, max_len = None):
        self.current = 0
        self.cfg = cfg
        self.image_names = glob(input_dir)
        self.image_names = list(filter(lambda x: x.split(".")[-1].lower() in ["jpeg", "jpg", "png"] ,self.image_names))
        if sort:
            self.image_names.sort()
        if max_len is not None:
            self.image_names = self.image_names[:max_len]

        self.len_data= len(self.image_names)
        print("num of images", len(self.image_names))

    def __iter__(self):
        return self

    def next(self): # Python 2: def next(self)

        if self.current < self.len_data:
            data = self.get_data_by_index(self.current)
            self.current += 1
            return data

        raise StopIteration

    def get_len_data(self):
        return self.len_data

    def read_data(self, im_name):  #data (1,3, 562,1000)
        data_names = ['data', 'im_info', 'data_cache', 'feat_cache']
        assert os.path.exists(im_name), ('%s does not exist'.format(im_name))
        print("About to read ", im_name)
        im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = self.cfg.SCALES[0][0]
        max_size = self.cfg.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=self.cfg.network.IMAGE_STRIDE)
        im_tensor = transform(im, self.cfg.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        d = {'data':  mx.nd.array(im_tensor), 'im_info': mx.nd.array(im_info), 'data_cache': mx.nd.array(im_tensor),
             'feat_cache': mx.nd.array(im_tensor)}

        return [d[data_name] for data_name in data_names]

    def get_data_by_index(self, idx):
        return self.read_data(self.image_names[idx])

    def get_provided_data_and_label(self):
        data_names = ['data', 'im_info', 'data_cache', 'feat_cache']
        provide_data = []
        provide_label = []
        for i in xrange(self.len_data-1):
            shapes = []
            data = self.read_data(self.image_names[i]) # TODO FIX IT
            for k,v in zip(data_names, data):
                shapes.append((k, v.shape))

          #  [(k, v.shape) for k, v in zip(data_names, data[i])]
            provide_data.append(shapes)
            provide_label.append(None)

        return provide_data, provide_label

