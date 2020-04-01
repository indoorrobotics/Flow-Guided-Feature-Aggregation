
import os
import json
import shutil
import xml.etree.cElementTree as ET
from xml.dom import minidom
import re
from PIL import Image
from os.path import join
import numpy as np
from glob import glob
import random


YMAX = "ymax"

YMIN = "ymin"

XMAX = "xmax"

XMIN = "xmin"

PERSON = "person"

BNDBOX = "bndbox"

IMAGE_ = "image_"

TRAIN = "train"

TEST = "test"

VID = "VID"

DATA = "Data"

ANNOTATIONS = "Annotations"


label_path = "/data2/cvgl/group/jrdb/data/train_dataset/labels/labels_2d"
xml_dir = "/data2/xmls"
session_base_dir = "/data2/xmls/Data/VID/train/"
base_output = "/media/indoordesk/653ce34c-0c14-4427-8029-be7afe6d1989/test_sets"
names = set()

def plot_image(image_path, xml_path):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    tree = ET.parse(xml_path)
    root = tree.getroot()
    im = np.array(Image.open(image_path))
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    for object in root.findall('object'):

        name = object.find('name').text
        box = object.find(BNDBOX)
        xmin = float(box.find('xmin').text)
        xmax = float(box.find('xmax').text)
        ymin = float(box.find('ymin').text)
        ymax = float(box.find('ymax').text)
        # Create a Rectangle patch
        w = xmax - xmin
        h = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
        print name, xmin
    plt.show()

def make_dirs(base_output):
    os.mkdir(join(base_output))
    os.mkdir(join(base_output, ANNOTATIONS))
    os.mkdir(join(base_output, ANNOTATIONS, VID))
    os.mkdir(join(base_output, DATA))
    os.mkdir(join(base_output, DATA, VID))
    os.mkdir(join(base_output, ANNOTATIONS, VID, TRAIN))
    os.mkdir(join(base_output, ANNOTATIONS, VID, TEST))
    os.mkdir(join(base_output, DATA, VID, TEST))
    os.mkdir(join(base_output, DATA, VID, TRAIN))

def create_xml(file_name, size_, objects, output_xml_file):


    root = ET.Element("annotation")
    #folder = ET.SubElement(root, "folder")
    ET.SubElement(root, "filename").text = file_name
    source = ET.SubElement(root, "source")
    size = ET.SubElement(root, "size")


    ET.SubElement(source, "database").text = "jrdb"
    ET.SubElement(size, "width").text = size_[0]
    ET.SubElement(size, "height").text = size_[1]
    for obj in objects:
        object = ET.SubElement(root, "object")
        ET.SubElement(object, "name").text = obj["name"]
        bndbox = ET.SubElement(object, BNDBOX)
        ET.SubElement(bndbox, XMIN).text = str(obj[BNDBOX][XMIN])
        ET.SubElement(bndbox, XMAX).text = str(obj[BNDBOX][XMAX])
        ET.SubElement(bndbox, YMIN).text = str(obj[BNDBOX][YMIN])
        ET.SubElement(bndbox, YMAX).text = str(obj[BNDBOX][YMAX])
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(output_xml_file, "w") as f:
        f.write(xmlstr)

def convert_to_xml(image_path, output_xml_file):
    objects = []
    file_name = os.path.basename(image_path)
    label_file = image_path.replace(".png", ".txt")
    width, height = Image.open(image_path).size
    if not os.path.exists(label_file):
        create_xml(file_name, (str(width), str(height)), objects, output_xml_file)
    else:
        arr = np.loadtxt(label_file)
        if arr.shape == (0,):
            create_xml(file_name, (str(width), str(height)), objects, output_xml_file)
        else:
            if len(arr.shape) == 1:
                arr = np.expand_dims(arr, axis=0)

            for row in arr:
                obj = {}
                obj["name"] = PERSON
                box = {XMIN: (row[1] - row[3]/2)* width, XMAX: (row[1]+ row[3]/2)* width, YMIN: (row[2]- row[4]/2) * height, YMAX: (row[2] + row[4]/2)* height}
                obj[BNDBOX] = box
                objects.append(obj)
            create_xml(file_name, (str(width), str(height)), objects, output_xml_file)

def get_index(test_str):

    regex = r".+(\d{10}).png"
    matches = list(re.findall(regex, test_str))
    assert len(matches)==1
    return int(matches[0])


def create_image_and_xml(sess, output_xml_file, base_input_dir):
    sess_full_path = join(base_input_dir, sess)
    output_xml_sess = join(output_xml_file, sess)
    if not os.path.exists(output_xml_sess):
        os.mkdir(output_xml_sess)
    output_image_sess = join(base_output, DATA, VID, TEST, sess)
    if not os.path.exists(output_image_sess):
        os.mkdir(output_image_sess)
    os_listdir = os.listdir(sess_full_path)
    sess_files_images = list(map(lambda x: join(sess_full_path, x),
                                 filter(lambda x: x.lower().split(".")[-1] in ["jpg", "jpeg", "png"], os_listdir)
                                 ))
    for image_path in sess_files_images:
        index = get_index(image_path)
        convert_to_xml(image_path, join(output_xml_sess, str(index).zfill(6) + ".xml"))
        dst = join(output_image_sess, str(index).zfill(6) + ".jpg")
        print("About to copy ", image_path, dst)
        shutil.copy(image_path, dst)

def plot_in(sess, idx):
    base = "/media/indoordesk/653ce34c-0c14-4427-8029-be7afe6d1989/test_sets/Data/VID/test"
    xml_path = join(base.replace(DATA, ANNOTATIONS), sess, str(idx).zfill(6) + ".xml")
    img_path = join(base, sess, str(idx).zfill(6) + ".png")
    plot_image(img_path, xml_path)


def create_image_set_test(data_dir, max_data = 100):
    image_sets = "ImageSets"
    if not os.path.exists(join(data_dir, image_sets)):
        os.makedirs(join(data_dir, image_sets))

    with open(join(data_dir, image_sets, "VID_val_frames.txt"), "w") as f:
        images = glob(join(data_dir, ANNOTATIONS, VID, TEST, "*", "*"))
        print(join(data_dir, ANNOTATIONS, VID, TEST, "*", "*"))
        random.shuffle(images)
        images = images[: max_data]
        print "total images %d" % len(images)
        for dire in images:
            file_name = dire.replace(join(join(data_dir, ANNOTATIONS, VID)) + "/", "").replace(".xml", "")
            line = '%s %s\n' % (file_name, '1')
            f.write(line)


if __name__ == '__main__':
    make_dirs(base_output)
    output_xml_file = join(base_output, ANNOTATIONS, VID, TEST)
    base_input_dir = "/media/indoordesk/653ce34c-0c14-4427-8029-be7afe6d1989/video/processed/"

    for sess in os.listdir(base_input_dir):
        create_image_and_xml(sess, output_xml_file, base_input_dir)
    # create_image_set_test(base_output)
    #plot_in("a", 1376)