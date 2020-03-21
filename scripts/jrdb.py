
import os
import json
import xml.etree.cElementTree as ET
from xml.dom import minidom
import re
from PIL import Image
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


BNDBOX = "bndbox"

IMAGE_ = "image_"

TRAIN = "train"

TEST = "test"

VID = "VID"

DATA = "Data"

ANNOTATIONS = "Annotations"

label_path = "/home/ron/Desktop/labels/labels_2d"
FILE_NAME = "stlc-111-2019-04-19_0_image6.json"
full_path = os.path.join(label_path, FILE_NAME)
xml_dir = "/home/ron/Desktop/xmls"
session_base_dir = "/media/ron/15GB"
base_output =  xml_dir
def make_dirs(base_output):
    os.makedirs(join(base_output, ANNOTATIONS))
    os.makedirs(join(base_output, ANNOTATIONS, VID))
    os.makedirs(join(base_output, DATA))
    os.makedirs(join(base_output, DATA, VID))
    os.makedirs(join(base_output, ANNOTATIONS, VID, TRAIN))
    os.makedirs(join(base_output, ANNOTATIONS, VID, TEST))
    os.makedirs(join(base_output, DATA, VID, TEST))
    os.makedirs(join(base_output, DATA, VID, TRAIN))

def get_file_name_and_dir(test_str):
    # coding=utf8
    # the above tag defines encoding for this document and is for Python 2.x compatibility

    regex = ur"(.+)_image(\d)\.json"
    matches = list(re.findall(regex, test_str))
    assert len(matches)==1
    assert len(matches[0]) ==2

    file_name = matches[0][0]
    directory = matches[0][1]
    return file_name, directory


def read_labels_dir(label_path):
    labels = os.listdir(label_path)
    labels.sort()
    for label_file_path in labels:
        sesion_file_name, session_directory = get_file_name_and_dir(label_file_path)
        input_img_session_full_path = join(session_base_dir, IMAGE_ + session_directory, sesion_file_name)
        output_img_session_full_path = join(base_output, DATA, VID, TRAIN, IMAGE_ + session_directory,
                                            sesion_file_name)
        output_xml_session_full_path = join(base_output, ANNOTATIONS, VID, TRAIN, IMAGE_ + session_directory,
                                            sesion_file_name)
        os.makedirs(output_img_session_full_path)
        os.makedirs(output_xml_session_full_path)
        json_path = join(label_path, label_file_path)

        convert_to_xml(json_path, input_img_session_full_path, output_img_session_full_path, output_xml_session_full_path)


def convert_to_xml(full_path, img_session_full_path, output_img, output_xml):
    with open(full_path) as f:
        js = json.loads(f.read())['labels']
        file_names = list(js.keys())
        file_names.sort()
        for file_name in file_names:
            image_path = os.path.join(img_session_full_path, file_name)
            if os.path.exists(image_path):
                width, height = Image.open(image_path).size
                file_na = file_name.replace(".jpg", "")
                objects = []
                for ob in js[file_name]:
                    name = ob["label_id"].split(":")[0]
                    box = ob["box"]
                    obj = {"name":name, BNDBOX: {"xmin": box[0], "ymin": box[1],"xmax": box[2] + box[0] ,"ymax": box[1] + box[3]}}
                    objects.append(obj)
                #    < name > n01674464 < / name >
                #    < bndbox >
                #    < xmax > 1050 < / xmax >
                #    < xmin > 323 < / xmin >
                #    < ymax > 428 < / ymax >
                #    < ymin > 216 < / ymin >
                #< / bndbox >
                output_xml_file = join(output_xml, file_na + ".xml")
                create_xml(file_na, (str(width), str(height)), objects, output_xml_file)
            else:
                print("File not found", image_path)
        return js


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
        ET.SubElement(bndbox, "xmin").text = str(obj[BNDBOX]["xmin"])
        ET.SubElement(bndbox, "xmax").text = str(obj[BNDBOX]["xmax"])
        ET.SubElement(bndbox, "ymin").text = str(obj[BNDBOX]["ymin"])
        ET.SubElement(bndbox, "ymax").text = str(obj[BNDBOX]["ymax"])
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(output_xml_file, "w") as f:
        f.write(xmlstr)

def plot_image(image_path, xml_path):
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

if __name__ == '__main__':

    #read_labels_dir(label_path)
    idx = 109
    loction = "clark-center-2019-02-28_0"
    xml_path = join("/home/ron/Desktop/xmls/Annotations/VID/train/image_2/", loction, str(idx).zfill(6) + ".xml")
    img_path = join("/media/ron/15GB/image_2/" ,loction, str(idx).zfill(6) + ".jpg")
    plot_image(img_path, xml_path)
