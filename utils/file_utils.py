import os
import random
import xml.etree.ElementTree as ET

import numpy as np
import pydicom


def select_random_directories(num_dirs):
    """
    From a given number num_dirs, generates a list of directories from which to perform the border correction processes.
    
    Outputs a list of directories with which to work
    """
    luna_dirs = [root for root, dirs, files in os.walk('/data/LUNA16/LIDC-IDRI') if not dirs]
    kaggle_dirs = [root for root, dirs, files in os.walk('/data/Kaggle') if not dirs]

    random_luna = random.sample(luna_dirs, int(num_dirs / 2))
    random_kaggle = random.sample(kaggle_dirs, int(num_dirs / 2))

    directories = random_luna + random_kaggle

    return directories


def read_dicom_directory(directory):
    """
    From a directory, reads all dicom images and sorts them according to the "Slice Location" attribute
    """
    files = os.listdir(directory)

    slice_order = {f: pydicom.dcmread(directory + '/' + f).SliceLocation
                   for f in files
                   if f.endswith('.dcm')}

    img_3d = np.dstack([pydicom.dcmread(directory + '/' + f).pixel_array
                        for f in sorted(slice_order, key=slice_order.get)
                        if f.endswith('.dcm')])

    return img_3d, sorted(slice_order.values())


def read_xml_file(directory):
    """
    From a directory, reads an XML file to generate the coordinate information for all identified nodules; both non-nodules and cancerous nodules are identified, and are output in a dictionary with "cancer_nodules" and "non_nodules" as the keys.
    Note: only LUNA directories have XML files in them, so Kaggle directories will not return anything if this function is called on them. 
    """
    cancer_nodules = {}
    non_nodules = {}
    for f in os.listdir(directory):
        if f.endswith('.xml'):
            xml_root = ET.parse(directory + '/' + f).getroot()
            study_uid = xml_root[0].find('{http://www.nih.gov}StudyInstanceUID').text
            polygons = {}
            # find all "item" objects and print their "name" attribute
            for elem in xml_root:
                for subelem in elem.findall('{http://www.nih.gov}unblindedReadNodule'):
                    polygons, nodule_id = get_nodule_coordinates(subelem, cancerous=True)
                    cancer_nodules[nodule_id] = sorted(polygons.items())
                for subelem in elem.findall('{http://www.nih.gov}nonNodule'):
                    polygons, nodule_id = get_nodule_coordinates(subelem, cancerous=False)
                    non_nodules[nodule_id] = sorted(polygons.items())

    all_nodules = {'cancer_nodules': cancer_nodules,
                   'non_nodules': non_nodules}
    return all_nodules


def get_nodule_coordinates(subelem, cancerous=True):
    """
    Within an XML subelement identified as a nodule (either nonNodule or unblindedReadNodule), generates the polygonal coordinates of any nodules identified, as well as their z-coordinates. If cancerous is specified as True, then cancerous nodules are searched and stored; if False, then non-nodules are identified. Stores these in a dictionary with nodule IDs as the keys.
    """
    polygons = {}
    if cancerous:
        nodule_id = subelem.find('{http://www.nih.gov}noduleID').text
        for charSubElem in subelem.findall('{http://www.nih.gov}roi'):
            z_coord = charSubElem.find('{http://www.nih.gov}imageZposition')
            edgemap = charSubElem.findall('{http://www.nih.gov}edgeMap')
            poly = []
            for p in edgemap:
                coords = [int(c.text) for c in p.getchildren()]
                poly.append(coords)
            polygons[z_coord.text] = poly
    else:
        nodule_id = subelem.find('{http://www.nih.gov}nonNoduleID').text
        z_coord = subelem.find('{http://www.nih.gov}imageZposition')
        poly = []
        locus = subelem.findall('{http://www.nih.gov}locus')
        for p in locus:
            coords = [int(c.text) for c in p.getchildren()]
            poly.append(coords)
        polygons[z_coord.text] = poly

    return polygons, nodule_id
