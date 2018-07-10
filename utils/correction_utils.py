from utils.correction_utils import *


def read_and_correct_dicom(directory, clf):
    """
    Given a directory and classifier, this function:
    1) reads all .dcm files and orders them according to location;
    2) generates a segmented mask of the lungs
    3) applies the border correction algorithm and fills in any identified point-pairs
    
    Returns the raw 3D image, the corrected 3D image, and the order of slices within the dicom image (which is important for the XML information, if available)
    """
    # read the dicom image
    img_3d, slice_order = read_dicom_directory(directory)

    # generate the mask
    mask = np.asarray([segment_lungs(slice) for slice in img_3d.T.copy()])

    # correct the mask based on classifying data
    mask_corr = np.asarray([border_correct(slice, clf) for slice in mask.astype(np.uint8).copy()])

    # apply the mask to the read image
    segmented_corrected = img_3d.T.copy() * (mask_corr * 1)

    return img_3d.T, segmented_corrected, slice_order
