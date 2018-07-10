import itertools

import numpy as np
from scipy import ndimage
from skimage.filters import sobel
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation
from skimage.morphology import disk, binary_erosion
from skimage.segmentation import clear_border


def segment_lungs(im):
    """
    Using a 2D image as input (in the form of an array), this function will generate a binary mask and an output image that both only show the lungs.
    """
    ##### BINARIZE IMAGE
    # first anything <0 (as some images are negative)
    im_b = np.where(im < 0, 0, im)
    # then based on otsu thresholding
    thresh = threshold_otsu(im_b)
    binary = im_b > thresh

    # invert the image to make the lungs the ROIs
    binary = np.invert(binary)

    ##### GENERATE BORDER OF THE BINARIES
    cleared = clear_border(binary)

    ##### LABEL DISTINCT BODIES IN IMAGE
    label_image = label(cleared)

    ##### KEEP TWO LARGEST AREAS
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0

    ##### EROSION TO GET RID OF ARTIFACTS
    selem = disk(4)
    binary = binary_erosion(binary, selem)

    ##### DILATION TO BRING BACK LUNG INFORMATION
    selem = disk(4)
    binary = binary_dilation(binary, selem)

    ##### FILL SMALL HOLES
    edges = sobel(binary)
    binary = ndimage.binary_fill_holes(edges)

    return binary


def bbox2_ND(img):
    """
    Generates a bounding box for 3D image.
    """
    N = img.ndim
    out = []
    for ax in itertools.combinations(range(N), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    b = tuple(out)
    img_roi = img[b[4]:b[5], b[2]:b[3], b[0]:b[1]]
    return img_roi
