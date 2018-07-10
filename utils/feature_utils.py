import math
import random

import cv2
import numpy as np


def distance_metrics(cont_full):
    """
    Takes the non-smoothed contour of BOTH LUNGS to generate distance metrics for the lungs, specifically:
    c: the center point of both lungs
    avg_dist: the average distance of every point in both lungs from c
    max_dist: the maximum distance that a point on the contour lies from c
    
    Outputs these measurements in order
    """

    if len(cont_full) == 1:
        lungs = np.asarray(cont_full[0][:, 0, :])
    else:
        lungs = np.vstack((cont_full[0][:, 0, :], cont_full[1][:, 0, :]))

    # calculate centroid
    c = [np.average(lungs[0]), np.average(lungs[1])]

    max_dist = 0
    distances = []
    for i in np.arange(0, lungs.shape[1], 1):
        p1 = lungs.T[i]
        dist_temp = math.sqrt((p1[0] - c[0]) ** 2 + (p1[1] - c[1]) ** 2)
        distances.append(dist_temp)
        if dist_temp > max_dist:
            max_dist = dist_temp
        else:
            continue

    # calculate average distance from centroid
    avg_dist = np.average(distances)

    return c, max_dist, avg_dist


def single_lung_features(cont_com):
    """
    From a SINGLE LUNG, COMPLETE contour, generates the following outputs:
    
    points_list: all of the pixel coordinates along the contour, in list format
    border_points: all of the pixel coordinates along the contour, in string format
    boundary_length: the number of pixels in the contour
    """

    points_list = cont_com[:, 0, :].tolist()
    border_points = [str(cont_com[i][0]) for i in np.arange(0, cont_com.shape[0], 1)]
    boundary_length = len(border_points)

    return points_list, border_points, boundary_length


def find_inflection_points(cont_smooth):
    """
    From an individual, smoothed contour, generates inflection points where the second-order differential is 0. It can also calculate points of inflection for a complete contour.
    """
    lung_area = np.asarray(cont_smooth[:, 0, :])
    cx = lung_area.T

    # generate second derivative in x-direction
    d2_x = np.gradient(np.gradient(cx[0]))

    # generate second derivative in y-direction
    d2_y = np.gradient(np.gradient(cx[1]))

    # find indices that matter
    indices = []

    for i in np.arange(0, d2_x.shape[0], 1):
        if d2_x[i] == 0:
            indices.append(i)
        elif d2_y[i] == 0:
            indices.append(i)
        else:
            continue
    return np.asarray([lung_area[n].tolist() for n in indices])


def generate_point_pairs(coords, p):
    """
    From a list of coordinates (presumably the inflection points) and a given starting point, generates a point pair from which features will be generated. Of note, the second point is generated somewhat randomly if there is a large amount of points that could be chosen.
    
    """
    # generate point pairs, done at random
    p1 = coords[p]

    if coords.shape[0] < 20:
        p2 = coords[p - 1]
    else:
        r = random.choice([i for i in range(-5, 5) if i not in [0]])
        if p == coords.shape[0] - 1:
            p2 = coords[0]
        if abs(p + r) > coords.shape[0] - 1:
            p2 = coords[p + r - coords.shape[0]]
        else:
            p2 = coords[p + r]

    return p1, p2


def border_correct(test_im, clf):
    """
    Given a 2D image (test_im) and classifier (clf), will generate the inflection points, classify them according to the training data, then fills the contour if the pair is identified with a positive label. Features used in the classifier are:
    f_con: the amount of concavity between two points
    f_len: the relative length of the segment between two points
    f_pos: the position of the midpoint between the two points
    
    If the classifier predicts a "1" label, the function will fill the area between those points and add it to the image.
    
    Returns test_im; modification of this image depends on whether positive points were identified or not. 
    """
    if not test_im.any():
        pass
    else:
        # generate border with no smoothing
        image_full, contours_full, hierarchy_full = cv2.findContours(test_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # generate border with smoothing approximation (see Teh Chi 1989)
        image, contours, hierarchy = cv2.findContours(test_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        # generate overall distance metrics
        c, max_dist, avg_dist = distance_metrics(contours_full)

        # try for random point-pairs
        for i in np.arange(0, len(contours_full), 1):
            points_list, border_points, boundary_length = single_lung_features(contours_full[i])
            coords = find_inflection_points(contours[i])

            for p in np.arange(0, coords.shape[0], 1):
                p1, p2 = generate_point_pairs(coords, p)

                # calculate euclidian distance
                euc_dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                if euc_dist == 0:
                    pass
                else:
                    p1_ind = border_points.index(str(p1))
                    p2_ind = border_points.index(str(p2))

                    points_to_close = points_list[p1_ind:p2_ind + 1]
                    if len(points_to_close) > len(points_list) / 2:
                        points_to_close = points_list[p2_ind:p1_ind + 1]
                    if len(points_to_close) == 0:
                        pass
                    else:
                        closure = cv2.fillPoly(np.zeros(test_im.shape), [np.array(points_to_close)],
                                               color=(255, 255, 255))
                        c_b = closure > 0
                        # this makes sure the randomly generated points do not intersect with the lung
                        if (c_b * test_im).all() != 0:
                            pass
                        else:
                            # border distance
                            seg_length = abs(p1_ind - p2_ind)

                            # sanity check to make sure the border length is less than the
                            if seg_length > boundary_length:
                                seg_length = abs(boundary_length - seg_length)

                            # F CONCAVE
                            if seg_length <= euc_dist:
                                pass
                            else:
                                f_con = seg_length / euc_dist

                                # F LENGTH
                                f_len = seg_length / boundary_length

                                # midpoint
                                m_p = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

                                # distance from midpoint to center
                                m_dist = math.sqrt((m_p[0] - c[0]) ** 2 + (m_p[1] - c[1]) ** 2)

                                # F POSITION
                                f_pos = m_dist / avg_dist
                                label = clf.predict(np.asarray([f_con, f_len, f_pos]).reshape(1, -1))

                                if label == 0:
                                    pass
                                else:
                                    test_im = c_b + test_im
                                    test_im = test_im > 0

    return test_im
