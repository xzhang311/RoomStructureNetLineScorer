import cv2
import os
import argparse
import tqdm
import pickle
import numpy as np
import fnmatch
from scipy.spatial.distance import cdist
import pickle
from numpy.linalg import inv

ANGLE_THRESHOLD = 10 # degree
DIST_THRESHOLD = 10 # pixels
OVERLAPPING_THRESHOLD = 0.9

def recursive_glob_full_path(rootdir='.', pattern='*'):
    """Search recursively for files matching a specified pattern.

    Adapted from http://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
    """

    matches_full_paths = []
    ids = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches_full_paths.append(os.path.join(root, filename))
            basename, ext = os.path.splitext(filename)
            ids.append(basename)

    return matches_full_paths, ids

def compute_pnt_to_line_dist(q1, q2, p):
    qvec = q2 - q1
    qvec_unit = qvec / np.linalg.norm(qvec)

    proj_p = np.dot((p - q1)[:, 0], qvec_unit[:, 0])
    dist_p = np.linalg.norm((p - q1) - (proj_p * qvec_unit.reshape(2, 1)))
    return dist_p

# def compute_ll_dist(p1, p2, q1, q2):
#     # q1, q2 define layout line.
#     # compute distance from p1 to the layout line and from p2 to the layout line.
#     dist_p1 = compute_pnt_to_line_dist(q1, q2, p1)
#     dist_p2 = compute_pnt_to_line_dist(q1, q2, p2)
#     dist = np.min((np.abs(dist_p1), np.abs(dist_p2)))
#     return dist

def comptue_ll_dist(p1, p2, q1, q2):
    l = np.sqrt(np.sum((p2 - p1)**2))
    steps = np.round(l)

    v1 = p2 - p1
    v1 = v1 / np.linalg.norm(v1)

    dists = []

    for s in range(steps.astype(np.int16)):
        p = p1 + v1 * s
        dists.append(compute_pnt_to_line_dist(q1, q2, p))

    dist_ave = np.mean(np.asarray(dists))

    return dist_ave


def compute_overlapping_ratio(p1, p2, q1, q2):
    # q1, q2 define layout line.
    # compute the overlapping ratio between the part that line(p1, p2) project to inside the range of line(q1, q2) and the whole
    # projection of line(p1, p2).

    qvec = q2 - q1
    qvec_unit = qvec / np.linalg.norm(qvec)

    proj_q1 = 0
    proj_q2 = np.dot((q2 - q1)[:, 0], qvec_unit[:, 0])

    proj_p1 = np.dot((p1 - q1)[:, 0], qvec_unit[:, 0])
    proj_p2 = np.dot((p2 - q1)[:, 0], qvec_unit[:, 0])

    if proj_p1 > proj_q2 and proj_p2 > proj_q2:
        return 0

    if proj_p1 < proj_q1 and proj_p2 < proj_q1:
        return 0

    d = np.asarray([proj_q1, proj_q2, proj_p1, proj_p2])
    d = np.sort(d)

    r = np.abs(d[1]-d[2])/np.abs(proj_p1 - proj_p2)

    return r


def compute_radian_dist(p1, p2, q1, q2):
    v1 = p2 - p1
    v1 = v1 / np.linalg.norm(v1)

    v2 = q2 - q1
    v2 = v2 / np.linalg.norm(v2)

    cos = np.linalg.norm(np.dot(v1[:, 0], v2[:, 0]))/(np.linalg.norm(v1) * np.linalg.norm(v2))
    cos = np.abs(cos)


    # In radians [0, pi]
    radian = 0
    try:
        cos = np.clip(cos, -1, 1)
        radian = np.arccos(cos)
    except:
        print('arccos exception, cos value is {}'.format(cos))
        radian = np.pi/2

    return radian

def compute_translate_vecs(p1, p2, q1, q2):
    # compute translate vecs for p1 and p2 so that they can overlap
    # with line(q1, q2) after applying the translation.
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    if p1[0] == p2[0]:
        if p1[1] > p2[1]:
            p1, p2 = p2, p1

    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)

    qvec = q2 - q1
    qvec_unit = qvec / np.linalg.norm(qvec)
    proj_p1 = np.dot((p1 - q1)[:, 0], qvec_unit)
    proj_p2 = np.dot((p2 - q1)[:, 0], qvec_unit)

    anchor_p1 = q1 + qvec_unit * proj_p1
    anchor_p2 = q2 + qvec_unit * proj_p2

    return anchor_p1 - p1[:, 0], anchor_p2 - p2[:, 0]

def is_line_layout_line(layout_lines, line):
    # 1. Find layout line that is closest to the input line according to designed metric.
    # 2. Compute the translation vectors for two end points of the input line, so that the line can
    #    merge with the closest layout line by making that translation.
    p1 = line[0, :].astype(np.float)[:, np.newaxis]
    p2 = line[1, :].astype(np.float)[:, np.newaxis]

    dists = []
    angles = []
    overlapping_ratio = []

    for layout_line in layout_lines:
        l1_p1 = np.asarray(layout_line['p1'])[:, np.newaxis]
        l1_p2 = np.asarray(layout_line['p2'])[:, np.newaxis]

        ll_dist = 0
        radian_dist = 0

        radian_dist = compute_radian_dist(p1, p2, l1_p1, l1_p2)
        angle = 180 * radian_dist/np.pi
        dist = comptue_ll_dist(p1, p2, l1_p1, l1_p2)
        olr = compute_overlapping_ratio(p1, p2, l1_p1, l1_p2)

        angles.append(angle)
        dists.append(dist)
        overlapping_ratio.append(olr)

    min_dist, min_dist_idx = np.min(dists), np.argmin(dists)
    is_layout_line = False

    if angles[min_dist_idx] < ANGLE_THRESHOLD and min_dist < DIST_THRESHOLD and overlapping_ratio[min_dist_idx] >= OVERLAPPING_THRESHOLD:
        is_layout_line = True

    trans_vec_p1, trans_vec_p2 = compute_translate_vecs(p1, p2, layout_lines[min_dist_idx]['p1'], layout_lines[min_dist_idx]['p2'])

    layout_line_type = layout_lines[min_dist_idx]['type']

    return min_dist, is_layout_line, trans_vec_p1, trans_vec_p2, layout_line_type

def main():
    parser = argparse.ArgumentParser(description='Turn edges in layout map to line segments')
    # /mnt/ebs_xizhn2/Data/DYR/OFFLINE_DATASET/lsun_experiment/lcnn_pred
    parser.add_argument('--in_dir_lcnn_lines', default='Input dir for lcnn detected lines')
    # /mnt/ebs_xizhn2/Data/DYR/OFFLINE_DATASET/lsun_experiment/labels_lines
    parser.add_argument('--in_dir_gt_layout_lines', default='Input dir for gt layout lines')
    # /mnt/ebs_xizhn2/Data/DYR/OFFLINE_DATASET/lsun_experiment/images
    parser.add_argument('--in_dir_original_images', default='Input dir of original rgb images')
    parser.add_argument('--out_dir', default='Output dir')
    parser.add_argument('--plot', action='store_true', help='if plot lines')
    args = parser.parse_args()

    paths_lcnn_lines, ids_lcnn_lines = recursive_glob_full_path(args.in_dir_lcnn_lines, pattern='*.pkl')

    count = 0
    for lcnn_lines_path, lcnn_lines_id in zip(paths_lcnn_lines, ids_lcnn_lines):
        print("{}: {}".format(count, lcnn_lines_id))

        line_scores = []

        # load lcnn prediction
        with open(lcnn_lines_path, 'rb') as f:
            lcnn_pred = pickle.load(f)
            lcnn_lines = lcnn_pred['preds']['lines']

        img_rgb = cv2.imread(os.path.join(args.in_dir_original_images, lcnn_lines_id+'.png'))

        with open(os.path.join(args.in_dir_gt_layout_lines, lcnn_lines_id+'.pkl'), 'rb') as f:
            layout_lines = pickle.load(f)

        # rearrange end point order in lcnn_lines
        for i in range(len(lcnn_lines)):
            line = lcnn_lines[i, :]
            p1 = line[0, :].astype(np.int16)
            p2 = line[1, :].astype(np.int16)

            # Need to reorder the points, so p1[0] is x p1[1] is y
            p1 = p1[::-1]
            p2 = p2[::-1]

            if p1[0] > p2[0]:
                p1, p2 = p2, p1

            if p1[0] == p2[0]:
                if p1[1] > p2[1]:
                    p1, p2 = p2, p1

            line[0, :] = p1
            line[1, :] = p2

            lcnn_lines[i, :] = line

        lcnn_pred['preds']['lines'] = lcnn_lines

        for i in range(len(lcnn_lines)):
            line = lcnn_lines[i, :]
            p1 = line[0, :].astype(np.int16)
            x1, y1 = p1[0], p1[1]
            p2 = line[1, :].astype(np.int16)
            x2, y2 = p2[0], p2[1]

            dist, is_layout_line, trans_vec_p1, trans_vec_p2, layout_line_type = is_line_layout_line(layout_lines, line)

            line_score = {}
            line_score['dist'] = dist
            line_score['is_layout_line'] = is_layout_line
            line_score['trans_vec_p1'] = trans_vec_p1
            line_score['trans_vec_p2'] = trans_vec_p2
            line_score['layout_line_type'] = layout_line_type

            line_scores.append(line_score)

            if args.plot:
                cv2.line(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                if dist<DIST_THRESHOLD and is_layout_line:
                    cv2.line(img_rgb, (x1, y1), (x2, y2), (0, 255, 255), 2)

        lcnn_pred['groundtruth'] = line_scores
        lcnn_pred['lines'] = lcnn_pred['preds']['lines']

        lcnn_pred.pop('preds', None)

        if args.plot:
            for layout_line in layout_lines:
                p1 = layout_line['p1']
                x1, y1 = p1[0], p1[1]
                p2 = layout_line['p2']
                x2, y2 = p2[0], p2[1]
                cv2.line(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.imwrite(os.path.join(args.out_dir, lcnn_lines_id + '_lines.png'), img_rgb)

        with open(os.path.join(args.out_dir, lcnn_lines_id + '.pkl'), 'wb') as f:
            pickle.dump(lcnn_pred, f)

        count = count + 1

if __name__ == '__main__':
    main()