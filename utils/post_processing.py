import cv2
import fnmatch
import os
import pickle
import numpy as np
from sklearn.cluster import DBSCAN

PROB_THRESHOLD=0.6

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


def load_offset(offset_dir, id):
    path = os.path.join(offset_dir, id + '.pkl')

    with open(path, 'rb') as f:
        return pickle.load(f)

def load_line_scorer_output(line_scorer_output_dir, id):
    path = os.path.join(line_scorer_output_dir, id + '.pkl')

    with open(path, 'rb') as f:
        return pickle.load(f)

def stitch_image(img_left, img_lr_mid, img_right, offset_left, offset_lr_mid, offset_right):
    h = img_left.shape[0]
    w = offset_right[1] + img_right.shape[1]

    offset_left = [int(offset_left[0]), int(offset_left[1])]
    offset_lr_mid = [int(offset_lr_mid[0]), int(offset_lr_mid[1])]
    offset_right = [int(offset_right[0]), int(offset_right[1])]

    canvas = np.zeros((h, w, 3))
    canvas[offset_left[0]:offset_left[0] + img_left.shape[0], offset_left[1]:offset_left[1]+img_left.shape[1], :] = img_left
    canvas[offset_lr_mid[0]:offset_lr_mid[0] + img_lr_mid.shape[0], offset_lr_mid[1]:offset_lr_mid[1] + img_lr_mid.shape[1], :] = img_lr_mid
    canvas[offset_right[0]:offset_right[0] + img_right.shape[0],offset_right[1]:offset_right[1] + img_right.shape[1], :] = img_right

    return canvas

def visualize_lines(img_merged, id_lines, id_prob, id):
    for line, prob in zip(id_lines[id], id_prob[id]):
        x1 = line[0, 0].astype(np.int16)
        y1 = line[0, 1].astype(np.int16)
        x2 = line[1, 0].astype(np.int16)
        y2 = line[1, 1].astype(np.int16)

        type = np.argmax(prob)
        prob_value = np.max(prob)

        color = [[255, 0, 0],
                 [0, 255, 0],
                 [0, 0, 255],
                 [170, 170, 170]]

        # wg:0, ww:1, wc:2, no_layout:3
        if type == 0:
            cv2.line(img_merged, (x1, y1), (x2, y2), np.asarray(color[type]) * prob_value, 3)
    cv2.imwrite('/mnt/ebs_xizhn2/Data/DYR/OFFLINE_DATASET/ARkitData/tmp/' + id + '.jpg', img_merged)

def line_slope_intercept_form(lines):
    # y = mx + b
    line_slope_intercept = []

    for line in lines:
        x1 = line[0, 0]
        y1 = line[0, 1]
        x2 = line[1, 0]
        y2 = line[1, 1]

        slope = (y2 - y1) / (x2 - x1)
        intercept = (x1 * y2 - x1 * y1 - x2 * y1 + x1 * y1) / (x2 - x1)
        line_slope_intercept.append([slope, intercept])

    return line_slope_intercept

def line_intercept_form(lines):
    # x/a + y/b = 1
    line_intercept = []

    for line in lines:
        x1 = line[0, 0]
        y1 = line[0, 1]
        x2 = line[1, 0]
        y2 = line[1, 1]

        x_intercept = (x1 * y2 - x1 * y1 - x2 * y1 + x1 * y1)/(y2 - y1)
        y_intercept = (x1 * y2 - x1 * y1 - x2 * y1 + x1 * y1)/(x1 - x2)

        line_intercept.append([x_intercept, y_intercept])

    return line_intercept

def line_standard_form(lines):
    # ax + by + c = 0
    line_standard_form = []

    for line in lines:
        x1 = line[0, 0]
        y1 = line[0, 1]
        x2 = line[1, 0]
        y2 = line[1, 1]

        a = (y2 - y1)
        b = (x1 - x2)
        c = -1 * (x1 * y2 - x1 * y1 - x2 * y1 + x1 * y1)

        # force a >= 0
        if a >=0:
            line_standard_form.append([a, b, c])
        else:
            line_standard_form.append([-a, -b, -c])

    return line_standard_form

def cluster_lines(lines, probs):
    line_params = line_standard_form(lines)
    layout_line_params = []
    final_picked_line_idx = []

    for idx, t in enumerate(zip(line_params, probs)):
        line = t[0]
        prob = t[1]
        type = np.argmax(prob)
        prob_value = np.max(prob)

        if type!=0:
            continue

        line = line / np.sqrt(np.power(line[0], 2) + np.power(line[1], 2))
        layout_line_params.append([line[0], line[1], line[2], prob_value, idx])

    layout_line_params = np.asarray(layout_line_params)

    if len(layout_line_params)==0:
        return []

    # standardize the offset
    gamma = np.std(layout_line_params[:, 2])
    if gamma != 0:
        layout_line_params[:, 2] = (layout_line_params[:, 2] - np.mean(layout_line_params[:, 2]))/gamma
    else:
        layout_line_params[:, 2] = (layout_line_params[:, 2] - np.mean(layout_line_params[:, 2]))

    clustering = DBSCAN(eps=0.5, min_samples = 1).fit(layout_line_params[:, :3])
    labels = clustering.labels_

    cluster_ids = set(labels)

    for cluster_id in cluster_ids:
        line_idices = np.where(labels==cluster_id)[0]
        line_idx = np.argmax(layout_line_params[line_idices, 3])

        # if layout_line_params[line_idices[line_idx], 3] < PROB_THRESHOLD:
        #     continue

        line_original_idx = int(layout_line_params[line_idices[line_idx], 4])

        picked_line = lines[line_original_idx]

        line_len = np.sqrt(np.power(picked_line[0, 0]-picked_line[1, 0], 2) + np.power(picked_line[0, 1]-picked_line[1, 1], 2))

        if np.abs(picked_line[0, 0] - picked_line[1, 0]) < 5 or line_len < 70:
            continue

        final_picked_line_idx.append(line_original_idx)

    return final_picked_line_idx

def draw_image(image, lines):
    for line in lines:
        x1 = line[0, 0].astype(np.int16)
        y1 = line[0, 1].astype(np.int16)
        x2 = line[1, 0].astype(np.int16)
        y2 = line[1, 1].astype(np.int16)

        cv2.line(image, (x1, y1), (x2, y2), [255, 0, 0], 3)

    return image

def save_lines(root_out, id, lines, prob):
    out_path = os.path.join(root_out, id + '.pkl')

    data = {}
    data['lines'] = lines
    data['prob'] = prob

    with open(out_path, 'wb') as f:
        pickle.dump(data, f)

def aggregate_split_image_lines(all_ids, root_split_offset, root_results_before_postprocessing, root_images, root_final_results):
    ids_left = []
    ids_right = []
    ids_lr_mid = []

    id_lines = {}
    id_prob = {}

    for id in all_ids:
        if 'left' in id:
            ids_left.append(id)
        if 'right' in id:
            ids_right.append(id)
        if 'lr_mid' in id:
            ids_lr_mid.append(id)

    count = 0

    for id in ids_left:
        id = id.replace('_left', '')
        id_left = id + '_left'
        id_lr_mid = id + '_lr_mid'
        id_right = id + '_right'

        print(count)
        print(id)

        offset_left = load_offset(root_split_offset, id_left)
        offset_lr_mid = load_offset(root_split_offset, id_lr_mid)
        offset_right = load_offset(root_split_offset, id_right)

        lines_left = load_line_scorer_output(root_results_before_postprocessing, id_left)
        lines_lr_mid = load_line_scorer_output(root_results_before_postprocessing, id_lr_mid)
        lines_right = load_line_scorer_output(root_results_before_postprocessing, id_right)

        img_left = cv2.imread(os.path.join(root_images, id_left+'.jpg'))
        img_lr_mid = cv2.imread(os.path.join(root_images, id_lr_mid+'.jpg'))
        img_right = cv2.imread(os.path.join(root_images, id_right+'.jpg'))

        img_merged = stitch_image(img_left, img_lr_mid, img_right, offset_left, offset_lr_mid, offset_right)

        # cv2.imwrite('/mnt/ebs_xizhn2/Data/DYR/OFFLINE_DATASET/ARkitData/tmp/'+id+'.jpg', img_merged)

        lines_left['lines'] = lines_left['lines'].data.cpu().numpy() + np.reshape(offset_left[::-1], (1, 1, 2))
        lines_lr_mid['lines'] = lines_lr_mid['lines'].data.cpu().numpy() + np.reshape(offset_lr_mid[::-1], (1, 1, 2))
        lines_right['lines'] = lines_right['lines'].data.cpu().numpy() + np.reshape(offset_right[::-1], (1, 1, 2))

        id_lines[id] = np.concatenate((lines_left['lines'], lines_lr_mid['lines'], lines_right['lines']), axis = 0)
        id_prob[id] = np.concatenate((lines_left['prob'], lines_lr_mid['prob'], lines_right['prob']), axis = 0)

        visualize_lines(img_merged, id_lines, id_prob, id)

        picked_line_idices = cluster_lines(id_lines[id], id_prob[id])

        images = draw_image(img_merged, id_lines[id][picked_line_idices])
        cv2.imwrite(os.path.join(root_final_results, id+'.jpg'), images)
        save_lines(root_final_results, id, id_lines[id][picked_line_idices], id_prob[id][picked_line_idices])

        zx = 0
        count = count + 1


