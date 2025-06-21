import os
import shutil
import fnmatch
import argparse
import cv2
import pickle

def recursive_glob(rootdir='.', pattern='*'):
    """Search recursively for files matching a specified pattern.

    Adapted from http://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
    """

    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))

    return matches

def horizontal_split(img, w, h):

    img_left = img[0:h, 0:h, :]
    img_right = img[0:h, w-h:w, :]
    img_lr_mid = img[0:h, int((w-h)/2) : (int((w-h)/2) + h), :]

    return img_left, img_lr_mid, img_right

def vertical_split(img, w, h):

    img_up = img[0:w, 0:w, :]
    img_down = img[h-w:h, 0:w, :]
    img_ud_mid = img[int((h-w)/2) : (int((h-2)/2) + w), 0:w, :]

    return img_up, img_ud_mid, img_down

def split_offset(w, h):
    if w < h: # vertical_split
        return [0, 0], [(h-w)/2, 0], [h - w, 0] # up, ud_mid, down
    else:     # horizontal_split
        return [0, 0], [0, (w-h)/2], [0, w - h] # left, lr_mid, right

def save_split_offset(path, offset):
    with open(path, 'wb') as f:
        pickle.dump(offset, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--image_output_dir')
    parser.add_argument('--offset_output_dir')
    parser.add_argument('--ext')
    args = parser.parse_args()

    # os.makedirs(args.image_output_dir, exist_ok=True)
    os.makedirs(args.offset_output_dir, exist_ok=True)

    paths = recursive_glob(args.input_dir, '*.' + args.ext)

    for path in paths:
        filename = os.path.basename(path)
        basename, ext = os.path.splitext(filename)

        img = cv2.imread(os.path.join(path))
        h, w = img.shape[0], img.shape[1]

        if w < h:
            # img_up, img_ud_mid, img_down = vertical_split(img, w, h)
            offset_up, offset_ud_mid, offset_down = split_offset(w, h)
            # cv2.imwrite(os.path.join(args.image_output_dir, basename + '_up.jpg'), img_up)
            # cv2.imwrite(os.path.join(args.image_output_dir, basename + '_ud_mid.jpg'), img_ud_mid)
            # cv2.imwrite(os.path.join(args.image_output_dir, basename + '_down.jpg'), img_down)

            save_split_offset(os.path.join(args.offset_output_dir, basename + '_up.pkl'), offset_up)
            save_split_offset(os.path.join(args.offset_output_dir, basename + '_ud_mid.pkl'), offset_ud_mid)
            save_split_offset(os.path.join(args.offset_output_dir, basename + '_down.pkl'), offset_down)


        else:
            # img_left, img_lr_mid, img_right = horizontal_split(img, w, h)
            offset_left, offset_lr_mid, offset_right = split_offset(w, h)
            # cv2.imwrite(os.path.join(args.image_output_dir, basename + '_left.jpg'), img_left)
            # cv2.imwrite(os.path.join(args.image_output_dir, basename + '_lr_mid.jpg'), img_lr_mid)
            # cv2.imwrite(os.path.join(args.image_output_dir, basename + '_right.jpg'), img_right)

            save_split_offset(os.path.join(args.offset_output_dir, basename + '_left.pkl'), offset_left)
            save_split_offset(os.path.join(args.offset_output_dir, basename + '_lr_mid.pkl'), offset_lr_mid)
            save_split_offset(os.path.join(args.offset_output_dir, basename + '_right.pkl'), offset_right)

if __name__ == '__main__':
    main()