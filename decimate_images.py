import sys
import os
import argparse
import cv2
import numpy as np

def parse_command_line():
    parser = argparse.ArgumentParser(prog='decimate_images.py', description='filter image files',
                                     epilog='Text at the bottom of help')
    parser.add_argument('sourceFolder', help='source folder containing image files')
    parser.add_argument('destFolder', help='Folder path to store output images')
    parser.add_argument('-t', '--threshold', required=False, type=float, default=0.5, help='threshold in percent. For example -t 0.5')
    return parser.parse_args()


def get_similarity_measure(base_image, image):
    xor_img = cv2.bitwise_xor(base_image, image)
    #print("xor_count=", np.sum(np.sign(xor_img)))
    median_img = cv2.medianBlur(xor_img, 3)
    height, width, layers = base_image.shape
    signal_count = np.sum(np.sign(median_img))
    #print("med_count=", signal_count)
    return (signal_count*100.0)/(height * width)



def decimate_images(source_image_folder, dest_image_folder, threshold):
    images = [img for img in os.listdir(source_image_folder) if img.endswith((".png"))]
    print("Files and directories in '", source_image_folder, "' :")
    images.sort()
    # prints all files
    if len(images) == 0:
        print("No images found in '", source_image_folder, "'")
        return

    print(images)
    print(len(images), " images found")

    # filename should be named chronologically for example image_0001.png, image_0002.png, ...
    # so image_0001.png should correspond to the first frame

    count = 0
    # Set frame from the first image
    base_idx = 0
    frame1 = cv2.imread(os.path.join(source_image_folder, images[base_idx]))
    finished = False
    while not finished:
        if base_idx >= len(images)-2:
            finished = True
            continue

        finished = True
        for i in range(base_idx+1, len(images)):
            frame2 = cv2.imread(os.path.join(source_image_folder, images[i]))
            p = get_similarity_measure(frame1, frame2)
            if p > threshold:
                cv2.imwrite(os.path.join(dest_image_folder, images[base_idx]), frame1)
                count += 1
                frame1 = frame2
                base_idx = i
                print("Next file #", i, " '", os.path.join(dest_image_folder, images[base_idx]), "'")
                finished = False
                break

    frame = cv2.imread(os.path.join(source_image_folder, images[base_idx]))
    cv2.imwrite(os.path.join(dest_image_folder, images[base_idx]), frame)
    print("Accepted ", count + 1, " files")

    return

#
# this script is used to filter out similar images.
# to measure how images differ number of nonzero pixels from
# med_flt(xor(img1, img2), 3) is taken.
# then percentage of nonzero pixels to the image size is compared with threshold value.
# default threshold value is 0.5%.
# Accepted files then are used to produce video.
#

def main():
    args = parse_command_line()
    if args is None: exit

    if not os.path.isdir(args.sourceFolder):
        print("Folder '", args.sourceFolder, "' doesn't exist!")
        exit(1)

    if not os.path.isdir(args.destFolder):
        print("Folder '", args.destFolder, "' doesn't exist!")
        exit(1)

    decimate_images(args.sourceFolder, args.destFolder, args.threshold)


if __name__ == "__main__":
    main()