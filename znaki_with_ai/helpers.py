# Helper functions
# Файл содержит вспомогательные функции и не предназначен для редактирования
import cv2
import os
import glob  # library for loading images from a directory
import eval


# This function loads in images and their labels and places them in a list
# The list contains all images and their associated labels
# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list
def load_dataset(image_dir):
    # Populate this empty image list
    im_list = []
    image_types = ["no entry", "pedestrian crossing", "road works",
                   "movement prohibition", "parking", "stop",
                   "give way", "artificial roughness"]

    # Iterate through each color folder
    for im_type in image_types:

        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):

            # Read in the image
            # im = mpimg.imread(file)
            im = cv2.imread(file)

            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                im_list.append((im, eval.one_hot_encode(im_type)))

    return im_list


