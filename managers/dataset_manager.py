import datetime
import logging
import pickle
import string
from os import listdir, path, makedirs, rename, remove

import matplotlib.image as mpimg
import numpy as np
from skimage.io import imread
from skimage.transform import resize

from commons import Commons
from managers.image_manager import ImageManager


def initialize():
    if path.exists(Commons.original_dataset_path) and \
            any(not file_name.endswith(".jpg") for file_name in listdir(Commons.original_dataset_path)):
        print("# Starting cleaning of original folder")
        simplify_original_dataset()
        print("     > Cleaning has finished")
    if not path.exists(Commons.grayscale_dataset_path):
        print("# Starting creation of folder with gray scale images")
        makedirs(Commons.grayscale_dataset_path)
        create_grayscale_dataset()
        print("     > Grayscale folder has been created")
    if not path.exists(Commons.binarized_dataset_path):
        print("# Starting creation of folder with binarized images")
        makedirs(Commons.binarized_dataset_path)
        create_binarized_dataset()
        print("     > Binzarized folder has been created")
    if not path.exists(Commons.segmented_dataset_path):
        print("# Starting creation of folders with segmented images")
        makedirs(Commons.segmented_dataset_path)
        for letter in list(string.ascii_lowercase):
            makedirs(path.join(Commons.segmented_dataset_path, letter))
        create_segmented_dataset()
        print("# Segmentation folders have been created")


def simplify_original_dataset():
    """
    The dataset is made up by a couple 0.138233284.png, 0.138233284 which is a text file that contains the solution.
    This method reads the first line of the text file and renames the .png file with it. Then the text file is
    removed
    """
    files = listdir(Commons.original_dataset_path)
    images = [file for file in files if file.endswith(".jpg")]
    for i in range(0, len(images)):
        image_name = images[i]
        image_path = path.join(Commons.original_dataset_path, image_name)
        solution_name = image_name[: -4]
        solution_file_path = path.join(Commons.original_dataset_path, solution_name)
        with open(solution_file_path) as solution_file:
            n_duplicates = ""
            i = 0
            captcha_solution = solution_file.readline().rstrip('\n')
            while path.exists(path.join(Commons.original_dataset_path, captcha_solution + n_duplicates + ".jpg")):
                i += 1
                n_duplicates = "({})".format(i)
            rename(image_path, path.join(Commons.original_dataset_path, captcha_solution + n_duplicates + ".jpg"))
        remove(solution_file_path)


def create_grayscale_image(input_path, output_path):
    """
    Takes in input an image and it saves the grayscale image in the specified output path
    :param input_path: where the image is located
    :param output_path: where the image will be saved
    """
    # Reading as gray scale and saving
    grayscale_image = imread(input_path, True)
    mpimg.imsave(output_path, grayscale_image, cmap="gray")


def create_grayscale_dataset():
    # Find the images in the directory
    files = listdir(Commons.original_dataset_path)
    images = [file for file in files if file.endswith(".jpg")]
    for image in images:
        # Defining input path and outputpath
        original_image_path = path.join(Commons.original_dataset_path, image)
        grayscale_image_path = path.join(Commons.grayscale_dataset_path, image)
        create_grayscale_image(original_image_path, grayscale_image_path)


def create_binarized_image(input_path, output_path):
    """
    Takes in input an image and it saves the binarized image in the specified output path.
    :param input_path: where the image is located
    :param output_path: where the image will be saved
    """
    # Converting and saving
    grayscale_image = imread(input_path, True)
    binarized_image = ImageManager.get_binarized_image(grayscale_image)
    mpimg.imsave(output_path, binarized_image, cmap="gray")


def create_binarized_dataset():
    # Find the images in the gray scale dataset
    images = listdir(Commons.grayscale_dataset_path)
    for image in images:
        # Defining input path and output path
        grayscale_image_path = path.join(Commons.grayscale_dataset_path, image)
        binarized_image_path = path.join(Commons.binarized_dataset_path, image)
        create_binarized_image(grayscale_image_path, binarized_image_path)


def create_segmented_dataset():
    LOG_FILENAME = 'dataset_captcha_solver.log'
    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    positive = 0
    negative = 0
    n_fixed_bigger_4 = 0
    average_error = 0
    failed_images = []
    # Find the images in the binarized scale dataset
    images = listdir(Commons.binarized_dataset_path)
    for image_name in images:
        image_file = imread(path.join(Commons.binarized_dataset_path, image_name), True)
        success, segmented_images, splitted = ImageManager.get_n_segments(image_file, Commons.segments_number,
                                                                          debug_image_name=image_name)
        if success:
            positive += 1
            for i, segment in enumerate(segmented_images):
                letter_folder_path = path.join(Commons.segmented_dataset_path, image_name[i])
                id_name = str(len(listdir(letter_folder_path)) + 1)
                segment = resize(segment, Commons.resize_resolution, anti_aliasing=Commons.anti_aliasing, cval=1)
                mpimg.imsave(path.join(letter_folder_path, id_name + ".png"), segment, cmap="gray")
                if splitted:
                    print("         [SAVED] in the folder {} as {}.png".format(image_name[i], id_name))
        else:
            failed_images.append((image_name, image_file))
            average_error += len(segmented_images)
            negative += 1
            print("[FAILED] {} with {} segments".format(image_name, len(segmented_images)))
    failed = []
    for n, f in failed_images:
        failed.append(n)
    print("Failed images >", failed)
    try:
        average = average_error / negative
    except ZeroDivisionError:
        average = "Division by zero"
    date = datetime.datetime.now()
    message = "Debug info on [" + str(date) + "]"
    logging.debug(message)
    message = "     positives " + str(positive)
    logging.debug(message)
    message = "     negatives " + str(negative)
    logging.debug(message)
    message = "     average number of fault segments " + str(average)
    logging.debug(message)
    message = "     fixed bigger than four " + str(n_fixed_bigger_4)
    logging.debug(message)
    logging.debug("------------------------------------------------------------------------------------------"
                  "-------------")
    logging.shutdown()
    print("POSITIVES", positive)
    print("NEGATIVES", negative)
    print("AVERAGE LENGTH OF FAULT SEGMENTS", average)
    print("FIXED BIGGER THAN FOUR", ImageManager.debug_fixed_bigger_than_4)


def split_dataset(label_map, trainset_percentage=80, testset_percentage=20, evaluationset_percentage=0, image_width=50,
                  image_height=50, save=True):
    if (trainset_percentage + testset_percentage + evaluationset_percentage) != 100:
        print("[ERROR] split percentage not equal to 100, using trainset 80\% test 20\%")
        trainset_percentage = 80
        testset_percentage = 20
        evaluationset_percentage = 0
    segmented_dataset_path = Commons.segmented_dataset_path
    dataset = np.ndarray((2256, image_width, image_height))
    # int_labels = np.ndarray((2256),dtype=np.dtype(np.uint8))
    int_labels = []
    label_list = listdir(segmented_dataset_path)
    # label map will contain a mapping from int to the string labels
    # like  {0 : 'a', 1: 'b', ... , ...}
    k = 0
    num_label = 0
    for i, label in enumerate(label_list):
        label_path = path.join(segmented_dataset_path, label)
        image_list = listdir(label_path)
        if len(image_list) == 0:
            continue

        for j, image_name in enumerate(image_list):
            image_path = path.join(label_path, image_name)
            image = imread(image_path, as_gray=True)
            image = resize(image, (image_width, image_height),
                           anti_aliasing=True, cval=1)
            dataset[k] = image
            int_labels.append(num_label)
            k += 1

        label_map[num_label] = label
        num_label += 1
    cutted_dataset = np.ndarray((k, image_width, image_height))
    cutted_dataset = np.array(dataset[0:k])
    labels_array = np.ndarray(k, dtype=np.dtype(np.uint8))
    labels_array = np.copy(np.array(int_labels))

    num_trainset = round(k * (trainset_percentage / 100))
    num_testset = round(k * (testset_percentage / 100))
    num_evalset = round(k * (evaluationset_percentage / 100))
    # shuffling let all the sets to have a randomly distributed
    # number of different letters
    # to make the same permutation in the image array and in the labels array
    # we use the same seed for shuffle, but we have to assert that the
    # lenght(1st-axes dimension) are equals

    np.random.seed(1919)
    np.random.shuffle(cutted_dataset)
    np.random.seed(1919)
    np.random.shuffle(int_labels)
    trainset = (
        np.array(cutted_dataset[0:num_trainset]), np.array(int_labels[0:num_trainset]))
    testset = (np.array(
        cutted_dataset[num_trainset: num_trainset + num_testset]),
               np.array(int_labels[num_trainset: num_trainset + num_testset]))
    evaluationset = (np.array(cutted_dataset[num_trainset + num_testset:k]),
                     np.array(
                         int_labels[num_trainset + num_testset: k]))
    if save:
        savepath = Commons.pickle_dataset_path
        if not path.exists(savepath):
            makedirs(savepath)
        if len(label_map) != 0:
            with open(path.join(savepath, 'label_map'), 'wb') as file:
                pickle.dump(label_map, file)
        if trainset[0].shape[0] != 0:
            with open(path.join(savepath, 'train_dataset'), 'wb') as file:
                pickle.dump(trainset, file)
        elif path.exists(path.join(savepath, 'train_dataset')):
            remove(path.join(savepath, 'train_dataset'))
        if testset[0].shape[0] != 0:
            with open(path.join(savepath, 'test_dataset'), 'wb') as file:
                pickle.dump(testset, file)
        elif path.exists(path.join(savepath, 'test_dataset')):
            remove(path.join(savepath, 'test_dataset'))
        if evaluationset[0].shape[0] != 0:
            with open(path.join(savepath, 'evaluation_dataset'), 'wb') as file:
                pickle.dump(evaluationset, file)
        elif path.exists(path.join(savepath, 'evaluation_dataset')):
            remove(path.join(savepath, 'evaluation_dataset'))
    return trainset, testset, evaluationset
