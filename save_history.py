import os
import traceback
from distutils.dir_util import copy_tree
from shutil import copy, rmtree

from commons import Commons


def save_history(description=None):
    dataset = Commons.dataset_path
    history = Commons.histrory_path

    count = len(os.listdir(Commons.histrory_path)) + 1
    directory_path = os.path.join(history, str(count))
    while os.path.exists(directory_path):
        count += 1
        directory_path = os.path.join(history, str(count))
    os.mkdir(directory_path)

    modelpath = Commons.trained_model_path
    models_list_path = [os.path.join(modelpath, model)
                        for model in os.listdir(modelpath)]

    timestamp_list = [os.path.getmtime(m) for m in models_list_path]
    max_timestamp_index = timestamp_list.index(max(timestamp_list))
    modelpath = models_list_path[max_timestamp_index]

    try:
        copy_tree(dataset, directory_path)
        copy(modelpath, directory_path)
    except:
        rmtree(directory_path)
        traceback.print_exc()

    if description != None:
        with open(os.path.join(directory_path, 'description'), 'w') as file:
            for line in description:
                file.write(line)


if __name__ == "__main__":
    descr = input('Insert a description of the trained model and dataset: \n')
    save_history(description=descr)
