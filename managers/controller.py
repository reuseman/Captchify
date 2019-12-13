from commons import Commons
from os import listdir, rename, remove, path as pathos
from train import predict
from train import train_nn, train_svm

import managers.dataset_manager as dmng

def predict_image(path, model):
    print("DEBUG: ", path, model)
    if model == 0 or model == 1:
        solution = predict(Commons.trained_nn, path, label_map=Commons.label_map, modeltype="nn")[0]
    else:
        solution = predict(Commons.trained_svm, path, label_map=Commons.label_map, modeltype="svm")[0]

    # model su range [0, 1, 2] rappresenta l'index di
    # Commons.models = ['Neural network relu', 'Neural network tanh', 'Support vector machine']
    return solution

def predict_folder(path, model):
    print("DEBUG: ", path, model)
    # Commons.models = ['Neural network relu', 'Neural network tanh', 'Support vector machine']

    if model == 0 or model == 1:
        solutions = predict(Commons.trained_nn, path, label_map=Commons.label_map, modeltype="nn")
    else:
        solutions = predict(Commons.trained_svm, path, label_map=Commons.label_map, modeltype="svm")

    files = listdir(path)
    images = [file for file in files if file.endswith(".jpg") or file.endswith(".png")]
    for i in range(0, len(images)):
        image_name = images[i]
        image_path = pathos.join(path, image_name)
        solution_name = solutions[i]
        format = image_name[-4:]
        solution_file_path = pathos.join(path, "predicted_" + solution_name)
        n_duplicates = ""
        i = 0
        while pathos.exists(pathos.join(solution_file_path + n_duplicates + format)):
            i += 1
            n_duplicates = "({})".format(i)
        rename(image_path, pathos.join(solution_file_path + n_duplicates + format))

def evaluate(type):
    pass

