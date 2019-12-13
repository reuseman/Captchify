import os
from glob import glob

import managers.dataset_manager as dmng
from commons import Commons
from train import train_nn, train_svm, predict, evaluate_model, kfold_nn, kfold_svm
import numpy as np


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


def initialize_dataset():
    dmng.initialize()


def get_number_input(prompt, message=None, accepted=None, error_message=None, minv=None, maxv=None, cast_function=int):
    if message:
        print(message)
    while True:
        try:
            answer = cast_function(input(prompt))
            if accepted or minv or maxv:
                valid = True
                if accepted:
                    valid = False if answer not in accepted else True

                if minv:
                    valid = False if answer < minv else True

                if maxv:
                    valid = False if answer > maxv else True

                if valid:
                    return answer
                else:
                    if error_message:
                        print(error_message)
            else:
                return answer
        except ValueError:
            print("@ You have to insert a number\n")


def train_model(trainset):
    if not trainset:
        print('@ Trainset not initialized, split the dataset first')
    else:
        valid_range = list(range(1, 4))
        err_msg = "@ Answer must be between 1 or 2 or 3\n"
        type = get_number_input("> ", "What model do you want to train?\n1 for NN-relu, 2 NN-tanh, 3 for SVM",
                                accepted=valid_range, error_message=err_msg)

        if type == 1 or type == 2:
            nn = "nn-relu" if type == 1 else "nn-tanh"

            nlayer_units = [(50, 50)]
            err_msg = "@ Value should be at least one\n"
            n_hidden_layers = get_number_input("> ", "How many hidden layers?", minv=1, error_message=err_msg)
            i = 1
            while n_hidden_layers > 0:
                n_hidden_layers -= 1
                neurons = get_number_input("> ", "How many neurons at level {}".format(i), minv=1,
                                           error_message=err_msg)
                nlayer_units.append(neurons)
                i += 1
            nlayer_units.append(19)
            print(nlayer_units)

            err_msg = "@ Value should be a positive integer\n"
            epochs = get_number_input("> ", "How many epochs?",
                                      minv=0, error_message=err_msg)

            train_nn(trainset, modelname=nn, epochs=epochs, nlayer_units=nlayer_units)
        else:
            valid_range = list(range(1, 4))
            err_msg = "@ Answer must be between 1 or 2 or 3\n"
            kernel = get_number_input("> ", "What kernel do you want to use?\n1 for linear, 2 for x, 3 for y",
                                      accepted=valid_range, error_message=err_msg)
            degree = 3
            if kernel == 1:
                kernel = "linear"
            elif kernel == 2:
                kernel = "rbf"
            else:
                kernel = "poly"
                err_msg = "@ Answer must be at least 1\n"
                degree = get_number_input("> ", "What is the degree of the polynomial?", minv=1, error_message=err_msg)

            err_msg = "@ Value should be a positive float\n"
            c = get_number_input("> ", "What C? (positive float)", minv=0, error_message=err_msg, cast_function=float)
            train_svm(trainset, kernel=kernel, C=c, degree=degree)

        path = Commons.trained_nn if type == 1 or type == 2 else Commons.trained_svm
        list_of_files = glob(os.path.join(path, "*"))
        latest_file = max(list_of_files, key=os.path.getctime)
        print("The name of the model created: {}".format(latest_file))


def view_models():
    i = 1
    for model in os.listdir(Commons.trained_nn):
        print("{} - {}".format(i, model))
        i += 1

    for model in os.listdir(Commons.trained_svm):
        print("{} - {}".format(i, model))
        i += 1


def evaluate_model_interface(trainset, testset, evaluationset):
    testset_id = get_number_input("> ", "What set do you want to use? Trainset(1), testset(2) or evaluationset(3)",
                                  accepted=[1, 2, 3], error_message="@ Value must be between 1 and 3")
    if testset_id == 1:
        if not trainset:
            print('@ Train set not initialized, split the dataset first')
        else:
            modeltype = get_number_input('> ',
                                         'which kind of trained model do you want to use? Neural network(1), Svm(2)',
                                         accepted=[1, 2])
            if modeltype == 1:
                acc = evaluate_model(trainset, Commons.trained_nn, modeltype='nn')
                print('last trained {model} accuracy on {dataset} is {acc:5.2f}%'.format(model='neural network',
                                                                                         dataset='trainset',
                                                                                         acc=100 * acc))
            else:
                acc = evaluate_model(trainset, Commons.trained_svm, modeltype='svm')
                print(
                    'last trained {model} accuracy on {dataset} is {acc:5.2f}%'.format(model='svm', dataset='trainset',
                                                                                       acc=100 * acc))

    elif testset_id == 2:
        if not testset:
            print('@ Test set not initialized, split the dataset first')
        else:
            modeltype = get_number_input('> ',
                                         'which kind of trained model do you want to use? Neural network(1), Svm(2)',
                                         accepted=[1, 2])
            if modeltype == 1:
                acc = evaluate_model(testset, Commons.trained_nn, modeltype='nn')
                print('last trained {model} accuracy on {dataset} is {acc:5.2f}%'.format(model='neural network',
                                                                                         dataset='testset',
                                                                                         acc=100 * acc))
            else:
                acc = evaluate_model(testset, Commons.trained_svm, modeltype='svm')
                print('last trained {model} accuracy on {dataset} is {acc:5.2f}%'.format(model='svm', dataset='testset',
                                                                                         acc=100 * acc))

    elif testset_id == 3:
        if not evaluationset:
            print('@ Evaluation set not initialized, split the dataset first')
        else:
            modeltype = get_number_input('> ',
                                         'which kind of trained model do you want to use? Neural network(1), Svm(2)',
                                         accepted=[1, 2])
            if modeltype == 1:
                acc = evaluate_model(evaluationset, Commons.trained_nn, modeltype='nn')
                print('last trained {model} accuracy on {dataset} is {acc:5.2f}%'.format(model='neural network',
                                                                                         dataset='evaluationset',
                                                                                         acc=100 * acc))
            else:
                acc = evaluate_model(evaluationset, Commons.trained_svm, modeltype='svm')
                print('last trained {model} accuracy on {dataset} is {acc:5.2f}%'.format(model='svm',
                                                                                         dataset='evaluationset',
                                                                                         acc=100 * acc))

    def evaluate_k_fold_validation():
        if not trainset:
            print('@ Trainset not initialized, split the dataset first')
        else:
            dataset = []
            dataset.append(np.append(
                np.append(trainset[0], testset[0], 0), evaluationset[0], 0))
            dataset.append(np.append(
                np.append(trainset[1], testset[1], 0), evaluationset[1], 0))

            valid_range = list(range(1, 4))
            err_msg = "@ Answer must be between 1 or 2 or 3\n"
            type = get_number_input("> ", "What model do you want to test?\n1 for NN-relu, 2 NN-tanh, 3 for SVM",
                                    accepted=valid_range, error_message=err_msg)

            if type == 1 or type == 2:
                nn = "nn-relu" if type == 1 else "nn-tanh"

                nlayer_units = [(50, 50)]
                err_msg = "@ Value should be at least one\n"
                n_hidden_layers = get_number_input("> ", "How many hidden layers?", minv=1, error_message=err_msg)
                i = 1
                while n_hidden_layers > 0:
                    n_hidden_layers -= 1
                    neurons = get_number_input("> ", "How many neurons at level {}".format(i), minv=1,
                                               error_message=err_msg)
                    nlayer_units.append(neurons)
                    i += 1
                nlayer_units.append(19)
                print(nlayer_units)

                err_msg = "@ Value should be a positive integer\n"
                epochs = get_number_input("> ", "How many epochs?",
                                          minv=0, error_message=err_msg)

                results = kfold_nn(dataset, modelname=nn, epochs=epochs, nlayer_units=nlayer_units)
                print("Accuracy average {}".format(results['train_score'].mean()))
                print("Test score average {}".format(results['test_score'].mean()))
            else:
                valid_range = list(range(1, 4))
                err_msg = "@ Answer must be between 1 or 2 or 3\n"
                kernel = get_number_input("> ", "What kernel do you want to use?\n1 for linear, 2 for x, 3 for y",
                                          accepted=valid_range, error_message=err_msg)
                degree = 3
                if kernel == 1:
                    kernel = "linear"
                elif kernel == 2:
                    kernel = "rbf"
                else:
                    kernel = "poly"
                    err_msg = "@ Answer must be at least 1\n"
                    degree = get_number_input("> ", "What is the degree of the polynomial?", minv=1,
                                              error_message=err_msg)

                err_msg = "@ Value should be a positive float\n"
                c = get_number_input("> ", "What C? (positive float)", minv=0, error_message=err_msg,
                                     cast_function=float)
                results = kfold_svm(dataset, kernel=kernel, C=c, degree=degree)
                print("Accuracy average {}".format(results['train_score'].mean()))
                print("Test score average {}".format(results['test_score'].mean()))




def solve_captcha():
    cwd = os.getcwd()
    print("\nCWD: {}".format(cwd))
    print("Insert the path to an image or to a folder (absolute or relative)")
    print("     - In the first case the solution will be printed")
    print("     - In the second case the image will be renamed with the solution")
    captcha_path = input("> ")

    # Transform it to absolute path
    if not os.path.isabs(captcha_path):
        captcha_path = os.path.join(cwd, captcha_path)

    # Check if exists
    if not os.path.exists(captcha_path):
        print("@ The inserted path does not exists")
    else:
        valid_range = list(range(1, 3))
        err_msg = "@ Answer must be 1 or 2\n"
        model = get_number_input("> ", "What model NN(1) or SVM(2)? The latest created will be used",
                                 accepted=valid_range,
                                 error_message=err_msg)
        path_model = Commons.trained_nn if model == 1 else Commons.trained_svm
        type_model = "nn" if model == 1 else "svm"

        if os.path.isdir(captcha_path):
            print("# Starting...")
            solutions = predict(Commons.trained_nn, captcha_path, label_map=Commons.label_map, modeltype="nn")
            files = os.listdir(captcha_path)
            images = [file for file in files if file.endswith(".jpg") or file.endswith(".png")]
            for i in range(0, len(images)):
                image_name = images[i]
                image_path = os.path.join(captcha_path, image_name)
                solution_name = solutions[i]
                format = image_name[-4:]
                solution_file_path = os.path.join(captcha_path, "predicted_" + solution_name)
                n_duplicates = ""
                i = 0
                while os.path.exists(os.path.join(solution_file_path + n_duplicates + format)):
                    i += 1
                    n_duplicates = "({})".format(i)
                os.rename(image_path, os.path.join(solution_file_path + n_duplicates + format))
            print("# Finished!")
        else:
            solution = predict(path_model, captcha_path, Commons.label_map, type_model)[0]
            print("SOLUTION: {}".format(solution))


def split_dataset():
    print("\nSplit the original dataset in train, test and evaluation set by specifying the percentage for every one")
    print("(ex. 80, 20, 0)")
    valid_range = list(range(0, 101))
    err_msg = "@ Answer must be between 0 and 100\n"
    trainperc = get_number_input("> ", "\nTrain", accepted=valid_range, error_message=err_msg)
    testperc = get_number_input("> ", "Test", accepted=valid_range, error_message=err_msg)
    evalperc = get_number_input("> ", "Evaluation", accepted=valid_range, error_message=err_msg)
    trainset, testset, evaluationset = dmng.split_dataset(Commons.label_map, trainset_percentage=trainperc,
                                                          testset_percentage=testperc,
                                                          evaluationset_percentage=evalperc)
    print("# Dataset generated")

    return trainset, testset, evaluationset


def main():
    initialize_dataset()
    # cls()
    trainset = None
    testset = None
    evaluationset = None
    answer = -1
    while answer != 0:

        print("\nWelcome to Captchify CLI!")
        print('[1] Split the dataset into train/test/evaluation set')
        print("[2] Train model")
        print("[3] View models")
        print("[4] Evaluate model")
        print('[5] Kfold validation')
        print("[6] Solve captcha")
        print("[0] Exit")

        answer = get_number_input("> ")
        if answer == 1:
            trainset, testset, evaluationset = split_dataset()
        elif answer == 2:
            train_model(trainset)
        elif answer == 3:
            view_models()
        elif answer == 4:
            evaluate_model_interface(trainset, testset, evaluationset)
        elif answer == 5:
            evaluate_k_fold_validation()
        elif answer == 6:
            solve_captcha()
        elif answer == 0:
            print("La revedere!")
        else:
            print("@ Command not available\n")


if __name__ == "__main__":
    main()
