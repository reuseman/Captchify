from os import path


class Commons:
    """Settings that the software uses"""

    # Paths
    current_path = path.dirname(path.abspath(__file__))
    dataset_path = path.join(current_path, "dataset")
    histrory_path = path.join(current_path, "history")
    trained_model_path = path.join(current_path, "trained_models")
    trained_svm = path.join(trained_model_path, 'svm')
    trained_nn = path.join(trained_model_path, 'nn')

    original_dataset_path = path.join(dataset_path, "original")
    grayscale_dataset_path = path.join(dataset_path, "grayscale")
    binarized_dataset_path = path.join(dataset_path, "binarized")
    segmented_dataset_path = path.join(dataset_path, "segmented")
    pickle_dataset_path = path.join(dataset_path, "pickle")

    # Parameters
    segments_number = 4
    resize_resolution = (50, 50)
    anti_aliasing = True

    # GUI
    models = ['Neural networks relu', 'Neural networks tanh', 'Support vector machine']
    label_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h',
                 8: 'm', 9: 'n', 10: 'o', 11: 'p', 12: 'q', 13: 'r', 14: 's', 15: 't', 16: 'w', 17: 'y', 18: 'z'}
