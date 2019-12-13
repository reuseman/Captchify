from __future__ import absolute_import, division, print_function

# system interface library
import os
from copy import copy
from shutil import rmtree

import joblib
# from tensorflow import keras
import keras
import matplotlib.image as mpimg
# Helper libraries
import numpy as np
# TensorFlow and tf.keras
import tensorflow as tf
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from skimage.io import imread
from skimage.transform import resize
# scikit and scikit wrappers
from sklearn import svm
from sklearn.metrics import jaccard_similarity_score as jss, make_scorer
from sklearn.model_selection import KFold, cross_validate

# functions
import managers.dataset_manager as DatasetManager
from commons import Commons
from managers.image_manager import ImageManager


def builder_creator(modelname="nn-relu", nlayer_units=[(28, 28), 128, 10]):
    def new_model():
        return nn_model(modelname, nlayer_units)

    return new_model


def nn_model(modelname="nn-relu", nlayer_units=[(28, 28), 128, 10]):
    nn = keras.Sequential()

    nn.add(keras.layers.Flatten(input_shape=nlayer_units[0]))
    for i in range(1, len(nlayer_units)):
        if (i != (len(nlayer_units) - 1)):
            if modelname == "nn-relu":
                nn.add(keras.layers.Dense(
                    nlayer_units[i], activation=tf.nn.relu, kernel_initializer=keras.initializers.he_uniform()))
            elif modelname == "nn-tanh":
                nn.add(keras.layers.Dense(
                    nlayer_units[i], activation=tf.nn.tanh, kernel_initializer=keras.initializers.glorot_uniform()))

        else:
            nn.add(keras.layers.Dense(
                nlayer_units[i], activation=tf.nn.softmax))

    nn.compile(optimizer="adam",
               loss="categorical_crossentropy", metrics=['accuracy'])

    return nn


def train_nn(train_dataset, modelname="nn-relu", nlayer_units=[(50, 50), 128, 19], epochs=5, savepath=None):
    train_images = np.array(copy(train_dataset[0]))
    train_labels = np.array(copy(train_dataset[1]), dtype=np.dtype(np.uint8))
    seed = 1919
    np.random.seed(seed)

    nn = nn_model(modelname, nlayer_units)
    train_categorical = to_categorical(
        train_labels, num_classes=nlayer_units[-1], dtype='uint8')
    nn.fit(train_images, train_categorical, epochs=epochs)
    path = Commons.trained_nn if savepath == None else savepath
    os.makedirs(path, exist_ok=True)
    nn.save('{path}/{modelname}-{units}-{epochs}.h5'.format(
        path=path, modelname=modelname, units=nlayer_units, epochs=epochs))


def train_svm(train_dataset, kernel='linear', gamma='auto', C=1.0, class_weight=None, degree=3, savepath=None):
    n_samples, width, height = train_dataset[0].shape
    flattened_input = train_dataset[0].reshape((n_samples, width * height))
    svm_classifier = build_svm(kernel=kernel, gamma=gamma, C=C, class_weight=class_weight, degree=degree)
    svm_classifier.fit(flattened_input, train_dataset[1])
    path = Commons.trained_svm if savepath == None else savepath
    os.makedirs(path, exist_ok=True)
    joblib.dump(svm_classifier, '{path}/svm-{kernel}-{gamma}-{C}.joblib'.format(
        path=path, kernel=kernel, gamma=gamma, C=C))


def build_svm(kernel='linear', gamma='auto', C=1.0, class_weight=None, degree=3):
    return svm.SVC(kernel=kernel, gamma=gamma, C=C, class_weight=class_weight, degree=degree)


def evaluate_model(test_dataset, modelpath, modeltype):
    model = load_latest_model(modelpath)
    if modeltype.startswith('nn'):
        test_categorical = to_categorical(
            test_dataset[1], num_classes=19, dtype='uint8')
        loss, accuracy = model.evaluate(test_dataset[0], test_categorical)
    elif modeltype == 'svm':
        print('Evaluating svm with:')
        print(model.get_params())
        n_samples, width, height = test_dataset[0].shape
        flattened_input = test_dataset[0].reshape((n_samples, width * height))
        accuracy = model.score(flattened_input, test_dataset[1])

    return accuracy


def load_latest_model(modelpath):
    if not (os.path.exists(modelpath)):
        raise FileNotFoundError('No model found at path: {}'.format(modelpath))
    if os.path.isdir(modelpath):
        models_list_path = [os.path.join(modelpath, model)
                            for model in os.listdir(modelpath) if (model.endswith('.h5') or model.endswith('.joblib'))]
    if len(models_list_path) == 0:
        raise FileNotFoundError('No model found at path: {}'.format(modelpath))
    else:
        timestamp_list = [os.path.getmtime(m) for m in models_list_path]

        max_timestamp_index = timestamp_list.index(max(timestamp_list))
        modelpath = models_list_path[max_timestamp_index]
    if modelpath.endswith('.h5'):
        return keras.models.load_model(modelpath)
    elif modelpath.endswith('.joblib'):
        return joblib.load(modelpath)


def get_images(image_path):
    x = 1
    image_list = []
    if os.path.exists(image_path):
        if os.path.isdir(image_path):
            image_list = [image_name for image_name in os.listdir(image_path) if
                          (image_name.endswith('.jpg') or image_name.endswith('.png'))]

            if len(image_list) == 0:
                raise FileNotFoundError(
                    'No image found at path: {}'.format(image_path))
            else:
                full_path = [os.path.join(image_path, image_name)
                             for image_name in image_list]
                images = [imread(path, True) for path in full_path]

        elif (image_path.endswith('.jpg') or image_path.endswith('.png')):
            images = [imread(image_path, as_gray=True)]

        else:
            raise FileNotFoundError(
                'No image found at path: {}'.format(image_path))
    else:
        raise FileNotFoundError(
            'The specified path does not exists: {}'.format(image_path))

    return images


def predict_one(model, image, label_map, modeltype):
    image = resize(image, (50, 50),
                   anti_aliasing=True, cval=1)
    img = (np.expand_dims(image, 0))

    word = []

    if modeltype.startswith('nn'):
        prediction = model.predict(img)
        index = np.argmax(prediction)
    elif modeltype == 'svm':
        n_samples, width, height = img.shape
        flattened_input = img.reshape((n_samples, width * height))
        index = model.predict(flattened_input)[0]
    letter = label_map[index]
    word.append(letter)
    # Casting into string
    word_string = ''.join(word)
    return word_string


def predict(modelpath, image_path, label_map, modeltype):
    images = get_images(image_path)
    model = load_latest_model(modelpath)
    words = []
    os.makedirs('tmp', exist_ok=True)
    for image in images:
        # preprocessing routine
        binarized_image = ImageManager.get_binarized_image(image)
        mpimg.imsave(os.path.join('tmp', 'tmp.jpg'), binarized_image, cmap='gray')
        binarized_image = imread(os.path.join('tmp', 'tmp.jpg'), True)
        success, segmented_images, splitted = ImageManager.get_n_segments(binarized_image, 4)
        if not success:
            words.append('Segmentation failure')
        else:
            word = []
            for letter in segmented_images:
                letter_prediction = predict_one(model, letter, label_map, modeltype)
                word.append(letter_prediction)
            words.append(''.join(word))
    rmtree('tmp')

    # for every image
    # for image in images:
    #    for word in predict_one(model, image, label_map):
    #        words.append(word)
    return words


def kfold_nn(dataset, k=10, modelname="nn-relu", nlayer_units=[(50, 50), 128, 19], epochs=5, savepath=""):
    seed = 1919
    np.random.seed(seed)
    dataset_labels = np.array(copy(dataset[1]), dtype=np.dtype(np.uint8))
    build_function = builder_creator(modelname, nlayer_units)
    estimator = KerasClassifier(
        build_fn=build_function, epochs=epochs, verbose=1)
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    categorical = to_categorical(
        dataset_labels, num_classes=19, dtype='uint8')
    jaccard_scorer = make_scorer(jss)
    return cross_validate(estimator, dataset[0], y=categorical, cv=kfold)


def kfold_svm(dataset, k=10, kernel='linear', gamma='auto', C=1.0, class_weight=None, degree=3):
    seed = 1919
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    svm_clf = build_svm(kernel=kernel, gamma=gamma, C=C, class_weight=class_weight, degree=degree)
    print('Evaluating Svm with configured with the following parameters')
    print(svm_clf.get_params())
    n_samples, width, height = dataset[0].shape
    flattened_input = dataset[0].reshape((n_samples, width * height))
    # results = cross_val_score(svm_clf, flattened_input, y=dataset[1], cv=kfold, n_jobs=-1, pre_dispatch=8)
    return cross_validate(svm_clf, flattened_input, y=dataset[1], cv=kfold, scoring='accuracy', n_jobs=-1,
                          pre_dispatch=8)


if __name__ == "__main__":
    label_map_path = os.path.join(Commons.pickle_dataset_path, 'label_map')
    train_path = os.path.join(Commons.pickle_dataset_path, 'train_dataset')
    test_path = os.path.join(Commons.pickle_dataset_path, 'test_dataset')
    evaluation_path = os.path.join(
        Commons.pickle_dataset_path, 'evaluation_dataset')
    width = 50
    height = 50

    #    if not(os.path.exists(train_path) and os.path.exists(test_path)
    #           and os.path.exists(label_map_path) and os.path.exists(evaluation_path)):
    #
    #        trainset, testset, evalset = DatasetManager.split_dataset(
    #            Commons.label_map, image_height=height, image_width=width, trainset_percentage=70, evaluationset_percentage=20,testset_percentage=10)
    #
    #    else:
    #        print('Loading dataset from pickle file')
    #        if os.path.exists(label_map_path) and os.path.isfile(label_map_path):
    #            with open(label_map_path, 'rb') as file:
    #                label_map = pickle.load(file)
    #
    #        if os.path.exists(train_path) and os.path.isfile(train_path):
    #            with open(train_path, 'rb') as file:
    #                trainset = pickle.load(file)
    #
    #        if os.path.exists(test_path) and os.path.isfile(test_path):
    #            with open(test_path, 'rb') as file:
    #                testset = pickle.load(file)
    #
    #        if os.path.exists(evaluation_path) and os.path.isfile(evaluation_path):
    #            with open(evaluation_path, 'rb') as file:
    #                evalset = pickle.load(file)

    trainset, testset, evalset = DatasetManager.split_dataset(
        Commons.label_map, image_height=height, image_width=width, trainset_percentage=70, evaluationset_percentage=20,
        testset_percentage=10)

    #    model_name = 'nn-tanh'
    #    train_nn(trainset,modelname=model_name, epochs=5, nlayer_units=[
    #        (width, height), 128, 19])
    #
    #    train_svm(trainset, C=1.0, kernel='rbf')
    #
    acc = evaluate_model(
        test_dataset=trainset, modelpath=Commons.trained_svm, modeltype='svm')
    print("Restored svm, accuracy on train set: {:5.2f}%".format(100*acc))
    #
    #    acc = evaluate_model(
    #        test_dataset=evalset, modelpath=Commons.trained_svm, modeltype='svm')
    #    print("Restored svm, accuracy on test set: {:5.2f}%".format(100*acc))

    #    acc = evaluate_model(
    #        test_dataset=trainset, modelpath=Commons.trained_nn, modeltype='nn')
    #    print("Restored nn, accuracy on train set: {:5.2f}%".format(100*acc))
    #
    #    acc = evaluate_model(
    #        test_dataset=evalset, modelpath=Commons.trained_nn, modeltype='nn')
    #    print("Restored nn, accuracy on test set: {:5.2f}%".format(100*acc))

    #    print(Commons.label_map)
    #    words = predict(modelpath='trained_models/svm',
    #                    image_path='dataset/original/aaam.jpg', label_map=Commons.label_map, modeltype='svm')
    #    for word in words:
    #        print(word)
    dataset = []
    dataset.append(np.append(
        np.append(trainset[0], testset[0], 0), evalset[0], 0))
    dataset.append(np.append(
        np.append(trainset[1], testset[1], 0), evalset[1], 0))

    #    model_name = 'nn-tanh'
    #    results = kfold_nn(dataset, modelname=model_name,k=10 ,epochs=20, nlayer_units=[
    #                          (width, height), 128, 19])
    #    print(results['train_score'].mean())
    #    print(results['test_score'].mean())
    #    print("Baseline: %.2f%% (%.2f%%)" %
    #          (results.mean()*100, results.std()*100))
    #    k = 10
    #    seed = 1919
    #    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    #    svm_clf = build_svm(kernel='rbf')
    #    n_samples, width, height = dataset[0].shape
    #    flattened_input = dataset[0].reshape((n_samples, width*height))
    #    results = cross_val_score(svm_clf, flattened_input, y=dataset[1], cv=kfold, n_jobs=-1, pre_dispatch=8)

    results = kfold_svm(dataset, kernel='linear', class_weight='balanced', C=0.005)
    print(results['train_score'].mean())
    print(results['test_score'].mean())

#    print("Baseline: %.2f%% (%.2f%%)" %
#          (results.mean()*100, results.std()*100))
#
#   train_svm(trainset)
