# About the project
This project was made by me and [Mario Parisi](https://gitlab.com/MaggioFG) for a university exam. More in the specific was for a Pattern Recognition exam. For others info and a complete documentation check [Captcha solver.pdf](https://github.com/reuseman/Captchify/blob/master/Captcha%20solver.pdf)

----

Mirror of the repository:
https://gitlab.com/reuseman/captcha-solver


The project used the version of python 3.6 and the current libraries:

matplotlib==3.0.2
scikit_image==0.14.2
tensorflow_gpu==1.12.0
numpy==1.13.3
joblib==0.13.1
scipy==1.2.0
Keras==2.2.4
PyQt5==5.12
skimage==0.0
scikit_learn==0.20.2
tensorflow==1.12.0

To give a brief overview on how to use this code, a description of how to use 
the main files is provided.

main_gui.py
Starts the qt5 gui, the model of ML can be choosed and then used to predict the
captcha. The model cannot be trained currently, and it will be improved in 
future version. The input can be an image and the solution will be written on the GUI.
Otherwise a folder can be selected and the solution will be written as the name 
of the files.

main_cli.py
Starts a cli that provides different operation that are quite sequential.
Indeed, the first operation is to split the dataset into train, test and evaluation set that will be used for the rest of operations.
The second operation is train model that asks for different parameters and then saves the model.
Indeed, the first operation is to split the dataset into train, test and evaluation
set that will be used for the rest of operations.
The second operation is train model that asks for different parameters and then
saves the model.
Then evaluate model provides the accuracy of the latest model on a specific set.
While kfold validation uses the kfold method to do the evaluation on all the dataset.
And finally solve captcha asks for a path of an image or a folder and will provide the solution.
The path can be absolute like /home/john/project/dataset/original or relative dataset/original if the current working directory is /home/john/project
And finally solve captcha asks for a path of an image or a folder and will 
provide the solution.
The path can be absolute like /home/john/project/dataset/original or 
relative dataset/original if the current working directory is /home/john/project
