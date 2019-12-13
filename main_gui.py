# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

import sys
from os import path

from PyQt5 import QtWidgets, QtGui, QtCore
from gui.captchify import Ui_Captchify

from commons import Commons
from managers import controller as ctrl
import managers.dataset_manager as dmng


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_Captchify()
        self.ui.setupUi(self)

        # Init data
        dmng.initialize()
        self.models = Commons.models
        self.solve_single = True
        self.current_image = None
        self.current_model = 0

        # Init gui
        self.ui.model_combo_box.addItems(self.models)

        # self.ui.mov = QtGui.QMovie("assets/ajax-loader.gif")
        # self.ui.captcha_label.setMovie(self.ui.mov)
        # self.ui.mov.start()

    def set_mode(self):
        self.ui.captcha_label.setText("cia")
        print("cia")

    def open_file_dialog(self):
        if self.solve_single:
            file_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', Commons.dataset_path,
                                                              "Images (*.png *.jpg)")
            file_name = path.basename(file_path[0])
            if file_path[0]:
                with open(file_path[0], 'rb') as f:
                    self.current_image = f.read()
                    pixmap = QtGui.QPixmap(file_path[0])

                    self.ui.captcha_image_label.setPixmap(pixmap)
                    self.ui.captcha_image_label.setMask(pixmap.mask())
                    self.ui.captcha_image_file_label.setText(file_name)

                self.ui.solution_line_edit.setText("Executing")
                solution = ctrl.predict_image(file_path[0], self.current_model)
                self.ui.solution_line_edit.setText(solution)
        else:
            files_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open file', Commons.dataset_path,
                                                              QtWidgets.QFileDialog.ShowDirsOnly |
                                                                    QtWidgets.QFileDialog.DontResolveSymlinks)
 
            self.ui.solution_line_edit.setText("Executing")
            ctrl.predict_folder(files_path, self.current_model)
            self.ui.solution_line_edit.setText("Finished, open folder to see solution")

            # x = QtWidgets.QDialog()
            # x.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowTitleHint | QtCore.Qt.CustomizeWindowHint)
            # x.exec()

    def reject(self):
        pass

    def change_model(self, index):
        self.model = self.models[index]
        self.current_model = index

    def set_mode(self, solve_single):
        self.solve_single = solve_single

        self.ui.captcha_image_label.setVisible(self.solve_single)

        file_label = "img4893743.png" if solve_single else Commons.dataset_path
        solution_label = "fdsd" if solve_single else "Select a folder first"
        self.ui.captcha_image_file_label.setText(file_label)
        self.ui.solution_line_edit.setText(solution_label)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
