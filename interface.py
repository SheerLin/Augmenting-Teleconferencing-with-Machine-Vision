from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPalette, QColor, QCursor
from PyQt5.QtCore import *

NO_UNDISTORTION = "NO_UNDISTORTION"
SELECT_CHESSBOARD_FOLDER = "SELECT_CHESSBOARD_FOLDER"
DEFAULT_PROFILE = "DEFAULT_PROFILE"
SELECT_ONE_PROFILE = "SELECT_ONE_PROFILE"


class MainWindow(QWidget):

    def __init__(self, profiles_map: dict):
        super(MainWindow, self).__init__(flags=Qt.Widget)
        self.left = 10
        self.top = 10
        self.width = 600
        self.height = 300
        self.profile_list_widget = list()
        self.selected_profile_pair = None
        self.layout = None
        self.available_profiles_map = profiles_map  # Dictionary:<device> => set( (pair of img_path, obj_path) )

        self.location_on_the_screen()

        self.set_buttons()

    def location_on_the_screen(self):
        self.setGeometry(self.left, self.top, self.width, self.height)

        # TODO - put in middle of the screen
        # ag = QDesktopWidget().availableGeometry()
        # sg = QDesktopWidget().screenGeometry()
        #
        # widget = self.geometry()
        # x = ag.width() - widget.width()
        # y = 2 * ag.height() - sg.height() - widget.height()
        # self.move(x, y)
        pass

    def set_buttons(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        radiobutton = QRadioButton("Not to use undistortion feature")
        radiobutton.setChecked(True)
        radiobutton.status = NO_UNDISTORTION
        self.add_button(layout, radiobutton)

        radiobutton = QRadioButton("Select a the folder for calibration pictures")
        radiobutton.status = SELECT_CHESSBOARD_FOLDER
        self.add_button(layout, radiobutton)

        radiobutton = QRadioButton("Use Default profile")
        radiobutton.status = DEFAULT_PROFILE
        self.add_button(layout, radiobutton)

        radiobutton = QRadioButton("Select a specific profile")
        radiobutton.status = SELECT_ONE_PROFILE
        self.add_button(layout, radiobutton)

        self.profile_list_widget = self.construct_profile_layout()
        self.layout = layout

    def add_profile_list_widget(self):
        if self.profile_list_widget and len(self.profile_list_widget) > 0:
            print("Add boxes")
            for cur_profile_list_widget in self.profile_list_widget:
                self.layout.addWidget(cur_profile_list_widget, alignment=Qt.AlignLeft)

    def delete_profile_list_widget(self):
        if self.profile_list_widget and len(self.profile_list_widget) > 0:
            print("Remove boxes")
            for cur_profile_list_widget in self.profile_list_widget:
                cur_profile_list_widget.setParent(None)

    def add_button(self, layout: QWidget, button: QRadioButton):
        button.toggled.connect(self.on_click_radio)
        button.setCursor(QCursor(Qt.PointingHandCursor))
        layout.addWidget(button, alignment=Qt.AlignLeft)

    def on_click_radio(self):
        radio_button = self.sender()
        if radio_button.isChecked():
            print("Status is :%s" % radio_button.status)

            if radio_button.status == SELECT_ONE_PROFILE:
                self.add_profile_list_widget()
            else:
                self.delete_profile_list_widget()

            if radio_button.status == SELECT_CHESSBOARD_FOLDER:
                selected_path = MainWindow.select_folder()
                print("selected_path:", selected_path)

    def construct_profile_layout(self):
        profile_list_widget = list()

        for cur_device, cur_pair_est in self.available_profiles_map.items():
            cur_box = QGroupBox(cur_device)
            layout = QVBoxLayout()
            # layout.addStretch(1)

            for cur_par in cur_pair_est:
                radio_button = QRadioButton(str(cur_par))
                radio_button.pair = cur_par
                radio_button.toggled.connect(self.on_select_pair)
                radio_button.setChecked(False)
                layout.addWidget(radio_button, alignment=Qt.AlignCenter)

            cur_box.setLayout(layout)

            profile_list_widget.append(cur_box)

        return profile_list_widget

    def on_select_pair(self):
        radio_button = self.sender()
        if radio_button.isChecked():
            print("Status is :%s" % str(radio_button.pair))
            self.selected_profile_pair = radio_button.pair
            return radio_button.pair

    def selecting_profiles(self):
        print("In selecting_profiles:self.available_profiles_map:", self.available_profiles_map)
        pass

    @staticmethod
    def select_folder():
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.DontUseCustomDirectoryIcons
        dialog = QFileDialog()
        dialog.setOptions(options)
        dialog.setFilter(dialog.filter() | QDir.Hidden)
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setDirectory(str('/'))

        if dialog.exec_() == QDialog.Accepted:
            path = dialog.selectedFiles()[0]  # returns a list
            return path
        else:
            return ''

    def closeEvent(self, e):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setText("Exiting the application")
        msg_box.setInformativeText("Confirm to exit?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msg_box.setDefaultButton(QMessageBox.Cancel)
        reply = msg_box.exec_()

        if reply & QMessageBox.Cancel:
            e.ignore()


def initialize_ui(cur_available_profiles_map: dict):
    app = QApplication([])

    # Force the style to be the same on all OSs:
    app.setStyle("Fusion")

    # Now use a palette to switch to dark colors:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    app.setApplicationName("Augmenting Teleconferencing")

    window = MainWindow(cur_available_profiles_map)
    window.show()

    app.exec_()


if __name__ == '__main__':
    # TODO(debug) - delete it
    available_profiles_map = dict()
    available_profiles_map["device1"] = set()
    available_profiles_map["device1"].add(("img_path1", "obj_path1"))
    available_profiles_map["device1"].add(("img_path2", "obj_path2"))

    available_profiles_map["device2"] = set()
    available_profiles_map["device2"].add(("img_path3", "obj_path3"))

    initialize_ui(available_profiles_map)
