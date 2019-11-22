from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPalette, QColor, QCursor
from PyQt5.QtCore import *
import main
import undistortion
import os
import signal
import subprocess

NO_UNDISTORTION = "NO_UNDISTORTION"
SELECT_CHESSBOARD_FOLDER = "SELECT_CHESSBOARD_FOLDER"
DEFAULT_PROFILE = "DEFAULT_PROFILE"
SELECT_ONE_PROFILE = "SELECT_ONE_PROFILE"


class MainWindow(QWidget):

    def __init__(self, profiles_map: dict, cam_device, cap_device=None, frame_width=1920, frame_height=1080):
        super(MainWindow, self).__init__(flags=Qt.Widget)
        self.left = 10
        self.top = 10
        self.width = 600
        self.height = 300
        self.profile_list_widget = list()
        self.selected_profile_pair = None
        self.layout = None
        self.all_profiles_map = profiles_map  # Dictionary:<device> => set( (pair of img_path, obj_path) )
        self.cam_device = cam_device
        self.cap_device = cap_device
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.do_undistortion = False
        self.child_pid = -1

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
        self.add_level1_radio_button(layout, radiobutton)

        radiobutton = QRadioButton("Select a the folder for calibration pictures")
        radiobutton.status = SELECT_CHESSBOARD_FOLDER
        self.add_level1_radio_button(layout, radiobutton)

        radiobutton = QRadioButton("Use Default profile")
        radiobutton.status = DEFAULT_PROFILE
        self.add_level1_radio_button(layout, radiobutton)

        radiobutton = QRadioButton("Select a specific profile")
        radiobutton.status = SELECT_ONE_PROFILE
        self.add_level1_radio_button(layout, radiobutton)
        # TODO - sub-options:
        #   1 - Display all profiles
        #   2 - Display all profiles for the devices attached to current computer
        #   3 - Display only the profiles for the camera that are currently in used

        # Next button
        next_button = QPushButton("Next")
        next_button.clicked.connect(self.on_click_next)
        next_button.setCursor(QCursor(Qt.PointingHandCursor))
        layout.addWidget(next_button, alignment=Qt.AlignLeft)

        # TODO(low) - can set this in self.construct_profile_layout()
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

    def add_level1_radio_button(self, layout: QWidget, button: QRadioButton):
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
                elif radio_button.status == DEFAULT_PROFILE:
                    self.selected_profile_pair = undistortion.get_default_profile_pair()
                    print("Select default profile:", self.selected_profile_pair)

            if radio_button.status == NO_UNDISTORTION:
                self.do_undistortion = False
            else:
                self.do_undistortion = True

    def on_click_next(self):
        img_path = None
        obj_path = None
        if self.selected_profile_pair:
            img_path, obj_path = self.selected_profile_pair

        self.child_pid = os.fork()
        if self.child_pid == 0:
            print("In child process, run main.process_video")
            main.process_video(cam_device=self.cam_device, cap_device=self.cap_device,
                               width=self.frame_width, height=self.frame_height, img_path=img_path,
                               obj_path=obj_path, do_undistort=self.do_undistortion)
        else:
            print("In main process:self.child_pid=", self.child_pid)

    def construct_profile_layout(self):
        profile_list_widget = list()

        for cur_device, cur_pair_est in self.all_profiles_map.items():
            cur_box = QGroupBox(cur_device)
            layout = QVBoxLayout()
            # layout.addStretch(1)

            for cur_par in cur_pair_est:
                radio_button = QRadioButton(str(cur_par))
                radio_button.pair = cur_par
                radio_button.toggled.connect(self.on_select_pair)
                radio_button.setChecked(False)
                layout.addWidget(radio_button, alignment=Qt.AlignLeft)

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
        print("In selecting_profiles:self.available_profiles_map:", self.all_profiles_map)
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
        else:
            e.accept()
            if self.child_pid:
                kill_main(self.child_pid)


def kill_main(pid):
    if pid:
        kill_command = ["kill", "-9", str(pid)]
        subprocess.Popen(kill_command)


def initialize_ui(all_profiles_map: dict, cam_device=None, cap_device=None, width=None, height=None):
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

    window = MainWindow(all_profiles_map, cam_device, cap_device, width, height)
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
