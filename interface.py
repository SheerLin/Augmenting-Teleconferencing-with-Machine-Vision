import logging
import os
import signal

from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QColor, QCursor
from PyQt5.QtWidgets import *

import main
import undistortion

# Options for profile selection
NO_UNDISTORTION = "NO_UNDISTORTION"
DEFAULT_PROFILE = "DEFAULT_PROFILE"
SELECT_ONE_PROFILE = "SELECT_ONE_PROFILE"

# Options for profile displayed in SELECT_ONE_PROFILE
ALL_PROFILES = "ALL_PROFILES"
PROFILES_ALL_ATTACHED_CAM = "PROFILES_ALL_ATTACHED_CAM"
PROFILES_CUR_CAM_ONLY = "PROFILES_CUR_CAM_ONLY"


class MainWindow(QWidget):

    def __init__(self, profiles_map, args):
        super(MainWindow, self).__init__(flags=Qt.Widget)
        self.left = 10
        self.top = 10
        self.width = 639
        
        self.layout = None
        self.list_widget = None
        self.profile_list_widget = list()
        self.video_list = list()
        
        self.all_profiles_map = profiles_map  # Dictionary:<device> => set( (pair of img_path, obj_path) )
        self.input_video = main.CAM_DEVICE_NUMBER
        self.output_video = main.CAP_DEVICE_NUMBER
        self.resolution = main.RESOLUTION
        self.enable_virtual_cam = main.ENABLE_VIRTUAL_CAM
        self.enable_undistortion = main.ENABLE_UNDISTORTER
        self.enable_beautifier = main.ENABLE_BEAUTIFIER
        self.debug = True
        self.benchmark = False
        
        self.cam_device = None
        self.cap_device = None
        self.child_pid = -1
        self.selected_profile_pair = None  # The option for selecting one from all profiles
        self.final_profile_pair = None  # The final decision of selected profile
        self.logger = logging.getLogger("ATCV")
        
        self.resize(self.width, self.sizeHint().height())
        self.init_interface()

    def init_interface(self):
        if not self.layout:
            self.layout = QVBoxLayout()
            self.setLayout(self.layout)

        MainWindow.delete_list_widget(self.list_widget)
        self.list_widget = list()
        self.enable_undistortion = False
        self.video_list = undistortion.UndistortionPreProcessor.get_videos_list()

        # Add input and output source
        self.add_input_output_source()
        self.input_video = max(self.input_video, 0)
        self.output_video = max(self.output_video, 0)

        # Add radio buttons for profile selection
        self.add_title_label("Undistorter selection")
        radiobutton = QRadioButton("Not to use undistortion feature")
        radiobutton.setChecked(True)
        radiobutton.status = NO_UNDISTORTION
        self.add_level1_radio_button(radiobutton)

        radiobutton = QRadioButton("Use Default profile")
        radiobutton.status = DEFAULT_PROFILE
        self.add_level1_radio_button(radiobutton)

        radiobutton = QRadioButton("Select a specific profile")
        radiobutton.status = SELECT_ONE_PROFILE
        self.add_level1_radio_button(radiobutton)

        # Add Run button
        self.add_run_button()

        MainWindow.add_list_widget(self.list_widget, self.layout)

        self.construct_profile_layout()
        self.auto_resize()

    def add_input_output_source(self):
        input_label = self.add_title_label("Input Device (camera)")
        input_combo_box = QComboBox()
        input_combo_box.addItems(self.video_list)
        input_combo_box.currentTextChanged.connect(self.change_input_source)

        try:
            self.input_video = int(input_combo_box.currentText().strip("video"))
        except ValueError:
            self.input_video = 0
        self.logger.debug('Initialized input source: video{}'.format(self.input_video))

        input_label.setBuddy(input_combo_box)
        self.list_widget.append(input_combo_box)

        output_label = self.add_title_label("Output Device (virtual camera)")
        output_combo_box = QComboBox()
        output_combo_box.addItems(self.video_list)
        output_combo_box.currentTextChanged.connect(self.change_output_source)

        try:
            self.output_video = int(output_combo_box.currentText().strip("video"))
        except ValueError:
            self.output_video = 0
        self.logger.debug('Initialized output source: video{}'.format(self.output_video))

        output_label.setBuddy(output_combo_box)
        self.list_widget.append(output_combo_box)

    def change_input_source(self, text: str):
        if text and "video" in text:
            self.input_video = int(text.strip("video"))
            self.logger.debug("Update input source: video{}".format(self.input_video))

    def change_output_source(self, text: str):
        if text and "video" in text:
            self.output_video = int(text.strip("video"))
            self.logger.debug("Update output source: video{}".format(self.output_video))

    def add_title_label(self, text: str):
        label = QLabel(text)
        self.list_widget.append(label)
        return label

    def add_run_button(self):
        run_button = QPushButton("Run")
        run_button.clicked.connect(self.on_click_run)
        run_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.list_widget.append(run_button)

    def add_rerun_button(self):
        self.list_widget = list()
        rerun_button = QPushButton("Rerun")
        rerun_button.clicked.connect(self.kill_main_process_video)
        rerun_button.clicked.connect(self.init_interface)
        rerun_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.list_widget.append(rerun_button)
        MainWindow.add_list_widget(self.list_widget, self.layout)

    @staticmethod
    def add_list_widget(list_widget, layout):
        if list_widget and len(list_widget) > 0:
            for cur_profile_list_widget in list_widget:
                layout.addWidget(cur_profile_list_widget, alignment=Qt.AlignLeft)

    @staticmethod
    def delete_list_widget(list_widget):
        if list_widget and len(list_widget) > 0:
            for cur_profile_list_widget in list_widget:
                cur_profile_list_widget.setParent(None)

    def add_level1_radio_button(self, button: QRadioButton):
        button.toggled.connect(self.on_click_radio)
        button.setCursor(QCursor(Qt.PointingHandCursor))
        self.list_widget.append(button)

    def on_click_radio(self):
        radio_button = self.sender()
        if radio_button.isChecked():
            self.logger.debug("Undistorter selection is:{}".format(radio_button.status))

            if radio_button.status == SELECT_ONE_PROFILE:
                MainWindow.add_list_widget(self.profile_list_widget, self.layout)
                self.final_profile_pair = self.selected_profile_pair
            else:
                MainWindow.delete_list_widget(self.profile_list_widget)

                if radio_button.status == DEFAULT_PROFILE:
                    self.final_profile_pair = undistortion.get_default_profile_pair()
                    self.logger.debug("Select default profile:{}".format(self.final_profile_pair))

            if radio_button.status == NO_UNDISTORTION:
                self.enable_undistortion = False
            else:
                self.enable_undistortion = True

        self.auto_resize()

    def auto_resize(self):
        self.resize(self.width, self.sizeHint().height())
        self.setMaximumHeight(self.sizeHint().height())

    def on_click_run(self):
        img_path = None
        obj_path = None
        if self.final_profile_pair:
            img_path, obj_path = self.final_profile_pair
            img_path += undistortion.npy_file_postfix
            obj_path += undistortion.npy_file_postfix

        self.child_pid = os.fork()
        if self.child_pid == 0:
            self.cam_device, self.cap_device, frame_width, frame_height \
                = main.configure_devices({
                    'inp': self.input_video,
                    'out': self.output_video,
                    'res': self.resolution,
                    'vcam': self.enable_virtual_cam,
                })

            try:
                main.process_video(
                    self.cam_device, self.cap_device,
                    frame_width, frame_height, 
                    img_path, obj_path, 
                    {
                        'vcam': self.enable_virtual_cam,
                        'undistorter': self.enable_undistortion,
                        'beautifier': True,
                        'benchmark': False,
                        'debug': self.debug,
                    }
                )
            except Exception as e:
                self.kill_main_process_video()
                self.logger.error("Exception when processing frame: {}".format(e))
        elif self.child_pid < 0:
            self.logger.error("Error in forking a new process")
        else:
            MainWindow.delete_list_widget(self.list_widget)
            MainWindow.delete_list_widget(self.profile_list_widget)
            self.add_rerun_button()
            self.auto_resize()

    def construct_profile_layout(self):
        if not self.profile_list_widget:
            self.profile_list_widget = list()

            layout = QVBoxLayout()
            cur_box = QGroupBox()
            set_default = False
            for cur_device, cur_tuple_set in self.all_profiles_map.items():
                device_name = undistortion.UndistortionPreProcessor.get_usb_device(device_id=cur_device)
                if device_name and len(device_name) > 0:
                    cur_device_label = QLabel(device_name)
                else:
                    cur_device_label = QLabel(cur_device + "(Detached)")

                layout.addWidget(cur_device_label, alignment=Qt.AlignLeft)

                for cur_tuple in cur_tuple_set:
                    radio_button = QRadioButton(str(cur_tuple[0]))
                    radio_button.pair = (cur_tuple[1], cur_tuple[2])
                    radio_button.toggled.connect(self.on_select_pair)

                    if not set_default:
                        radio_button.setChecked(True)
                        self.selected_profile_pair = radio_button.pair
                        set_default = True
                    else:
                        radio_button.setChecked(False)

                    layout.addWidget(radio_button, alignment=Qt.AlignLeft)

            cur_box.setLayout(layout)

            self.profile_list_widget.append(cur_box)

    def on_select_pair(self):
        radio_button = self.sender()
        if radio_button.isChecked():
            self.logger.debug("Selected profile:{}".format(radio_button.pair))
            self.selected_profile_pair = radio_button.pair
            self.final_profile_pair = self.selected_profile_pair
            return radio_button.pair

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
            self.kill_main_process_video()

    def kill_main_process_video(self):
        if self.child_pid > 0:
            os.kill(self.child_pid, signal.SIGKILL)

        if self.cam_device:
            del self.cam_device

        if self.cap_device:
            self.cap_device.close()


def initialize_ui(profiles_map, args):
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

    window = MainWindow(profiles_map, args)
    window.show()

    app.exec_()
