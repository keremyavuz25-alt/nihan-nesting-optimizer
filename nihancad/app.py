"""NihanCAD — Tekstil Kalip Goruntuluyeci
Usage: python -m nihancad.app [file.dxf|file.gem|file.gemx]
"""
import sys
import os
import traceback
import faulthandler

# Catch segfaults and write to crash log
_CRASH_LOG = os.path.join(os.path.dirname(__file__), '..', 'nihancad_crash.log')
faulthandler.enable(open(_CRASH_LOG, 'w'))


def _exception_hook(exc_type, exc_value, exc_tb):
    """Write unhandled exceptions to crash log before dying."""
    with open(_CRASH_LOG, 'a') as f:
        traceback.print_exception(exc_type, exc_value, exc_tb, file=f)
    traceback.print_exception(exc_type, exc_value, exc_tb)
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _exception_hook


def main():
    from PyQt6.QtWidgets import QApplication
    from nihancad.ui.styles import apply_theme
    from nihancad.ui.main_window import NihanCADWindow

    app = QApplication(sys.argv)
    app.setApplicationName('NihanCAD')
    app.setOrganizationName('Nihan')
    apply_theme(app)

    window = NihanCADWindow()
    window.show()

    # Open file from command line arg if provided
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if os.path.exists(filepath):
            window.open_file(filepath)

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
