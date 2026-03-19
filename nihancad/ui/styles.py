"""NihanCAD dark theme — VS Code meets AutoCAD aesthetic."""

COLORS = {
    'bg': '#1a1b2e',
    'surface': '#252640',
    'surface_hover': '#2f3052',
    'border': '#3d3e5c',
    'text': '#e2e4f0',
    'muted': '#8b8fa8',
    'accent': '#e94560',
    'accent_hover': '#f25a73',
    'accent_30': 'rgba(233, 69, 96, 0.3)',
    'selection': 'rgba(233, 69, 96, 0.25)',
    'success': '#5bd88a',
    'warning': '#f5a623',
    'info': '#22c7e8',
}

DARK_THEME = """
/* ===== NihanCAD Dark Theme ===== */

QMainWindow {
    background-color: #1a1b2e;
    color: #e2e4f0;
}

QMainWindow::separator {
    background: #3d3e5c;
    width: 2px;
    height: 2px;
}

QMainWindow::separator:hover {
    background: #e94560;
}

/* ===== Menu Bar ===== */

QMenuBar {
    background-color: #1a1b2e;
    color: #e2e4f0;
    border-bottom: 1px solid #3d3e5c;
    padding: 2px 0;
    font-size: 12px;
}

QMenuBar::item {
    padding: 4px 10px;
    border-radius: 4px;
    margin: 1px 2px;
}

QMenuBar::item:selected {
    background-color: #2f3052;
}

QMenuBar::item:pressed {
    background-color: rgba(233, 69, 96, 0.3);
}

QMenu {
    background-color: #252640;
    color: #e2e4f0;
    border: 1px solid #3d3e5c;
    border-radius: 6px;
    padding: 4px 0;
}

QMenu::item {
    padding: 6px 28px 6px 20px;
    margin: 1px 4px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: rgba(233, 69, 96, 0.3);
    color: #ffffff;
}

QMenu::item:disabled {
    color: #8b8fa8;
}

QMenu::separator {
    height: 1px;
    background: #3d3e5c;
    margin: 4px 8px;
}

QMenu::shortcut {
    color: #8b8fa8;
}

/* ===== Toolbar ===== */

QToolBar {
    background-color: #1a1b2e;
    border-bottom: 1px solid #3d3e5c;
    padding: 3px 6px;
    spacing: 4px;
}

QToolBar::separator {
    width: 1px;
    margin: 4px 6px;
    background: #3d3e5c;
}

QToolButton {
    background-color: transparent;
    color: #e2e4f0;
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 500;
    min-width: 36px;
}

QToolButton:hover {
    background-color: #2f3052;
    border-color: #3d3e5c;
}

QToolButton:pressed {
    background-color: rgba(233, 69, 96, 0.3);
}

QToolButton:checked {
    background-color: #e94560;
    color: #ffffff;
    border-color: #e94560;
}

QToolButton:checked:hover {
    background-color: #f25a73;
    border-color: #f25a73;
}

/* ===== Dock Widgets ===== */

QDockWidget {
    color: #e2e4f0;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}

QDockWidget::title {
    background-color: #252640;
    color: #e2e4f0;
    border: 1px solid #3d3e5c;
    border-bottom: 2px solid #e94560;
    padding: 6px 10px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}

QDockWidget::close-button,
QDockWidget::float-button {
    background: transparent;
    border: none;
    padding: 2px;
}

QDockWidget::close-button:hover,
QDockWidget::float-button:hover {
    background-color: #2f3052;
    border-radius: 3px;
}

/* ===== List Widget ===== */

QListWidget {
    background-color: #1a1b2e;
    color: #e2e4f0;
    border: none;
    outline: none;
    font-size: 12px;
}

QListWidget::item {
    padding: 6px 8px;
    border-bottom: 1px solid #252640;
    border-radius: 0;
}

QListWidget::item:hover {
    background-color: #2f3052;
}

QListWidget::item:selected {
    background-color: rgba(233, 69, 96, 0.25);
    color: #ffffff;
    border-left: 3px solid #e94560;
}

/* ===== Labels ===== */

QLabel {
    color: #e2e4f0;
    font-size: 12px;
}

QLabel[class="muted"] {
    color: #8b8fa8;
}

QLabel[class="header"] {
    font-size: 13px;
    font-weight: 600;
    color: #e94560;
}

/* ===== Line Edit ===== */

QLineEdit {
    background-color: #1a1b2e;
    color: #e2e4f0;
    border: 1px solid #3d3e5c;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
    selection-background-color: rgba(233, 69, 96, 0.3);
}

QLineEdit:focus {
    border-color: #e94560;
}

QLineEdit:hover {
    border-color: #8b8fa8;
}

/* ===== Buttons ===== */

QPushButton {
    background-color: #252640;
    color: #e2e4f0;
    border: 1px solid #3d3e5c;
    border-radius: 5px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #2f3052;
    border-color: #8b8fa8;
}

QPushButton:pressed {
    background-color: rgba(233, 69, 96, 0.3);
}

QPushButton:disabled {
    color: #8b8fa8;
    background-color: #1a1b2e;
    border-color: #252640;
}

QPushButton[class="accent"] {
    background-color: #e94560;
    color: #ffffff;
    border-color: #e94560;
}

QPushButton[class="accent"]:hover {
    background-color: #f25a73;
}

/* ===== Sliders ===== */

QSlider::groove:horizontal {
    height: 4px;
    background: #3d3e5c;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #e2e4f0;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}

QSlider::handle:horizontal:hover {
    background: #e94560;
}

QSlider::sub-page:horizontal {
    background: #e94560;
    border-radius: 2px;
}

/* ===== Checkboxes ===== */

QCheckBox {
    color: #e2e4f0;
    spacing: 6px;
    font-size: 12px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #3d3e5c;
    border-radius: 3px;
    background-color: #1a1b2e;
}

QCheckBox::indicator:hover {
    border-color: #8b8fa8;
}

QCheckBox::indicator:checked {
    background-color: #e94560;
    border-color: #e94560;
}

/* ===== Status Bar ===== */

QStatusBar {
    background-color: #1a1b2e;
    color: #9da1b8;
    border-top: 1px solid #3d3e5c;
    font-size: 11px;
    padding: 0;
}

QStatusBar::item {
    border: none;
}

QStatusBar QLabel {
    padding: 2px 8px;
    font-size: 11px;
    color: #9da1b8;
}

/* ===== Scrollbars ===== */

QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #4a4c6a;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #8b8fa8;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: transparent;
    height: 0;
}

QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background: #4a4c6a;
    border-radius: 4px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background: #8b8fa8;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background: transparent;
    width: 0;
}

/* ===== Splitter ===== */

QSplitter::handle {
    background: #3d3e5c;
}

QSplitter::handle:hover {
    background: #e94560;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

/* ===== Group Box ===== */

QGroupBox {
    color: #e2e4f0;
    border: 1px solid #3d3e5c;
    border-radius: 6px;
    margin-top: 8px;
    padding: 12px 8px 8px 8px;
    font-size: 11px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #e94560;
}

/* ===== Tooltips ===== */

QToolTip {
    background-color: #252640;
    color: #e2e4f0;
    border: 1px solid #3d3e5c;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
}

/* ===== Tab Widget (for tabified docks) ===== */

QTabBar::tab {
    background-color: #1a1b2e;
    color: #8b8fa8;
    border: 1px solid #3d3e5c;
    border-bottom: none;
    padding: 6px 14px;
    font-size: 11px;
    margin-right: 1px;
}

QTabBar::tab:selected {
    background-color: #252640;
    color: #e2e4f0;
    border-bottom: 2px solid #e94560;
}

QTabBar::tab:hover:!selected {
    background-color: #2f3052;
    color: #e2e4f0;
}

QTabWidget::pane {
    border: 1px solid #3d3e5c;
    background-color: #252640;
}

/* ===== Frame (used in panels) ===== */

QFrame[class="separator"] {
    background-color: #3d3e5c;
    max-height: 1px;
    margin: 4px 0;
}

QWidget[class="panel-content"] {
    background-color: #252640;
}
"""


def apply_theme(app):
    """Apply NihanCAD dark theme to a QApplication instance."""
    app.setStyleSheet(DARK_THEME)
    # Set palette-level defaults for widgets that don't fully respect QSS
    from PyQt6.QtGui import QPalette, QColor
    from PyQt6.QtCore import Qt
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor('#1a1b2e'))
    palette.setColor(QPalette.ColorRole.WindowText, QColor('#e2e4f0'))
    palette.setColor(QPalette.ColorRole.Base, QColor('#1a1b2e'))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor('#252640'))
    palette.setColor(QPalette.ColorRole.Text, QColor('#e2e4f0'))
    palette.setColor(QPalette.ColorRole.Button, QColor('#252640'))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor('#e2e4f0'))
    palette.setColor(QPalette.ColorRole.Highlight, QColor('#e94560'))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor('#ffffff'))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor('#252640'))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor('#e2e4f0'))
    app.setPalette(palette)
