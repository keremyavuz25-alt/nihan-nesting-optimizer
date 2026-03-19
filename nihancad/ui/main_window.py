"""NihanCAD main window — assembles all UI components."""

import os

from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QFileDialog, QApplication, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence

from nihancad.ui.toolbar import CADToolBar
from nihancad.ui.piece_panel import PiecePanel
from nihancad.ui.properties_panel import PropertiesPanel
from nihancad.ui.layer_panel import LayerPanel


class NihanCADWindow(QMainWindow):
    """Main application window for NihanCAD."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('NihanCAD')
        self.resize(1400, 900)
        self._center_on_screen()

        self._current_file: str | None = None
        self._pieces: list = []
        self._canvas = None  # Set when graphics module is available

        self._build_menu_bar()
        self._build_toolbar()
        self._build_panels()
        self._build_status_bar()
        self._build_canvas()
        self._setup_shortcuts()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _center_on_screen(self):
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = (geo.width() - self.width()) // 2 + geo.x()
            y = (geo.height() - self.height()) // 2 + geo.y()
            self.move(x, y)

    def _build_menu_bar(self):
        menubar = self.menuBar()

        # -- Dosya --
        file_menu = menubar.addMenu('Dosya')
        open_action = QAction('Ac', self)
        open_action.setShortcut(QKeySequence('Ctrl+O'))
        open_action.triggered.connect(lambda: self.open_file())
        file_menu.addAction(open_action)

        export_action = QAction('PNG Olarak Kaydet', self)
        export_action.setShortcut(QKeySequence('Ctrl+E'))
        export_action.triggered.connect(self._export_png)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        quit_action = QAction('Cikis', self)
        quit_action.setShortcut(QKeySequence('Ctrl+Q'))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # -- Gorunum --
        view_menu = menubar.addMenu('Gorunum')
        fit_action = QAction('Tumu Sigdir', self)
        fit_action.setShortcut(QKeySequence('F'))
        fit_action.triggered.connect(self._fit_all)
        view_menu.addAction(fit_action)

        grid_action = QAction('Izgara', self)
        grid_action.setShortcut(QKeySequence('G'))
        grid_action.triggered.connect(self._toggle_grid)
        view_menu.addAction(grid_action)

        snap_action = QAction('Yakalama', self)
        snap_action.setShortcut(QKeySequence('S'))
        snap_action.triggered.connect(self._toggle_snap)
        view_menu.addAction(snap_action)

        # -- Araclar --
        tools_menu = menubar.addMenu('Araclar')
        sel_action = QAction('Sec', self)
        sel_action.triggered.connect(lambda: self._set_tool('select'))
        tools_menu.addAction(sel_action)

        pan_action = QAction('Pan', self)
        pan_action.triggered.connect(lambda: self._set_tool('pan'))
        tools_menu.addAction(pan_action)

        measure_action = QAction('Mesafe Olc', self)
        measure_action.setShortcut(QKeySequence('M'))
        measure_action.triggered.connect(lambda: self._set_tool('measure_dist'))
        tools_menu.addAction(measure_action)

        angle_action = QAction('Aci Olc', self)
        angle_action.setShortcut(QKeySequence('A'))
        angle_action.triggered.connect(lambda: self._set_tool('measure_angle'))
        tools_menu.addAction(angle_action)

        # -- Yardim --
        help_menu = menubar.addMenu('Yardim')
        shortcuts_action = QAction('Kisayollar', self)
        shortcuts_action.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_action)

    def _build_toolbar(self):
        self._toolbar = CADToolBar(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._toolbar)

    def _build_panels(self):
        # Left dock — Piece list
        self._piece_panel = PiecePanel(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._piece_panel)

        # Left dock — Properties (below piece panel)
        self._props_panel = PropertiesPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._props_panel)

        # Right dock — Layers
        self._layer_panel = LayerPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._layer_panel)

        # Set sensible dock sizes
        self._piece_panel.setMinimumWidth(220)
        self._piece_panel.setMaximumWidth(320)
        self._props_panel.setMinimumWidth(220)
        self._props_panel.setMaximumWidth(320)
        self._layer_panel.setMinimumWidth(240)
        self._layer_panel.setMaximumWidth(340)

    def _build_status_bar(self):
        sb = self.statusBar()

        self._status_coords = QLabel('X: 0.0  Y: 0.0 mm')
        self._status_coords.setMinimumWidth(160)
        sb.addWidget(self._status_coords)

        self._status_zoom = QLabel('Zum: 100%')
        self._status_zoom.setMinimumWidth(80)
        sb.addWidget(self._status_zoom)

        self._status_snap = QLabel('Snap: Kapali')
        self._status_snap.setMinimumWidth(100)
        sb.addWidget(self._status_snap)

        self._status_tool = QLabel('Arac: Sec')
        self._status_tool.setMinimumWidth(90)
        sb.addWidget(self._status_tool)

        self._status_count = QLabel('0 Parca')
        sb.addPermanentWidget(self._status_count)

    def _build_canvas(self):
        """Try to import and set up the CADCanvas. Falls back to placeholder."""
        try:
            from nihancad.graphics.canvas import CADCanvas
            self._canvas = CADCanvas(self)
            self.setCentralWidget(self._canvas)
            self._canvas.mouse_moved.connect(self._on_mouse_moved)
            self._canvas.piece_selected.connect(self._on_piece_selected)
            self._canvas.zoom_changed.connect(self._on_zoom_changed)
            self._canvas.file_dropped.connect(self.open_file)
        except ImportError:
            # Graphics module not yet built — show placeholder
            placeholder = QLabel('NihanCAD\n\nDXF dosyasi yuklemek icin Ctrl+O')
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(
                'color: #8b8fa8; font-size: 18px; background-color: #1a1b2e;'
            )
            self.setCentralWidget(placeholder)

    def _setup_shortcuts(self):
        """Additional keyboard shortcuts not covered by menu actions."""
        # Escape — deselect
        esc = QAction(self)
        esc.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        esc.triggered.connect(self._deselect)
        self.addAction(esc)

        # Z — zoom to selection
        z_key = QAction(self)
        z_key.setShortcut(QKeySequence('Z'))
        z_key.triggered.connect(self._fit_selection)
        self.addAction(z_key)

        # + / - zoom
        plus_key = QAction(self)
        plus_key.setShortcut(QKeySequence(Qt.Key.Key_Plus))
        plus_key.triggered.connect(lambda: self._zoom_step(1.2))
        self.addAction(plus_key)

        minus_key = QAction(self)
        minus_key.setShortcut(QKeySequence(Qt.Key.Key_Minus))
        minus_key.triggered.connect(lambda: self._zoom_step(0.8))
        self.addAction(minus_key)

    def _connect_signals(self):
        # Toolbar signals
        self._toolbar.tool_changed.connect(self._set_tool)
        self._toolbar.action_triggered.connect(self._on_action)

        # Piece panel signals
        self._piece_panel.piece_clicked.connect(self._on_piece_selected)
        self._piece_panel.piece_double_clicked.connect(self._on_piece_double_clicked)

        # Layer panel signals
        self._layer_panel.visibility_changed.connect(self._on_layer_visibility)
        self._layer_panel.opacity_changed.connect(self._on_layer_opacity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_file(self, path: str | None = None):
        """Open a DXF, GEM, or GEMX file. Shows file dialog if path is None."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, 'Kalip Dosyasi Ac',
                os.path.expanduser('~\\Downloads'),
                'Kalip Dosyalari (*.dxf *.gem *.gemx);;DXF (*.dxf);;GEM (*.gem *.gemx);;Tum Dosyalar (*)',
            )
        if not path:
            return

        self._update_status(f'Yukleniyor: {os.path.basename(path)}...')
        QApplication.processEvents()

        try:
            ext = os.path.splitext(path)[1].lower()

            if ext == '.dxf':
                from nihancad.core.dxf_parser import DXFParser
                from nihancad.core.piece import build_pieces
                parser = DXFParser()
                parsed = parser.parse(path)
                pieces = build_pieces(parsed)
            elif ext in ('.gem', '.gemx'):
                from nihancad.core.gem_parser import GemParser
                parser = GemParser()
                pieces = parser.parse(path)
            else:
                QMessageBox.warning(self, 'Hata', f'Desteklenmeyen dosya formati: {ext}')
                return
        except ImportError as e:
            self._update_status(f'Hata: modul bulunamadi — {e}')
            return
        except Exception as e:
            QMessageBox.critical(self, 'Hata', f'Dosya okuma hatasi:\n{e}')
            return

        if not pieces:
            QMessageBox.warning(self, 'Uyari', 'Dosyada parca bulunamadi.')
            return

        self._current_file = path
        self._pieces = pieces
        self.setWindowTitle(f'NihanCAD \u2014 {os.path.basename(path)}')

        # Load into canvas
        if self._canvas:
            self._canvas.load_pieces(pieces)
            QTimer.singleShot(100, self._canvas.fit_all)

        # Load into panels
        self._piece_panel.load_pieces(pieces)
        self._props_panel.clear()

        # Update layer panel from canvas layer manager if available
        if self._canvas and hasattr(self._canvas, 'layer_manager'):
            try:
                layers = self._canvas.layer_manager.get_all()
                self._layer_panel.load_layers(layers)
            except Exception:
                pass  # Keep default layers

        # Update status
        self._status_count.setText(f'{len(pieces)} Parca')
        self._update_status(f'{len(pieces)} parca yuklendi')

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_piece_selected(self, piece_id: int):
        """Handle piece selection from canvas or piece panel."""
        self._piece_panel.set_selected(piece_id)

        if piece_id < 0:
            self._props_panel.clear()
            return

        # Find piece by id
        piece = next((p for p in self._pieces if p.id == piece_id), None)
        if piece:
            self._props_panel.update_piece(piece)

    def _on_piece_double_clicked(self, piece_id: int):
        """Zoom to piece on double click."""
        if self._canvas and piece_id >= 0:
            self._canvas.fit_to_piece(piece_id)

    def _on_mouse_moved(self, x: float, y: float):
        self._status_coords.setText(f'X: {x:.1f}  Y: {y:.1f} mm')

    def _on_zoom_changed(self, factor: float):
        pct = factor * 100
        self._status_zoom.setText(f'Zum: {pct:.0f}%')
        self._toolbar.set_zoom_display(pct)

    def _on_action(self, action: str):
        if action == 'open':
            self.open_file()
        elif action == 'fit_all':
            self._fit_all()
        elif action == 'fit_selection':
            self._fit_selection()
        elif action == 'grid':
            self._toggle_grid()
        elif action == 'snap':
            self._toggle_snap()
        elif action == 'export_png':
            self._export_png()
        elif action.startswith('zoom:'):
            try:
                pct = float(action.split(':')[1])
                if self._canvas:
                    self._canvas.set_zoom_percent(pct)
            except (ValueError, IndexError):
                pass

    def _on_layer_visibility(self, layer_id: str, visible: bool):
        if self._canvas:
            try:
                from nihancad.graphics.layers import LayerManager
                # Canvas should expose its layer manager
                if hasattr(self._canvas, 'layer_manager'):
                    self._canvas.layer_manager.set_visible(layer_id, visible)
            except ImportError:
                pass

    def _on_layer_opacity(self, layer_id: str, opacity: float):
        if self._canvas:
            try:
                if hasattr(self._canvas, 'layer_manager'):
                    self._canvas.layer_manager.set_opacity(layer_id, opacity)
            except ImportError:
                pass

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _set_tool(self, tool: str):
        self._toolbar.set_active_tool(tool)
        if self._canvas:
            self._canvas.set_tool(tool)
        tool_names = {
            'select': 'Sec',
            'pan': 'Pan',
            'measure_dist': 'Olcu',
            'measure_angle': 'Aci',
        }
        self._status_tool.setText(f'Arac: {tool_names.get(tool, tool)}')

    def _fit_all(self):
        if self._canvas:
            self._canvas.fit_all()

    def _fit_selection(self):
        if self._canvas:
            # Get current selection from piece panel
            current = self._piece_panel._list.currentItem()
            if current:
                pid = current.data(Qt.ItemDataRole.UserRole)
                if pid is not None:
                    self._canvas.fit_to_piece(pid)

    def _toggle_grid(self):
        if self._canvas:
            is_on = self._toolbar.is_toggled('grid')
            self._toolbar.set_toggle_state('grid', not is_on)
            self._canvas.set_grid_visible(not is_on)

    def _toggle_snap(self):
        is_on = self._toolbar.is_toggled('snap')
        self._toolbar.set_toggle_state('snap', not is_on)
        self._status_snap.setText(f'Snap: {"Acik" if not is_on else "Kapali"}')

    def _zoom_step(self, factor: float):
        if self._canvas:
            current = self._canvas.get_zoom_percent()
            self._canvas.set_zoom_percent(current * factor)

    def _export_png(self):
        if not self._canvas:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'PNG Olarak Kaydet', '', 'PNG Dosyasi (*.png)',
        )
        if path:
            self._canvas.export_png(path)
            self._update_status(f'PNG kaydedildi: {os.path.basename(path)}')

    def _deselect(self):
        self._piece_panel.set_selected(-1)
        self._props_panel.clear()
        if self._canvas:
            self._canvas.piece_selected.emit(-1)

    def _show_shortcuts(self):
        text = (
            'Klavye Kisayollari\n\n'
            'Ctrl+O    Dosya ac\n'
            'Ctrl+E    PNG olarak kaydet\n'
            'Ctrl+Q    Cikis\n'
            'F         Tumu sigdir\n'
            'Z         Secime zum\n'
            'G         Izgara ac/kapat\n'
            'S         Yakalama ac/kapat\n'
            'M         Mesafe olcum araci\n'
            'A         Aci olcum araci\n'
            'Escape    Secimi iptal et\n'
            '+/-       Yaklas/Uzaklas\n'
        )
        QMessageBox.information(self, 'Kisayollar', text)

    def _update_status(self, text: str, timeout: int = 3000):
        self.statusBar().showMessage(text, timeout)
