import sys
from matplotlib import container
import numpy as np
import pandas as pd
import h5py

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QFileDialog, QGroupBox, QComboBox,
    QPushButton, QListWidget, QListWidgetItem, QDoubleSpinBox
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from find_and_plot_modes import (
    build_mode,
    plot_mode,
    find_eigenfrequencies
)

# ======================================================
# 4. MEASUREMENT-MAPS PRO KANTE
# ======================================================
MEASUREMENT_MAP = {
    "X": {
        "LEFT": [(8, 45, +1), (10, 33, +1), (12, 21, +1)],
        #"RIGHT": [(0, 34, -1), (3, 22, -1), (5, 46, -1)],
        #"BACK_RIGHT": [(6, 47, -1), (14, 35, -1), (16, 23, -1)],
    },
    "Y": {
        "FRONT_LEFT": [(7, 45, +1), (9, 33, +1), (11, 21, +1)],
        #"FRONT_RIGHT": [(1, 34, +1), (4, 22, -1), (12, 46, +1)],
        #"BACK_RIGHT": [(13, 47, -1), (15, 35, -1), (17, 23, -1)],
    }
}


class Plott_UEfkt_Modes(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Schwingungsanlyse")
        self.resize(1200, 800)

        # =========================
        # Datencontainer
        # =========================
        self.frequency = None
        self.node_ids = None
        self.edges = None
        self.tf_imag = None
        self.tf_mag = None
        self.kanalnamen = []
        self.nodes = {}

        # =========================
        # Zentrales Widget
        # =========================
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # =========================
        # Controls
        # =========================
        controls = self.create_controls()
        main_layout.addWidget(controls)

        # =========================
        # Matplotlib Canvas
        # =========================
        self.figure = Figure(figsize=(6, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        main_layout.addWidget(self.canvas, stretch=3)
        
        # Leeres Axes erstellen
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5, 0.5, 
            "Zum Start den gewünschten Button drücken", 
            ha='center', va='center', fontsize=14,
            wrap=True
        )
        ax.set_xticks([])  # keine x-Achse
        ax.set_yticks([])  # keine y-Achse
        ax.set_frame_on(False)  # optional: Rahmen aus
        self.canvas.draw()

        # =========================
        # Menü
        # =========================
        self.create_menus()
        self.statusBar().showMessage("Laden Sie das FFT-File")

    # ======================================================
    # MENÜ
    # ======================================================
    def create_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&Load File")

        open_fft = QAction("Öffne Üfkt FFT Imaginärteil File", self)
        open_fft.triggered.connect(self.open_fft_file)
        file_menu.addAction(open_fft)

        open_uefkt = QAction("Öffne Üfkt FFT Betrag File", self)
        open_uefkt.triggered.connect(self.open_uefkt_file)
        file_menu.addAction(open_uefkt)

        open_geom = QAction("Öffne &Geometry File", self)
        open_geom.triggered.connect(self.open_geometry_file)
        file_menu.addAction(open_geom)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    # ======================================================
    # CONTROLS
    # ======================================================
    def create_controls(self):
        main_box = QVBoxLayout()

        # ==========================
        # Übertragungsfunktion
        # ==========================
        tf_box = QGroupBox("Übertragungsfunktion")
        tf_layout = QVBoxLayout(tf_box)
        tf_layout.addWidget(QLabel("Analysebereich Max [Hz]:"))
        self.freq_input = QDoubleSpinBox()
        self.freq_input.setMinimum(0.0)
        self.freq_input.setMaximum(1e6)
        self.freq_input.setValue(0.0)
        self.freq_input.setSingleStep(1.0)
        tf_layout.addWidget(self.freq_input)

        # Plot-Button
        self.plot_tf_btn = QPushButton("Übertragungsfunktion plotten")
        self.plot_tf_btn.clicked.connect(self.plot_transfer_function)
        tf_layout.addWidget(self.plot_tf_btn)

        # Eigenfrequenzen-Button
        self.find_eigenfreqs_btn = QPushButton("Eigenfrequenzen bestimmen")
        self.find_eigenfreqs_btn.clicked.connect(
          lambda: (
            self.show_info("Laden Sie das Betrag-FFT-File.") 
            if self.tf_mag is None 
            else self.find_and_fill_eigenfreqs(
                np.mean(self.tf_mag, axis=1) if self.tf_mag.ndim > 1 else self.tf_mag,
                xmax=self.freq_input.value()
                )
            )
        )
        tf_layout.addWidget(self.find_eigenfreqs_btn)

        tf_layout.addWidget(QLabel("Gefundene Eigenfrequenzen:"))
        self.eigenfreq_list = QListWidget()
        self.eigenfreq_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        tf_layout.addWidget(self.eigenfreq_list)

        main_box.addWidget(tf_box)

        # ==========================
        # Mode Control
        # ==========================
        mode_box = QGroupBox("Mode Control")
        mode_layout = QVBoxLayout(mode_box)

        # Mode auswählen
        mode_layout.addWidget(QLabel("Mode auswählen:"))
        self.mode_combo = QComboBox()
        mode_layout.addWidget(self.mode_combo)

        # Richtung wählen
        mode_layout.addWidget(QLabel("Richtung:"))
        self.dir_combo = QComboBox()
        self.dir_combo.addItems(["X", "Y"])
        mode_layout.addWidget(self.dir_combo)

        # Kanal wählen
        mode_layout.addWidget(QLabel("Kanal:"))
        self.channel_combo = QComboBox()
        mode_layout.addWidget(self.channel_combo)

        # Mode-Plot Button
        self.plot_mode_btn = QPushButton("Mode plotten")
        self.plot_mode_btn.clicked.connect(self.plot_selected_mode_from_button)
        mode_layout.addWidget(self.plot_mode_btn)

        # Reset Button
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)
        mode_layout.addWidget(reset_btn)

        main_box.addWidget(mode_box)
        container = QWidget()
        container.setLayout(main_box)
        container.setFixedWidth(300)
        return container

    # ======================================================
    # View / Plot zurücksetzen
    # ======================================================
    def reset_view(self):
        self.figure.clear()
        self.canvas.draw()
    # ======================================================
    # Öffnen Übertragungsfunktion Betrag File
    # ======================================================
    def open_uefkt_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Übertragungsfunktion FFT Betrag",
            "",
            "TSV Files (*.tsv)"
        )
        if not path:
            return

        df = pd.read_csv(path, sep="\t", comment="#", decimal=",")
        self.frequency = df["frequency"].astype(float).values

        kanalnamen = [c for c in df.columns if "UEBERTRAGUNGSFUNKTION" in c]
        kanalnamen = sorted(kanalnamen, key=lambda s: int(s.split("_")[-1]))

        self.tf_mag = df[kanalnamen].astype(float).values
        self.kanalnamen = kanalnamen

        self.channel_combo.clear()
        self.channel_combo.addItems(self.kanalnamen)

        self.statusBar().showMessage(f"Betrag-FFT geladen ({len(self.kanalnamen)} Kanäle)", 5000)

    # ======================================================
    # GEOMETRIE LADEN
    # ======================================================
    def open_geometry_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Geometry HDF5", "", "HDF5 Files (*.h5)"
        )
        if not path:
            return

        with h5py.File(path, "r") as f:
            nodes = f["nodes"][:]
            self.edges = f["edges"][:]

        self.node_ids = nodes["id"]
        self.nodes = {
            "x": nodes["x"],
            "y": nodes["y"] if "y" in nodes.dtype.names else np.zeros(len(nodes)),
            "z": nodes["z"],
        }

        self.statusBar().showMessage("Geometry loaded")

    # ======================================================
    # Öffnen Übertragungsfunktion Imaginärteil File
    # ======================================================
    def open_fft_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Übertragungsfunktion FFT Imaginärteil",
            "",
            "TSV Files (*.tsv)"
        )
        if not path:
            return

        df = pd.read_csv(path, sep="\t", comment="#", decimal=",")
        self.frequency = df["frequency"].values

        kanalnamen = sorted(
            [c for c in df.columns if "Imaginaer" in c],
            key=lambda s: int(s.split("_")[-1])
        )

        self.tf_imag = df[kanalnamen].values
        self.statusBar().showMessage("Imaginär-FFT geladen", 5000)

    # ======================================================
    # Eigenfrequenzen bestimmen
    # ======================================================
    def find_and_fill_eigenfreqs(self, tf_mag, xmax=None):
        if not isinstance(tf_mag, np.ndarray):
            self.show_info("Fehler: tf_mag ist kein Array.")
            return

        if xmax is not None and xmax > 0:
            mask = self.frequency <= xmax
            freq = self.frequency[mask]
            tf_mag = tf_mag[mask]
        else:
            freq = self.frequency

        self.eigenfreqs, self.peak_indices = find_eigenfrequencies(freq, tf_mag)

        self.eigenfreq_list.clear()
        if hasattr(self, "mode_combo"):
            self.mode_combo.clear()
            for i, f in enumerate(self.eigenfreqs):
                self.mode_combo.addItem(f"Mode {i + 1}")

        for f in self.eigenfreqs:
            self.eigenfreq_list.addItem(f"{f:.2f} Hz")

    # ======================================================
    # Modus aus Measurement Map extrahieren
    # ======================================================
    @staticmethod
    def extract_mode_from_measurement_map(
        tf_imag,
        freq_idx,
        measurement_maps,
        node_ids,
        nodes,
        normalize=True,
    ):
        """
        Baut eine Modeform aus Imaginär-FFT über die Höhe auf
        """

        z_meas = []
        u_meas = []

        print("\n================ MODE DEBUG =================")
        print(f"Frequenzindex: {freq_idx}")

        # -------------------------
        # Messwerte sammeln
        # -------------------------
        print("\nMesspunkte (z, u_imag):")
        for measurement_map in measurement_maps:
            for meas_idx, node_id, sign in measurement_map:
                node_pos = np.where(node_ids == node_id)[0][0]
                z = nodes["z"][node_pos]
                u = sign * tf_imag[freq_idx, meas_idx]

                z_meas.append(z)
                u_meas.append(u)
                print(
                    f"  Node {node_id:>3} | "
                    f"z = {z:6.2f} | "
                    f"TF[{meas_idx}] = {u:+.4e}"
                )
        z_meas = np.asarray(z_meas)
        u_meas = np.asarray(u_meas)

        # -------------------------
        # Mode über Höhe aufbauen
        # -------------------------
        z_nodes = nodes["z"]
        u_all = build_mode(z_nodes, z_meas, u_meas)

        # -------------------------
        # Normieren
        # -------------------------
        if normalize:
            max_val = np.max(np.abs(u_all))
            if max_val > 0:
                u_all /= max_val

        # -------------------------
        # Knoten-Auslenkungen
        # -------------------------
        print("\nInterpolierte Knoten-Auslenkungen:")
        for nid, z, u in zip(node_ids, nodes["z"], u_all):
            print(f"  Node {nid:>3} | z = {z:6.2f} | u = {u:+.4f}")

        print("============================================\n")

        return u_all

    # ======================================================
    # Mode plotten
    # ======================================================
    def plot_selected_mode(self, mode_idx):

        if self.nodes is None or "z" not in self.nodes:
            self.show_info("Laden Sie das Geometry-File.")
            return

        if self.tf_imag is None:
            self.show_info("Laden Sie das Imaginär-FFT-File.")
            return

        if not hasattr(self, "peak_indices"):
            self.show_info("Eigenfrequenzen fehlen.")
            return

        freq_idx = self.peak_indices[mode_idx]

        direction = "X"  # "X" oder "Y"

        # alle Kanten dieser Richtung verwenden
        measurement_maps = list(MEASUREMENT_MAP[direction].values())

        u_all = self.extract_mode_from_measurement_map(
            tf_imag=self.tf_imag,
            freq_idx=freq_idx,
            measurement_maps=measurement_maps,
            node_ids=self.node_ids,
            nodes=self.nodes,
            normalize=True,
        )


        self.figure.clear()
        ax = self.figure.add_subplot(111)

        plot_mode(
            ax=ax,
            node_ids=self.node_ids,
            nodes=self.nodes,
            edges=self.edges,
            plot_lines=[("b", u_all)],
            axis_a="x",
            axis_b="z",
            disp_axis="x",  # "x" oder "y"
            scale=0.4,
        )

        ax.set_title(f"Mode {mode_idx + 1} ({direction})")
        self.canvas.draw()


    # ======================================================
    # Mode plotten von Button
    # ======================================================
    def plot_selected_mode_from_button(self):
        idx = self.mode_combo.currentIndex()
        if self.nodes is None or "z" not in self.nodes or self.node_ids is None or self.edges is None:
            self.show_info("Laden Sie das Geometry-File, um Modes zu plotten.")
            return
        if not hasattr(self, "peak_indices"):
            self.show_info("Die Eigenfrequenzen wurden noch nicht bestimmt.")
            return
        if idx < 0:
            self.show_info("Wählen Sie einen Mode aus.")
            return
        self.plot_selected_mode(idx)

    # ======================================================
    # Übertragungsfunktion plotten
    # ======================================================
    def plot_transfer_function(self):
        if self.tf_mag is None:
            self.show_info("Laden Sie das Betrag-FFT-File.")
            return

        xmax = float(self.freq_input.value())
        if xmax <= 0.0:
            xmax = self.frequency.max()

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for i, kanal in enumerate(self.kanalnamen):
            tf = self.tf_mag[:, i]
            ax.plot(self.frequency, tf, label=f"{kanal}")

        ax.set_xlabel("Frequenz [Hz]")
        ax.set_ylabel("Betrag")
        ax.grid(True)
        if len(self.kanalnamen) > 0:
            ax.set_xlim(0, xmax)
            ax.set_ylim(np.min(self.tf_mag), 10)

        ax.set_title("Übertragungsfunktion (alle Kanäle)")
        self.canvas.draw()

    # ======================================================
    # Info-Text anzeigen
    # ======================================================
    def show_info(self, text):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=14, wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        self.canvas.draw()


# ======================================================
# MAIN
# ======================================================
def main():
    app = QApplication(sys.argv)
    win = Plott_UEfkt_Modes()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
