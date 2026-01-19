import sys
from matplotlib import container
import numpy as np
import pandas as pd
import h5py

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QFileDialog, QGroupBox, QComboBox,
    QPushButton,QListWidget, QListWidgetItem, QDoubleSpinBox
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

        open_uefkt=QAction("Öffne Üfkt FFT Betrag File", self)
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
        self.freq_input.setMaximum(1e6)  # großzügig, passt für jede Frequenz
        self.freq_input.setValue(0.0)    # 0 = kein Limit, volle Achse
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
        self.eigenfreq_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)  # oder MultiSelection
        tf_layout.addWidget(self.eigenfreq_list)

        main_box.addWidget(tf_box)
        

        # ==========================
        # Mode Control
        # ==========================
        mode_box = QGroupBox("Mode Control")
        mode_layout = QVBoxLayout(mode_box)

        # Mode auswählen
        mode_layout.addWidget(QLabel("Mode auswählen:"))
        self.mode_combo = QComboBox()  # ← für Mode 1,2,3...
        mode_layout.addWidget(self.mode_combo)

        # Richtung wählen
        mode_layout.addWidget(QLabel("Richtung:"))
        self.dir_combo = QComboBox()  # ← für X/Y
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
        # Matplotlib Figure zurücksetzen
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

        # TSV korrekt laden
        df = pd.read_csv(path, sep="\t", comment="#", decimal=",")
        
        # Frequency-Spalte
        self.frequency = df["frequency"].astype(float).values

        # Alle Kanäle extrahieren, die "UEBERTRAGUNGSFUNKTION" enthalten
        kanalnamen = [c for c in df.columns if "UEBERTRAGUNGSFUNKTION" in c]
        kanalnamen = sorted(kanalnamen, key=lambda s: int(s.split("_")[-1]))

        # TF-Magnitude als float
        self.tf_mag = df[kanalnamen].astype(float).values
        self.kanalnamen = kanalnamen

        # Kanal-Combo füllen (optional)
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

        self.hammer_node_ids = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19])
    # ======================================================
    # MODE-AUSWAHL AKTIVIEREN
    # ======================================================
    def try_enable_modes(self):
        if not hasattr(self, "peak_indices"):
            return

        if self.node_ids is None or self.edges is None:
            return

        self.mode_combo.clear()

        for i in range(len(self.peak_indices)):
            self.mode_combo.addItem(f"Mode {i + 1}")

    # ======================================================
    # Eigenfrequenzen plotten
    # ======================================================
    def find_and_fill_eigenfreqs(self, tf_mag, xmax=None):
        # Prüfen, ob tf_mag ein NumPy Array ist
        if not isinstance(tf_mag, np.ndarray):
            self.show_info("Fehler: tf_mag ist kein Array.")
            return

        # Frequenzen auf xmax begrenzen
        if xmax is not None and xmax > 0:
            mask = self.frequency <= xmax
            freq = self.frequency[mask]
            tf_mag = tf_mag[mask]
        else:
            freq = self.frequency

        # Eigenfrequenzen finden
        self.eigenfreqs, self.peak_indices = find_eigenfrequencies(
            freq,
            tf_mag
        )

        # Liste leeren
        self.eigenfreq_list.clear()

        # Nur wenn Mode-Control existiert
        if hasattr(self, "mode_combo"):
            self.mode_combo.clear()
            for i, f in enumerate(self.eigenfreqs):
                self.mode_combo.addItem(f"Mode {i + 1}")

        for f in self.eigenfreqs:
            self.eigenfreq_list.addItem(f"{f:.2f} Hz")


    # ======================================================
    # MODE PLOTTEN
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

        if not hasattr(self, "hammer_node_ids"):
            self.show_info("Zuordnung Hammer → Knoten fehlt.")
            return

        freq_idx = self.peak_indices[mode_idx]

        #Peakwerte je Hammerposition
        mode_values = self.tf_imag[freq_idx, :]  # Länge = Anzahl Messungen

        #  Verschiebungsvektor (alle Knoten)
        u_all = np.zeros(len(self.node_ids))

        #Zuordnung Messung → Knoten
        for k, node_id in enumerate(self.hammer_node_ids):
            idx = np.where(self.node_ids == node_id)[0][0]
            u_all[idx] = mode_values[k]

        #Normierung
        if np.max(np.abs(u_all)) > 0:
            u_all /= np.max(np.abs(u_all))

        #Plot
        self.figure.clear()
        ax = self.figure.subplots()

        plot_mode(
            ax=ax,
            node_ids=self.node_ids,
            nodes=self.nodes,
            edges=self.edges,
            plot_lines=[("b", u_all)],
            axis_a="x",
            axis_b="z",
            disp_axis="x",
            scale=0.4,
        )

        ax.set_title(f"Mode {mode_idx + 1}")
        self.canvas.draw()


    # def plot_selected_mode(self, mode_idx):
    #     # Prüfen, ob Geometrie geladen ist
    #     if self.nodes is None or "z" not in self.nodes or self.node_ids is None or self.edges is None:
    #         self.show_info("Laden Sie das Geometry-File, um Modes zu plotten.")
    #         return

    #     if self.tf_imag is None:
    #         self.show_info("Laden Sie das Imaginär-FFT-File.")
    #         return

    #     if not hasattr(self, "peak_indices"):
    #         self.show_info("Die Eigenfrequenzen wurden noch nicht bestimmt.")
    #         return

    #     freq_idx = self.peak_indices[mode_idx]

    #     # Imaginärteil aller Sensoren bei dieser Frequenz
    #     u_meas = self.tf_imag[freq_idx, :]

    #     z_nodes = self.nodes["z"]
    #     z_meas = z_nodes[:len(u_meas)]

    #     u_all = build_mode(z_nodes, z_meas, u_meas)

    #     self.figure.clear()
    #     ax = self.figure.subplots()
    #     u_all_scaled = u_all / np.max(np.abs(u_all))  # Normierung auf 1
    #     plot_mode(
    #         ax=ax,
    #         node_ids=self.node_ids,
    #         nodes=self.nodes,
    #         edges=self.edges,
    #         plot_lines=[("b", u_all_scaled)],
    #         axis_a="x",
    #         axis_b="z",
    #         disp_axis="x",
    #         scale=0.4,
    #     )

    #     ax.set_title(f"Mode {mode_idx + 1}")
    #     self.canvas.draw()

    # ======================================================
    # Eigenfrequenz aus Liste auswählen und Mode plotten
    # ======================================================
    def plot_mode_from_eigenfreq(self, idx):
        if idx < 0:
            return

        self.mode_combo.setCurrentIndex(idx)
        self.eigenfreq_list.currentRowChanged.connect(
        self.plot_mode_from_eigenfreq
        )
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
    # Übertragungsfunktion PLOTTEN
    # ======================================================
    def plot_transfer_function(self):
        if self.tf_mag is None:
            self.show_info("Laden Sie das Betrag-FFT-File.")
            return
        try:
            xmax = float(self.freq_input.value())
        except AttributeError:
            xmax = 0.0  # falls self.freq_input noch nicht existiert

        if xmax <= 0.0:  # 0 oder negativ = volle Achse
            xmax = self.frequency.max()

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Alle Kanäle übereinander plotten
        for i, kanal in enumerate(self.kanalnamen):
            tf = self.tf_mag[:, i]
            ax.plot(self.frequency, tf, label=f"{kanal}")

        ax.set_xlabel("Frequenz [Hz]")
        ax.set_ylabel("Betrag")
        ax.grid(True)
        if len(self.kanalnamen) > 0:
            #ax.legend(fontsize=8)

            ax.set_xlim(0, xmax)
            ax.set_ylim(np.min(self.tf_mag), 10)

        ax.set_title("Übertragungsfunktion (alle Kanäle)")
        #ax.legend(fontsize=8)
        self.canvas.draw()



    # ======================================================
    # Hilfsfunktion: Info-Text anzeigen sofern keine Dateien geladen werden
    # ======================================================

    def show_info(self, text):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(
            0.5, 0.5, text,
            ha='center', va='center', fontsize=14,
            wrap=True
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        self.canvas.draw()

    def update_xlim_from_slider(self):
        if self.tf_mag is None or self.tf_mag.size == 0:
            return

        # Sliderwert als Prozent der maximalen Frequenz
        max_freq = self.frequency.max()
        slider_val = self.freq_slider.value()
        xmax = max_freq * slider_val / 100

        # Plot aktualisieren
        self.plot_transfer_function(xmax=xmax)


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
