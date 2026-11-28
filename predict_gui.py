# CycloneNet desktop app: predict cyclone / no cyclone from INSAT (or Kalpana) .npz files,
# evaluate whole INSAT test folders (confusion matrix, accuracy, precision ...) and show the training graphs.
#   python predict_gui.py
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from common import ROOT
from predict import load_npz_pair, load_model_and_stats, predict_from_arrays
import plot_results

OUT = ROOT / "outputs"


class CycloneApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CycloneNet - Bay of Bengal Cyclone Detector")
        self.root.geometry("1100x700")
        self.root.minsize(950, 600)

        self.model = None
        self.stats = None
        self.results = {}       # row id -> dict(prob, label, tir, wv)
        self._n = 0             # running row counter (row ids must be unique)

        self._build_layout()
        self._load_model_async()

    def _build_layout(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Predict:", font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        self.upload_btn = ttk.Button(top, text="Upload .npz file(s)", command=self.upload_npz_files)
        self.upload_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.upload_folder_btn = ttk.Button(top, text="Upload folder", command=self.upload_npz_folder)
        self.upload_folder_btn.pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(top, text="Test:", font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        self.eval_btn = ttk.Button(top, text="Evaluate INSAT (cyclone + non-cyclone folders)", command=self.evaluate_folders)
        self.eval_btn.pack(side=tk.LEFT, padx=(0, 16))

        self.graphs_btn = ttk.Button(top, text="Show graphs", command=self.show_graphs)
        self.graphs_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.clear_btn = ttk.Button(top, text="Clear results", command=self.clear_all)
        self.clear_btn.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Loading model...")
        ttk.Label(top, textvariable=self.status_var, foreground="#555").pack(side=tk.RIGHT)

        # one-line summary of the last evaluation
        self.summary_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.summary_var, font=("TkDefaultFont", 10, "bold"),
                  foreground="#1f4e79", padding=(10, 0, 10, 6)).pack(side=tk.TOP, anchor=tk.W)

        body = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        columns = ("file", "true", "prediction", "probability", "check")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", selectmode="browse", height=18)
        for col, text, width, anchor in [("file", "File", 270, tk.W), ("true", "True label", 100, tk.CENTER),
                                         ("prediction", "Prediction", 100, tk.CENTER),
                                         ("probability", "Cyclone probability", 130, tk.CENTER),
                                         ("check", "Correct?", 70, tk.CENTER)]:
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_row)

        self.tree.tag_configure("cyclone", background="#fde2e2")
        self.tree.tag_configure("nocyclone", background="#e2f5e5")
        self.tree.tag_configure("wrong", background="#ffe9a8")     # misclassified (only when true label is known)

        scrollbar = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(body, width=380)
        right.pack(side=tk.LEFT, fill=tk.BOTH, padx=(12, 0))

        self.detail_label = ttk.Label(right, text="Select a result to preview", font=("TkDefaultFont", 11, "bold"))
        self.detail_label.pack(side=tk.TOP, anchor=tk.W, pady=(0, 6))

        self.fig = Figure(figsize=(4.2, 4.8), dpi=90)
        self.ax_tir = self.fig.add_subplot(211)
        self.ax_wv = self.fig.add_subplot(212)
        self._blank_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.prob_label = ttk.Label(right, text="", font=("TkDefaultFont", 14, "bold"))
        self.prob_label.pack(side=tk.TOP, pady=(8, 0))

    def _blank_axes(self):
        for ax, title in ((self.ax_tir, "TIR (thermal infrared)"), (self.ax_wv, "Water vapor")):
            ax.clear(); ax.axis('off'); ax.set_title(title, fontsize=9)
        self.fig.tight_layout()

    # model is loaded in a thread so the window doesn't freeze
    def _load_model_async(self):
        def worker():
            try:
                model, stats = load_model_and_stats()
                self.model, self.stats = model, stats
                msg = f"Model ready (threshold {stats['threshold']:.2f})"
                self.root.after(0, lambda: self.status_var.set(msg))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: self.status_var.set("Failed to load model"))
                self.root.after(0, lambda: messagebox.showerror(
                    "Model load error", f"Could not load outputs/final.pt\n(run train.py first)\n\n{err}"))
        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _npz_in(folder):
        paths = []
        for root_dir, _, files in os.walk(folder):
            for fn in files:
                if fn.lower().endswith(".npz"):
                    paths.append(os.path.join(root_dir, fn))
        return sorted(paths)

    def upload_npz_files(self):
        paths = filedialog.askopenfilenames(title="Select satellite .npz file(s)",
                                            filetypes=[("NumPy archive", "*.npz"), ("All files", "*.*")])
        if paths:
            self._process([(p, None) for p in paths])

    def upload_npz_folder(self):
        folder = filedialog.askdirectory(title="Select a folder of .npz files")
        if folder:
            paths = self._npz_in(folder)
            if not paths:
                messagebox.showinfo("No files found", "No .npz files were found in that folder.")
                return
            self._process([(p, None) for p in paths])

    def evaluate_folders(self):
        cyc = filedialog.askdirectory(title="Select the INSAT CYCLONE folder (true label = cyclone)")
        if not cyc:
            return
        non = filedialog.askdirectory(title="Select the INSAT NON-CYCLONE folder (true label = no cyclone)")
        if not non:
            return
        items = [(p, 1) for p in self._npz_in(cyc)] + [(p, 0) for p in self._npz_in(non)]
        if not items:
            messagebox.showinfo("No files found", "No .npz files were found in those folders.")
            return
        self._process(items, evaluate=True)

    # predictions run in a worker thread; rows are added on the tk thread with root.after
    def _process(self, items, evaluate=False):
        if self.model is None:
            messagebox.showwarning("Model not ready", "The model is still loading - try again in a moment.")
            return
        self.status_var.set(f"Running prediction on {len(items)} file(s)...")
        self._set_buttons_enabled(False)
        thr = self.stats["threshold"]

        def worker():
            rows = []
            for i, (p, true) in enumerate(items):
                try:
                    tir, wv = load_npz_pair(p)
                    prob, label = predict_from_arrays(tir, wv, self.model, self.stats)
                    self.root.after(0, self._add_result, os.path.basename(p), true, prob, label, tir, wv)
                    if true is not None:
                        rows.append(dict(file=os.path.basename(p), label=true, prob_cyclone=round(prob, 4),
                                         pred=int(label == "CYCLONE"), threshold=round(thr, 4)))
                except Exception as e:
                    self.root.after(0, self._add_error_row, os.path.basename(p), str(e))
                if i % 10 == 0:
                    self.root.after(0, self.status_var.set, f"Running prediction... {i + 1}/{len(items)}")
            self.root.after(0, self._done_processing, rows, evaluate)

        threading.Thread(target=worker, daemon=True).start()

    def _add_result(self, name, true, prob, label, tir, wv):
        self._n += 1
        iid = str(self._n)
        # keep raw images as uint16 to save memory (counts are <= 1023)
        self.results[iid] = dict(name=name, prob=prob, label=label,
                                 tir=np.round(tir).astype(np.uint16), wv=np.round(wv).astype(np.uint16))
        if true is None:
            true_txt, check, tag = "-", "", ("cyclone" if label == "CYCLONE" else "nocyclone")
        else:
            true_txt = "CYCLONE" if true == 1 else "NO CYCLONE"
            ok = (label == "CYCLONE") == (true == 1)
            check = "yes" if ok else "NO"
            tag = ("cyclone" if label == "CYCLONE" else "nocyclone") if ok else "wrong"
        self.tree.insert("", tk.END, iid=iid, values=(name, true_txt, label, f"{prob*100:.2f}%", check), tags=(tag,))

    def _add_error_row(self, name, err):
        self.tree.insert("", tk.END, values=(name, "", "ERROR", err[:40], ""))

    def _done_processing(self, rows, evaluate):
        self.status_var.set(f"Model ready (threshold {self.stats['threshold']:.2f})")
        self._set_buttons_enabled(True)
        if evaluate and rows:
            df = pd.DataFrame(rows)
            OUT.mkdir(exist_ok=True)
            df.to_csv(OUT / "insat_predictions.csv", index=False)      # same file test_insat.py writes
            m = plot_results.compute_metrics(df.label, df.pred, df.prob_cyclone)
            auc = "n/a" if np.isnan(m["auc"]) else f"{m['auc']:.3f}"
            self.summary_var.set(
                f"INSAT test, {m['n']} images  |  Accuracy {m['accuracy']:.3f}  Precision {m['precision']:.3f}  "
                f"Recall {m['recall']:.3f}  F1 {m['f1']:.3f}  AUC {auc}  |  "
                f"TN {m['tn']}  FP {m['fp']}  FN {m['fn']}  TP {m['tp']}")
            messagebox.showinfo("Evaluation finished",
                                "Saved outputs/insat_predictions.csv.\nClick 'Show graphs' to see the confusion matrix, ROC and more.")

    def _set_buttons_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in (self.upload_btn, self.upload_folder_btn, self.eval_btn, self.graphs_btn):
            btn.config(state=state)

    def clear_all(self):
        self.tree.delete(*self.tree.get_children())
        self.results.clear()
        self._blank_axes()
        self.canvas.draw()
        self.prob_label.config(text="")
        self.summary_var.set("")
        self.detail_label.config(text="Select a result to preview")

    def on_select_row(self, _event):
        sel = self.tree.selection()
        if not sel or sel[0] not in self.results:
            return
        r = self.results[sel[0]]
        self.detail_label.config(text=r['name'])
        self.ax_tir.clear(); self.ax_tir.imshow(r['tir'], cmap='gray'); self.ax_tir.axis('off')
        self.ax_tir.set_title("TIR (thermal infrared)", fontsize=9)
        self.ax_wv.clear(); self.ax_wv.imshow(r['wv'], cmap='gray'); self.ax_wv.axis('off')
        self.ax_wv.set_title("Water vapor", fontsize=9)
        self.canvas.draw()
        color = "#c0392b" if r['label'] == "CYCLONE" else "#27ae60"
        self.prob_label.config(text=f"{r['label']}  -  {r['prob']*100:.2f}% probability", foreground=color)

    # opens a window with one tab per graph (training curves, CV results, confusion matrix, ROC ...)
    def show_graphs(self):
        figs = plot_results.build_figures(OUT)
        if not figs:
            messagebox.showinfo("Nothing to plot", "No results found in outputs/. Run train.py first, "
                                                   "then evaluate INSAT folders (or run test_insat.py).")
            return
        win = tk.Toplevel(self.root)
        win.title("CycloneNet - training and test graphs")
        win.geometry("1150x740")
        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True)
        for name, fig in figs.items():
            tab = ttk.Frame(nb)
            nb.add(tab, text=name)
            canvas = FigureCanvasTkAgg(fig, master=tab)
            toolbar = NavigationToolbar2Tk(canvas, tab)       # zoom / save-as-PNG buttons
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            canvas.draw()


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    CycloneApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
