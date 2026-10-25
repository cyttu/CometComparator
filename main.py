import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from astropy.io import fits
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tempfile
import os
import algoritmi.ssim as ssim
import algoritmi.emd as emd
import algoritmi.mutualinfo as mutualinfo
import algoritmi.mix as mix
import numpy as np

class ImageComparatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Software di Comparazione Immagini")
        self.root.geometry("500x350")
        self.root.resizable(False, False)

        # Lista di "n" algoritmi (puoi modificarla o aggiungerne quanti ne vuoi)
        self.algorithms_list = [
            "Structural Similarity (SSIM)",
            "Earth Mover's Distance (EMD)",
            "Mutual Information",
            "Mix"
        ]

        # --- SEZIONE FILE 1 ---
        lbl_file1 = tk.Label(root, text="Fotografia:", font=("Arial", 10, "bold"))
        lbl_file1.pack(anchor="w", padx=20, pady=(15, 0))
        
        frame1 = tk.Frame(root)
        frame1.pack(fill="x", padx=20, pady=5)
        
        self.entry_file1 = tk.Entry(frame1, width=45)
        self.entry_file1.pack(side="left", padx=(0, 10))
        
        btn_browse1 = tk.Button(frame1, text="Sfoglia...", command=lambda: self.browse_file(self.entry_file1))
        btn_browse1.pack(side="left")

        # --- SEZIONE FILE 2 ---
        lbl_file2 = tk.Label(root, text="Simulazione:", font=("Arial", 10, "bold"))
        lbl_file2.pack(anchor="w", padx=20, pady=(10, 0))
        
        frame2 = tk.Frame(root)
        frame2.pack(fill="x", padx=20, pady=5)
        
        self.entry_file2 = tk.Entry(frame2, width=45)
        self.entry_file2.pack(side="left", padx=(0, 10))
        
        btn_browse2 = tk.Button(frame2, text="Sfoglia...", command=lambda: self.browse_file(self.entry_file2))
        btn_browse2.pack(side="left")

        # --- SEZIONE SCELTA ALGORITMO ---
        lbl_algo = tk.Label(root, text="Seleziona Algoritmo:", font=("Arial", 10, "bold"))
        lbl_algo.pack(anchor="w", padx=20, pady=(10, 0))
        
        self.selected_algo = tk.StringVar(root)
        self.selected_algo.set(self.algorithms_list[0]) # Valore di default
        
        dropdown_algo = tk.OptionMenu(root, self.selected_algo, *self.algorithms_list)
        dropdown_algo.config(width=40)
        dropdown_algo.pack(padx=20, pady=5, anchor="w")

        # --- PULSANTE DI INVIO ---
        btn_submit = tk.Button(root, text="Avvia Confronto", bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), command=self.on_submit)
        btn_submit.pack(pady=20)

    def browse_file(self, entry_widget):
        """Apre la finestra di dialogo per selezionare un file e riempie la casella di testo."""
        file_path = filedialog.askopenfilename(
            title="Seleziona immagine",
            filetypes=[("File Immagine", "*.jpg *.jpeg *.png *.bmp *.fit *.fits"), ("Tutti i file", "*.*")]
        )
        if file_path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, file_path)

    
    def prepare_image(self, file_path: str) -> str:
        """
        Controlla il formato del file:
        - Se è già un'immagine standard (png, jpg, ecc.), restituisce il percorso originale.
        - Se è un file FITS (.fit, .fits), lo converte in PNG temporaneo e restituisce il nuovo percorso.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in ['.fit', '.fits']:
            try:
                # Legge il file FITS
                with fits.open(path) as hdul:
                    image_data = hdul[0].data
                
                # Crea un file temporaneo in formato PNG
                temp_dir = tempfile.gettempdir()
                temp_png_path = os.path.join(temp_dir, f"{path.stem}_converted.png")
                
                # Salva l'array come immagine PNG pulita
                plt.figure(figsize=(6, 6))
                plt.imshow(image_data, cmap='gray', origin='lower')
                plt.axis('off')
                plt.savefig(temp_png_path, bbox_inches='tight', pad_inches=0, dpi=150)
                plt.close()
                
                return temp_png_path
            except Exception as e:
                raise RuntimeError(f"Errore nella conversione del file FITS {path.name}: {e}")
        
        # Se non è FITS, restituisce il percorso così com'è
        return str(path)

    def on_submit(self):
        """Funzione eseguita al click del pulsante di invio."""
        path1 = self.entry_file1.get().strip()
        path2 = self.entry_file2.get().strip()
        chosen_algorithm = self.selected_algo.get()

        # Validazione di base
        if not path1 or not path2:
            messagebox.showerror("Errore", "Devi selezionare entrambi i file immagine!")
            return
        
        if not Path(path1).is_file() or not Path(path2).is_file():
            messagebox.showerror("Errore", "Uno o entrambi i percorsi inseriti non sono file validi.")
            return

        try:
            # Converte i file se necessario (gestisce automaticamente i .fit/.fits)
            processed_path1 = self.prepare_image(path1)
            processed_path2 = self.prepare_image(path2)
        except Exception as e:
            messagebox.showerror("Errore di Conversione", str(e))
            return
        
        try:
            # 1. Caricamento e ridimensionamento a priori
            pil_img1 = Image.open(processed_path1).convert('L')
            pil_img2 = Image.open(processed_path2).convert('L')

            if pil_img1.size != pil_img2.size:
                pil_img2 = pil_img2.resize(pil_img1.size, Image.Resampling.LANCZOS)
                messagebox.showwarning(
                    "Adattamento Dimensioni", 
                    f"Le immagini avevano risoluzioni differenti.\n"
                    f"La seconda immagine è stata ridimensionata a priori a {pil_img1.size[0]}x{pil_img1.size[1]} pixel."
                )
                
            # 2. Esecuzione in base all'algoritmo selezionato
            if chosen_algorithm == "Structural Similarity (SSIM)":
                # Usiamo direttamente le immagini PIL già ridimensionate e le convertiamo in array
                img1 = np.array(pil_img1)
                img2 = np.array(pil_img2)

                # Calcola l'SSIM usando la funzione importata
                ssim_map = ssim.ssim(img1, img2)
                score = ssim_map.mean()

                # Mostra il risultato
                messagebox.showinfo("Risultato SSIM", f"Il punteggio SSIM medio è: {score:.4f}")
                return
            elif chosen_algorithm == "Earth Mover's Distance (EMD)":
    # Usiamo direttamente le immagini PIL già ridimensionate e le convertiamo in array
                img1 = np.array(pil_img1)
                img2 = np.array(pil_img2)
    
    # Calcola l'EMD usando la funzione importata
                score = emd.calculate_emd(img1, img2)
    
    # Mostra il risultato
                messagebox.showinfo("Risultato EMD", f"Il punteggio EMD medio è: {score:.4f}")
                return
            elif chosen_algorithm == "Mutual Information":
             # Usiamo direttamente le immagini PIL già ridimensionate a monte
                img1 = np.array(pil_img1)
                img2 = np.array(pil_img2)

             # Calcola la Mutual Information
                score = mutualinfo.calculate_mutual_information(img1, img2)

             # Mostra il risultato
                messagebox.showinfo(
                 "Risultato Mutual Information", 
                 f"Mutual Information (MI): {score:.4f} bit\n(Valori più alti indicano una forte dipendenza e somiglianza statistica)"
                )
                return
            elif chosen_algorithm == "Mix":
                #coordinate polari
                img1 = np.array(pil_img1)
                img2 = np.array(pil_img2)
                mix.mix(img1,img2)
                
                
                return
            else:
                # Placeholder per gli altri algoritmi futuri
                messagebox.showinfo(
                    "Confronto avviato", 
                    f"File 1 elaborato: {Path(processed_path1).name}\n"
                    f"File 2 elaborato: {Path(processed_path2).name}\n"
                    f"Algoritmo: {chosen_algorithm}\n\n"
                    f"(Pronto per essere collegato!)"
                )

        except Exception as e:
            messagebox.showerror("Errore di Elaborazione", f"Si è verificato un errore durante l'esecuzione:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageComparatorApp(root)
    root.mainloop()