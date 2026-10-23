import cv2
import numpy as np

def calculate_emd(img1, img2):
    """
    Calcola l'Earth Mover's Distance (EMD) basato sui livelli di intensità (luminosità),
    perfetto per funzionare correttamente anche con immagini in scala di grigi o FITS.
    """
    # Se l'immagine è a colori (3 canali), la convertiamo in scala di grigi
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1

    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2

    # Definiamo il numero di bin per l'intensità (es. 64 bin per una buona precisione)
    bins = 64
    ranges = [0, 256]

    # Calcoliamo l'istogramma 1D basato sulla luminosità
    hist1 = cv2.calcHist([gray1], [0], None, [bins], ranges, accumulate=False)
    hist2 = cv2.calcHist([gray2], [0], None, [bins], ranges, accumulate=False)

    # Normalizzazione L1 (la somma dei pesi deve essere rigorosamente 1 per l'EMD)
    cv2.normalize(hist1, hist1, alpha=1, beta=0, norm_type=cv2.NORM_L1)
    cv2.normalize(hist2, hist2, alpha=1, beta=0, norm_type=cv2.NORM_L1)

    # Costruiamo le firme (signatures) per cv2.EMD
    # Ogni riga contiene: [peso, posizione_del_bin]
    sig1 = np.zeros((bins, 2), dtype=np.float32)
    sig2 = np.zeros((bins, 2), dtype=np.float32)

    for i in range(bins):
        sig1[i] = [hist1[i], i]
        sig2[i] = [hist2[i], i]

    # Calcoliamo l'Earth Mover's Distance usando la distanza L2
    emd_value, _, _ = cv2.EMD(sig1, sig2, cv2.DIST_L2)
    
    return emd_value