import cv2
from matplotlib import pyplot as plt
import numpy as np

def remove_high_contrast_noise(img, kernel_size=25, 
                               y_start=800, threshold_start=1.0, 
                               y_end=900, threshold_end=20.0):
    """
    Rimuove le tracce ad alto contrasto (stelle nel background) 
    con soglie fisse prima di y_start e dopo y_end,
    e una transizione lineare nell'intervallo tra y_start e y_end.
    """
   
    h, w = img.shape
    
    if y_end is None:
        y_end = h - 1

    #media locale e la deviazione standard (contrasto locale)
    mean = cv2.blur(img, (kernel_size, kernel_size))
    mean_sq = cv2.blur(img**2, (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    
    #profilo delle soglie riga per riga
    threshold_profile = np.empty(h, dtype=np.float32)
    
    #prima di y_start: valore fisso di partenza
    threshold_profile[:y_start] = threshold_start
    
    #tra y_start e y_end: transizione graduale
    if y_end > y_start:
        ramp_len = y_end - y_start + 1
        threshold_profile[y_start:y_end+1] = np.linspace(threshold_start, threshold_end, ramp_len)
    else:
        threshold_profile[y_start] = threshold_start
        
    #dopo y_end: valore fisso finale
    threshold_profile[y_end+1:] = threshold_end
    
    #espande la soglia alla dimensione della larghezza (h, 1) -> (h, w)
    threshold_matrix = threshold_profile[:, np.newaxis]
    
    #maschera basata sulla soglia dinamica dipendente da y
    mask = local_std > threshold_matrix
    
    #gestione opzionale del fade verticale
    y_ramp = np.linspace(0, 1, h)[:, np.newaxis]
    weight = 1.0 - y_ramp
    
    effective_mask = mask.astype(np.float32) * weight
    #sostituzione delle zone ad alto contrasto
    cleaned = img * (1.0 - effective_mask) + mean * effective_mask
    
    return np.clip(cleaned, 0, 255).astype(np.uint8)

def mix(img1, img2):
        
        #coordinate polari
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        center1= (w1 // 2, h1 // 2)
        center2= (w2 // 2, h2 // 2)
        maxRadius1 = np.sqrt(center1[0]**2 + center1[1]**2)
        maxRadius2 = np.sqrt(center2[0]**2 + center2[1]**2)

        dsize1 = (w1, int(maxRadius1))
        dsize2 = (w2, int(maxRadius2))

        polar_img1 = cv2.warpPolar(img1, dsize1, center1, maxRadius1, 
                              cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR)
        polar_img2 = cv2.warpPolar(img2, dsize2, center2, maxRadius2, 
                              cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR)
        polar_img1 = cv2.transpose(polar_img1) # centro dell'immagine originale corrisponde alla prima riga
        polar_img2 = cv2.transpose(polar_img2)

        # per visualizzazione: centro corrisponde all'ultima riga, che appare in basso
        polar_img1_flipped = cv2.flip(polar_img1, 0)
        polar_img2_flipped = cv2.flip(polar_img2, 0)
        #pulisce solo fotografia
        polar_img1_flipped=remove_high_contrast_noise(polar_img1_flipped)
        #visualizzatemporaneo
        _, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(polar_img1_flipped, cmap='gray')
        ax[0, 1].imshow(polar_img2_flipped, cmap='gray')

        # raggio approssimato del modello: raggio della circonferenza più grande lungo la quale
        # ci sia ancora informazione visiva (anziché vuoto)

        # valore arbitrario
        threshold = 64 
        model_radius = polar_img2.shape[0] - 1
        while (np.sum(polar_img2[model_radius]) < threshold) and (model_radius > 0):
                model_radius -= 1
        print(f"raggio di analisi: {model_radius} px")
        # analizzare alcune o tutte le circonferenze tra il centro e il raggio del modello

        # esempio: raggio // 2
        ax[1, 0].plot(polar_img1_flipped[model_radius // 2])
        ax[1, 1].plot(polar_img2_flipped[model_radius // 2])
        plt.show()
                #find traiettorie
                #correlazione
        return 