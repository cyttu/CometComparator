import cv2
from matplotlib import pyplot as plt
import numpy as np

            

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

        #visualizzatemporaneo
        _, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(polar_img1_flipped, cmap='gray')
        ax[0, 1].imshow(polar_img2_flipped, cmap='gray')
        #cv2.imshow("pol1", polar_img1)
        #cv2.imshow("pol2", polar_img2)

        # raggio approssimato del modello: raggio della circonferenza più grande lungo la quale
        # ci sia ancora informazione visiva (anziché vuoto)
        # (si assume che `polar_img2` è il modello)
        threshold = 64 # idk, arbitrario
        model_radius = polar_img2.shape[0] - 1
        while (np.sum(polar_img2[model_radius]) < threshold) and (model_radius > 0):
                model_radius -= 1
        print(f"raggio di analisi: {model_radius} px")
        # analizzare alcune o tutte le circonferenze tra il centro e il raggio del modello

        # esempio: raggio // 2
        ax[1, 0].plot(polar_img1[model_radius // 2])
        ax[1, 1].plot(polar_img2[model_radius // 2])
        plt.show()
                #find traiettorie
                #correlazione
        return 