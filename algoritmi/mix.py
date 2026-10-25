import cv2
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
        polar_img1 = cv2.flip(cv2.transpose(polar_img1), 0)
        polar_img2 = cv2.flip(cv2.transpose(polar_img2), 0)
        #visualizzatemporaneo
        cv2.imshow("pol1", polar_img1)
        cv2.imshow("pol2", polar_img2)
                #find traiettorie
                #correlazione
        return 