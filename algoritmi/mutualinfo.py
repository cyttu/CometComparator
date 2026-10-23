import numpy as np

def calculate_mutual_information(img1, img2, bins=256):
    """
    Calcola la Mutual Information (Informazione Mutua) tra due immagini 
    utilizzando le distribuzioni di probabilità dei pixel (entropia congiunta e marginale).
    Un valore più alto indica una maggiore correlazione/somiglianza statistica.
    """
    # Assicura che le immagini siano in formato array 1D
    flat1 = img1.ravel()
    flat2 = img2.ravel()
    
    # Calcola l'istogramma congiunto 2D
    hist_joint, _, _ = np.histogram2d(flat1, flat2, bins=bins, range=[[0, 256], [0, 256]])
    
    # Converte l'istogramma congiunto in probabilità congiunta P(X, Y)
    p_joint = hist_joint / np.sum(hist_joint)
    
    # Calcola le probabilità marginali P(X) e P(Y) sommando lungo gli assi
    p1 = np.sum(p_joint, axis=1)
    p2 = np.sum(p_joint, axis=0)
    
    # Termine epsilon per evitare errori log(0)
    eps = 1e-10
    
    # Calcola le entropie di Shannon (marginali e congiunta)
    h1 = -np.sum(p1 * np.log2(p1 + eps))
    h2 = -np.sum(p2 * np.log2(p2 + eps))
    h_joint = -np.sum(p_joint * np.log2(p_joint + eps))
    
    # Formula della Mutual Information: I(X; Y) = H(X) + H(Y) - H(X, Y)

    mi = h1 + h2 - h_joint
    nmi = (2.0 * mi) / (h1 + h2 + eps)
    return float(nmi)