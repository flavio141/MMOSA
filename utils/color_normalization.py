import torch
import numpy as np
from PIL import Image
import torch.linalg as linalg


def normalizeStaining(img, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images
    
    Example use:
        see test.py
        
    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''
             
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
        
    maxCRef = np.array([1.9705, 1.0308])
    array_img = np.array(img)
    h, w, _ = array_img.shape
    
    # reshape image
    array_img = torch.tensor(array_img.reshape((-1,3)), dtype=torch.float64).cuda()

    # calculate optical density
    OD = -torch.log((array_img + 1) / Io)
    
    # remove transparent pixels
    ODhat = OD[~torch.any(OD<beta, axis=1)]
        
    # compute eigenvectors
    cov_ODhat = torch.cov(ODhat.T)
    _, eigvecs = linalg.eigh(cov_ODhat)
    
    #project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
    That = ODhat @ eigvecs[:, 1:3]
    
    phi = torch.atan2(That[:, 1], That[:, 0])
    
    minPhi = torch.quantile(phi, alpha / 100)
    maxPhi = torch.quantile(phi, 1 - alpha / 100)
    
    vMin = eigvecs[:, 1:3] @ torch.tensor([torch.cos(minPhi), torch.sin(minPhi)]).cuda()
    vMax = eigvecs[:, 1:3] @ torch.tensor([torch.cos(maxPhi), torch.sin(maxPhi)]).cuda()
    

    if vMin[0] > vMax[0]:
        HE = torch.stack((vMin, vMax), axis=1)
    else:
        HE = torch.stack((vMax, vMin), axis=1)
    
    Y = OD.T
    C = linalg.lstsq(HE, Y)[0]
    
    maxC = torch.tensor([torch.quantile(C[0, :], 0.99), torch.quantile(C[1, :], 0.99)]).cuda()
    tmp = maxC / torch.tensor(maxCRef).cuda()
    C2 = C / tmp[:, None]
    
    # Recreate the image using reference mixing matrix
    Inorm = Io * torch.exp(-torch.tensor(HERef).cuda() @ C2)
    Inorm = torch.clamp(Inorm, max=254)
    Inorm = Inorm.reshape((3, h, w))

    return Inorm
