import numpy as np

# get cos(theta) and sin(theta) arrays
def getCosSinThetaValues():
    thetas = np.deg2rad(np.arange(-90.0, 91.0))
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    return thetas, cos_thetas, sin_thetas

