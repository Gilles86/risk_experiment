from nilearn import image

def psc(data):
    return image.math_img('(data/data.mean(-1)[..., np.newaxis]) * 100 - 100', data=data)
