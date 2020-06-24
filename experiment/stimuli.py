import numpy as np
from psychopy.visual import TextStim, Line

class FixationLines(object):

    def __init__(self, win, circle_radius, color, *args, **kwargs):
        self.line1 = Line(win, start=(-circle_radius, -circle_radius),
                end=(circle_radius, circle_radius), lineColor=color, *args, **kwargs)
        self.line2 = Line(win, start=(-circle_radius, circle_radius),
                end=(circle_radius, -circle_radius), lineColor=color, *args, **kwargs)

    def draw(self):
        self.line1.draw()
        self.line2.draw()

class ImageArrayStim(object):

    def __init__(self,
            window, 
            image,
            xys,
            size,
            *args,
            **kwargs):

        self.xys = xys
        self.size = size
        self.image = image

    def draw(self):
        for pos in self.xys:
            self.image.pos = pos
            self.image.draw()

def _create_stimulus_array(win, n_dots, circle_radius, dot_radius,
        image):
    xys = _sample_dot_positions(n_dots, circle_radius, dot_radius) 

    return ImageArrayStim(win,
            image,
            xys,
            dot_radius*2)
                

def _sample_dot_positions(n=10, circle_radius=20, dot_radius=1, max_tries=100000):

    counter = 0

    distances = np.zeros((n, n))
    while(((distances < dot_radius*2).any())):
        radius = np.random.rand(n) * np.pi * 2
        # Sqrt for uniform distribution (https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly)
        ecc = np.sqrt(np.random.rand(n)) * (circle_radius - dot_radius)

        coords = np.vstack(([np.cos(radius)], [np.sin(radius)])).T * ecc[:, np.newaxis]

        distances = np.sqrt(((coords[:, np.newaxis, :] - coords[np.newaxis, ...])**2).sum(2))

        np.fill_diagonal(distances, np.inf)
        counter +=1

        if counter > max_tries:
            raise Exception('Too many tries')

    return coords


