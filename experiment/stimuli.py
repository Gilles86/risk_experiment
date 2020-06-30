import numpy as np
from psychopy.visual import TextStim, Line, Rect, Pie

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


class CertaintyStimulus(object):
    def __init__(self,
            window, 
            response_size=(1, .5),
            fillColor=(1, 1, -1),
            *args,
            **kwargs):

        n_responses = 4
        total_width = n_responses * response_size[0] + (n_responses - 1) * .1 * response_size[0]
        positions = -total_width / 2. + response_size[0]/2. + np.arange(n_responses) * 1.1 * response_size[0]
        positions = [(pos, 0) for pos in positions]

        self.rectangles = [Rect(window, size=response_size, pos=positions[n], opacity=.5,
            fillColor=fillColor) for n in range(n_responses)]

        self.stim_texts = [TextStim(window, 
            height=.45*response_size[1],
            wrapWidth = .9*response_size[0], 
            pos=positions[n],
            color=(-1, -1, -1),
            text=n+1) for n in range(n_responses)]

        
        self.description_text = TextStim(window, pos=(0, 1.5 * response_size[1]), wrapWidth=response_size[0] * 8.8,
        text='How certain are you about your choice (1=very uncertain,  4= very certain)')

    def draw(self):
        for rect in self.rectangles:
            rect.draw()

        for txt in self.stim_texts:
            txt.draw()

        self.description_text.draw()


class ProbabilityPieChart(object):

    def __init__(self, window, prob, size, prefix='', pos=(0.0, 0.0), color_pos=(.65, .65, .65), color_neg=(-.65, -.65, -.65)):

        deg = prob * 360.

        self.piechart_pos =  Pie(window, end=deg, fillColor=color_pos,
                pos=pos,
                size=size)
        self.piechart_neg =  Pie(window, start=deg, end=360, fillColor=color_neg,
                pos=pos,
                size=size)

        txt = f'{prefix}{int(prob*100):d}%'
        self.text = TextStim(window, pos=(pos[0], pos[1]+size*1.5),
                text=txt, wrapWidth=size*3, height=size*.75)

    def draw(self):
        self.piechart_pos.draw()
        self.piechart_neg.draw()
        self.text.draw()


def _sample_dot_positions(n=10, circle_radius=20, dot_radius=1, max_tries=500000):

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


