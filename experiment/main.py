import os.path as op
from exptools2.core import Session
from exptools2.core import Trial
from psychopy.visual import TextStim, Line
from psychopy import visual
from exptools2 import utils
from psychopy import logging
import numpy as np
import argparse

class InstructionTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations=[10000],
            txt=None, **kwargs):

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        if txt is None:
            txt = '''Keep looking at the colored cross in the middle of the screen.
            Indicate when the fixation cross changes to green with your 
            index finger. Response with your middle finger if the fixation
            cross changes to red.'''
            self.text = TextStim(self.session.win, txt)

        self.n_triggers = 0

    def draw(self):
        if self.phase == 0:
            self.text.draw()
        if self.phase > 0:
            self.session.fixation_lines.draw()

    def get_events(self):
        events = super().get_events()

        for key, t in events:
            if key == self.session.mri_trigger:
                self.stop_phase()

        
class MapperTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, txt=None,
            n_dots=5, **kwargs):

        phase_durations = []

        for key in session.settings['paradigm'].keys():
            setattr(self, key, session.settings['paradigm'].get(key))

        self.color1 = tuple(self.color1)
        self.color2 = tuple(self.color2)

        for i in range(self.n_repeats_stimulus):
            phase_durations += [self.on_duration, self.off_duration]

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        self.parameters['n_dots'] = n_dots

        self.colors = list(((np.random.rand(self.n_repeats_stimulus) < self.p_oddball) + 1).astype(int))

        self.stimulus_arrays = [self._create_stimulus_array(n_dots,
            self.aperture_size, self.dot_size/2.,
            image=[self.session.image1, self.session.image2][color-1])  \
                    for i, color in enumerate(self.colors)]


    def draw(self):
        """ Draws stimuli """
        self.session.fixation_lines.draw()

        if self.phase % 2 == 0:
            self.stimulus_arrays[int(self.phase / 2)].draw()

    def log_phase_info(self, phase=None):
        self.parameters['color'] = self.colors[int(self.phase/2)]
        super().log_phase_info(phase=phase)

    def _create_stimulus_array(self, n_dots, circle_radius, dot_radius,
            image):
        xys = _sample_dot_positions(n_dots, circle_radius, dot_radius) 

        return ImageArrayStim(self.session.win,
                image,
                xys,
                dot_radius*2)
                

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


class MapperSession(Session):
    """ Simple session with x trials. """
    def __init__(self, output_str, output_dir=None, settings_file=None, n_trials=10):
        """ Initializes TestSession object. """
        self.n_trials = n_trials
        super().__init__(output_str, output_dir=None, settings_file=settings_file)

        self.change_times = np.hstack([[0], np.cumsum(np.random.rand(1000) * 5.)])


        self.default_fix = TextStim(self.win, '+')
        self.default_fix.height = self.settings['paradigm'].get('fixation_size')
        self.dot_size = self.settings['paradigm'].get('dot_size')

        self.fixation_lines = FixationLines(self.win,
                self.settings['paradigm'].get('aperture_size'),
                color=(1, -1, -1))
        
        self.image1 = visual.ImageStim(self.win, 
                self.settings['paradigm'].get('image1'),
                size=self.settings['paradigm'].get('dot_size'))
        self.image2 = visual.ImageStim(self.win, 
                self.settings['paradigm'].get('image2'),
                size=self.settings['paradigm'].get('dot_size'))

    def create_trials(self, durations=(2., 5.), timing='seconds'):
        n_dummies = self.settings['paradigm'].get('n_dummy_scans')

        phase_durations = [10000] * (n_dummies + 1)

        design = self.settings['paradigm'].get('design')
        n_blocks = self.settings['paradigm'].get('n_repeats_blocks')
        block_length = len(self.settings['paradigm'].get('design'))

        self.trials = [InstructionTrial(session=self, trial_nr=0,
            phase_durations=phase_durations)]
        for block in range(n_blocks):
            for trial_nr, n_dots in enumerate(design):

                trial_nr += block*block_length + 1
                self.trials.append(
                    MapperTrial(session=self,
                              trial_nr=trial_nr,
                              phase_durations=durations,
                              txt='Trial %i' % (trial_nr),
                              n_dots=n_dots,
                              verbose=True,
                              timing=timing)
                )

    def run(self):
        """ Runs experiment. """
        self.start_experiment()
        for trial in self.trials:
            trial.run()

        self.close()

class FixationLines(object):

    def __init__(self, win, circle_radius, color, *args, **kwargs):
        self.line1 = Line(win, start=(-circle_radius, -circle_radius),
                end=(circle_radius, circle_radius), lineColor=color, *args, **kwargs)
        self.line2 = Line(win, start=(-circle_radius, circle_radius),
                end=(circle_radius, -circle_radius), lineColor=color, *args, **kwargs)

    def draw(self):
        self.line1.draw()
        self.line2.draw()


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, nargs='?')
    parser.add_argument('session', default=None, nargs='?')
    args = parser.parse_args()
    
    if args.subject is None:
        subject = input('Subject? (999): ')
        subject = 999 if subject  == '' else subject
    else:
        subject = args.subject

    if args.subject is None:
        session = input('Session? (1): ')
        session = 1 if session  == '' else session
    else:
        session = args.session

    settings = op.join(op.dirname(__file__), 'settings.yml')
    session = MapperSession(f'sub-{subject}_ses-{session}',
            n_trials=3, settings_file=settings)
    session.create_trials(durations=(.4, .6), timing='seconds')

    session.run()
    session.quit()
