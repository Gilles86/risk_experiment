import os.path as op
from exptools2.core import Trial
from psychopy import visual
import numpy as np
import argparse
from stimuli import _create_stimulus_array
from session import PileSession
from trial import InstructionTrial

        
class MapperTrial(Trial):
    def __init__(self, session, trial_nr, phase_durations, txt=None,
            n_dots=5, **kwargs):

        phase_durations = []

        mapper_settings = session.settings['mapper']

        for i in range(mapper_settings.get('n_repeats_stimulus')):
            phase_durations += [mapper_settings.get('on_duration'),
                    mapper_settings.get('off_duration')]

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        self.parameters['n_dots'] = n_dots

        self.colors = list(((np.random.rand(mapper_settings.get('n_repeats_stimulus')) < mapper_settings.get('p_oddball')) + 1).astype(int))

        self.stimulus_arrays = [_create_stimulus_array(self.session.win, n_dots,
            self.session.settings['pile'].get('aperture_size'),
            self.session.settings['pile'].get('dot_size')/2.,
            image=[self.session.image1, self.session.image2][color-1])  \
                    for i, color in enumerate(self.colors)]

    def draw(self):
        """ Draws stimuli """
        self.session.fixation_lines.draw()

        if self.phase % 2 == 0:
            self.stimulus_arrays[int(self.phase / 2)].draw()

    def log_phase_info(self):
        self.parameters['color'] = self.colors[int(self.phase/2)]
        super().log_phase_info()

class MapperSession(PileSession):

    def __init__(self, output_str, output_dir=None, settings_file=None):
        print(settings_file)
        super().__init__(output_str, output_dir=None, settings_file=settings_file)
        print(self.settings)
        self.image2 = visual.ImageStim(self.win, 
                self.settings['pile'].get('image2'),
                size=self.settings['pile'].get('dot_size'))

    def create_trials(self, durations=(2., 5.), timing='seconds'):
        n_dummies = self.settings['mri'].get('n_dummy_scans')

        phase_durations = [10000] * (n_dummies + 1)

        design = self.settings['mapper'].get('design')
        n_blocks = self.settings['mapper'].get('n_repeats_blocks')
        block_length = len(self.settings['mapper'].get('design'))

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
    session = MapperSession(f'sub-{subject}_ses-{session}', settings_file=settings)
    session.create_trials(durations=(.4, .6), timing='seconds')

    session.run()
    session.quit()
