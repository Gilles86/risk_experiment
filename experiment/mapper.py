import os.path as op
import pandas as pd
from exptools2.core import Trial
from psychopy import visual
import numpy as np
import argparse
from stimuli import _create_stimulus_array
from session import PileSession
from trial import InstructionTrial, DummyWaiterTrial, OutroTrial
from utils import run_experiment, sample_isis, create_stimulus_array_log_df


class MapperTrial(Trial):
    def __init__(self, session, trial_nr, phase_durations, colors,
                 n_dots=5, **kwargs):

        phase_durations = []

        mapper_settings = session.settings['mapper']

        for i in range(mapper_settings.get('n_repeats_stimulus')):
            phase_durations += [mapper_settings.get('on_duration'),
                                mapper_settings.get('off_duration')]

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        self.parameters['n_dots'] = n_dots
        self.colors = colors

        self.stimulus_arrays = [_create_stimulus_array(self.session.win, n_dots,
                                                       self.session.settings['pile'].get(
                                                           'aperture_radius'),
                                                       self.session.settings['pile'].get(
                                                           'dot_radius'),
                                                       image=[self.session.image1, self.session.image2][color])
                                for i, color in enumerate(self.colors)]

        print(self.get_stimulus_array_log())

    def draw(self):
        """ Draws stimuli """
        self.session.fixation_lines.draw()

        if self.phase % 2 == 0:
            self.stimulus_arrays[int(self.phase / 2)].draw()

    def log_phase_info(self, phase=None):
        self.parameters['color'] = self.colors[int(self.phase/2)]
        super().log_phase_info(phase=phase)

    def get_stimulus_array_log(self):

        n_repeats = len(self.colors)
        n_dots = self.parameters['n_dots']

        # trial_ix, stim_array, color, stimulus_ix
        trial_ix = np.ones(n_repeats * n_dots) * self.trial_nr
        array_ix = np.repeat(np.arange(n_repeats) + 1, n_dots)
        color_ix = np.repeat(self.colors, n_dots)
        stim_ix = np.tile(np.arange(n_dots) + 1, n_repeats)

        index = pd.MultiIndex.from_arrays([trial_ix, array_ix, color_ix, stim_ix],
                                          names=('trial_nr', 'array_nr', 'color', 'stim_nr'))

        log = create_stimulus_array_log_df(self.stimulus_arrays, index=index)

        return log


class MapperSession(PileSession):

    Trial = MapperTrial

    def __init__(self, output_str, subject=None, output_dir=None, settings_file=None, run=None, **kwargs):
        super().__init__(output_str, output_dir=output_dir,
                         settings_file=settings_file, run=run, **kwargs)

        self.image2 = visual.ImageStim(self.win,
                                       self.settings['pile'].get('image2'),
                                       texRes=32,
                                       size=self.settings['pile'].get('dot_radius'))

    def create_trials(self):

        txt = f"""
        This is run {self.settings['run']}/4.

        You will now see piles of one-CHF coins in rapid succession.
        Your task is to indicate every time you see coins that are a bit
        darker, by pressing the first button (index finger).\n
        It is important that you do not move your eyes. Keep looking
        at where the two red lines cross each other.
        Press any of your buttons to start.
        """

        self.trials = [InstructionTrial(session=self, trial_nr=0, txt=txt,)]

        n_dummies = self.settings['mri'].get('n_dummy_scans')
        self.trials.append(DummyWaiterTrial(
            session=self, n_triggers=n_dummies, trial_nr=0))

        design = self.settings['mapper'].get('design')
        n_blocks = self.settings['mapper'].get('n_repeats_blocks')
        block_length = len(self.settings['mapper'].get('design'))
        n_repeats_stimulus = self.settings['mapper'].get('n_repeats_stimulus')

        colors = sample_isis(n_blocks * block_length * n_repeats_stimulus)

        for block in range(n_blocks):
            for trial_nr, n_dots in enumerate(design):
                trial_nr += block*block_length + 1

                color_ix = (trial_nr-1) * \
                    n_repeats_stimulus, trial_nr*n_repeats_stimulus
                self.trials.append(
                    self.Trial(session=self,
                               trial_nr=trial_nr,
                               phase_durations=[],
                               n_dots=n_dots,
                               colors=colors[color_ix[0]:color_ix[1]],
                               verbose=True,)
                )

        outro_trial = OutroTrial(session=self, trial_nr=n_blocks*len(design)+1,
                                       phase_durations=[np.inf])
        self.trials.append(outro_trial)


if __name__ == '__main__':
    session_cls = MapperSession
    task = 'mapper'
    run_experiment(session_cls, task=task, n_runs=3)
