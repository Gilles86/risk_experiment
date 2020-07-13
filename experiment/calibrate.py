from exptools2.core import Session
from exptools2.core import Trial
from psychopy.visual import Pie, TextStim
from trial import InstructionTrial
from session import PileSession
from utils import run_experiment, create_stimulus_array_log_df
from stimuli import _create_stimulus_array, CertaintyStimulus, ProbabilityPieChart
import numpy as np
from psychopy import logging
import os.path as op
import pandas as pd


class IntroBlockTrial(Trial):

    def __init__(self, session, trial_nr, phase_durations=[5.],
                 prob1=0.55, prob2=1.0, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)

        txt = f"""
        In this block of 16 trials, the first option will have a
        winning change of {int(prob1*100):d}%.\n\n
        The second option will have a winning chance of {int(prob2*100):d}%.
        """

        text_width = self.session.settings['various'].get('text_width')
        text_height = self.session.settings['various'].get('text_height')
        piechart_width = self.session.settings['various'].get('piechart_width')

        piechart_pos1 = .5 * -text_width - .25 * piechart_width, 1.5 * piechart_width
        piechart_pos2 = .5 * -text_width - .25 * piechart_width, -1.5 * piechart_width

        self.piechart1 = ProbabilityPieChart(
            self.session.win, prob1, pos=piechart_pos1, size=piechart_width)
        self.piechart2 = ProbabilityPieChart(
            self.session.win, prob2, pos=piechart_pos2, size=piechart_width)

        self.text = TextStim(self.session.win, txt,
                             wrapWidth=text_width,
                             height=text_height)

    def draw(self):
        self.text.draw()
        self.piechart1.draw()
        self.piechart2.draw()


class GambleTrial(Trial):
    def __init__(self, session, trial_nr, phase_durations=None,
                 prob1=0.55, prob2=1.0, num1=10, num2=5,
                 jitter1=2.5, jitter2=4.0, **kwargs):

        if phase_durations is None:
            phase_durations = [.3, .5, .6, jitter1, .3, .5, .6, jitter2]
        else:
            raise Exception(
                "Don't directly set phase_durations for GambleTrial!")

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        self.parameters['prob1'] = prob1
        self.parameters['prob2'] = prob2

        self.parameters['n1'] = num1
        self.parameters['n2'] = num2

        self.buttons = self.session.settings['various'].get('buttons')
        piechart_width = self.session.settings['various'].get('piechart_width')

        self.piechart1 = ProbabilityPieChart(
            self.session.win, prob=prob1, size=piechart_width)
        self.piechart2 = ProbabilityPieChart(
            self.session.win, prob=prob2, size=piechart_width)

        # self.text1 = TextStim('Pie1: {int(

        self.pile1 = _create_stimulus_array(self.session.win,
                                            num1,
                                            self.session.settings['pile'].get(
                                                'aperture_radius'),
                                            self.session.settings['pile'].get(
                                                'dot_radius'),
                                            image=self.session.image1)

        self.pile2 = _create_stimulus_array(self.session.win,
                                            num2,
                                            self.session.settings['pile'].get(
                                                'aperture_radius'),
                                            self.session.settings['pile'].get(
                                                'dot_radius'),
                                            image=self.session.image1)
        self.stimulus_arrays = [self.pile1, self.pile2]

        self.choice_stim = TextStim(self.session.win)
        button_size = self.session.settings['various'].get('button_size')
        self.certainty_stim = CertaintyStimulus(
            self.session.win, response_size=[button_size, button_size])

        self.choice = None
        self.certainty = None
        self.certainty_time = np.inf

    def draw(self):
        self.session.fixation_lines.draw()

        if self.phase == 0:
            self.piechart1.draw()
        elif self.phase == 2:
            self.pile1.draw()
        elif self.phase == 4:
            self.piechart2.draw()
        elif self.phase == 6:
            self.pile2.draw()

        if self.phase == 7:
            if self.choice is not None:
                if (self.session.clock.getTime() - self.choice_time) < .5:
                    self.choice_stim.draw()
                else:
                    if (self.session.clock.getTime() - self.certainty_time) < .5:
                        self.certainty_stim.draw()

    def get_events(self):
        events = super().get_events()

        for key, t in events:
            if self.phase > 5:
                if self.choice is None:
                    if key in [self.buttons[0], self.buttons[1]]:
                        self.choice_time = self.session.clock.getTime()
                        if key == self.buttons[0]:
                            self.choice = 1
                        elif key == self.buttons[1]:
                            self.choice = 2
                        self.choice_stim.text = f'You chose pile {self.choice}'

                        self.log(choice=self.choice)

                elif (self.phase > 6) & (self.certainty is None) & ((self.session.clock.getTime() - self.certainty_time) < .5):
                    if key in self.buttons:
                        self.certainty_time = self.session.clock.getTime()
                        self.certainty = self.buttons.index(key)
                        self.certainty_stim.rectangles[self.certainty].opacity = 1.0
                        self.log(certainty=self.certainty+1)

        return events

    def get_stimulus_array_log(self):

        n_dots1 = self.parameters['n1']
        n_dots2 = self.parameters['n2']

        # trial_ix, stim_array, stimulus_ix
        trial_ix = np.ones(n_dots1 + n_dots2) * self.trial_nr
        array_ix = [1] * n_dots1 + [2] * n_dots2
        stim_ix = np.hstack((np.arange(n_dots1) + 1, np.arange(n_dots2)+1))

        index = pd.MultiIndex.from_arrays([trial_ix, array_ix, stim_ix],
                                          names=('trial_nr', 'array_nr', 'stim_nr'))

        log = create_stimulus_array_log_df(self.stimulus_arrays, index=index)

        return log

    def log(self, choice=None, certainty=None):

        if (choice is not None) or (certainty is not None):
            onset = self.session.clock.getTime()
            idx = self.session.global_log.shape[0]
            self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
            self.session.global_log.loc[idx, 'onset'] = onset
            self.session.global_log.loc[idx, 'phase'] = self.phase
            self.session.global_log.loc[idx,
                                        'nr_frames'] = self.session.nr_frames

        if choice is not None:
            self.session.global_log.loc[idx, 'event_type'] = 'choice'
            self.session.global_log.loc[idx, 'choice'] = choice
        if certainty is not None:
            self.session.global_log.loc[idx, 'event_type'] = 'certainty'
            self.session.global_log.loc[idx, 'certainty'] = certainty


class CalibrationSession(PileSession):

    Trial = GambleTrial

    def create_trials(self):

        n_dummies = self.settings['mri'].get('n_dummy_scans')
        phase_durations = [np.inf] * (n_dummies + 1)

        txt = """
        In this task, you will see two piles of swiss Franc coins in
        succession. Both piles are combined with a pie chart in.
        The part of the pie chart that is lightly colored indicates
        the probability of a lottery you will gain the amount of
        Swiss Francs represented by the pile.
        There is always one pile that has a probability of 100% for payout.
        The other probability changes every 16 trials.

        Your task is to either select the first lottery or
        the second lottery, by using your index or middle finger.
        Immediately after your choice, we ask how certain you were
        about your choice from a scale from 1 (very uncertain)
        to 4 (very certain).

        This is run {run}/3.

        Press any of your buttons to continue.

        """

        calibrate_settings_folder = op.abspath(
            op.join('settings', 'calibration'))
        trial_settings = pd.read_csv(op.abspath(op.join(calibrate_settings_folder,
                                                        f'sub-{self.subject}_ses-calibrate.tsv')), sep='\t')

        self.trials = []

        jitter1 = self.settings['calibrate'].get('jitter1')
        jitter2 = self.settings['calibrate'].get('jitter2')

        trial_settings = trial_settings

        for run, d in trial_settings.groupby(['run'], sort=False):
            self.trials.append(InstructionTrial(self, trial_nr=run,
                                                txt=txt.format(run=run)))
            for (p1, p2), d2 in d.groupby(['p1', 'p2'], sort=False):
                self.trials.append(IntroBlockTrial(session=self, trial_nr=run,
                                                   prob1=p1,
                                                   prob2=p2))

                for ix, row in d2.iterrows():
                    self.trials.append(GambleTrial(self, row.trial,
                                                   prob1=row.p1, prob2=row.p2,
                                                   num1=int(row.n1),
                                                   num2=int(row.n2),
                                                   jitter1=jitter1,
                                                   jitter2=jitter2))


if __name__ == '__main__':

    session_cls = CalibrationSession
    task = 'calibration'
    run_experiment(session_cls, task=task)
