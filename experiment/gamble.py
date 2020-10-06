from exptools2.core import Session
from exptools2.core import Trial
from psychopy.visual import Pie, TextStim
from utils import run_experiment, create_stimulus_array_log_df
from stimuli import _create_stimulus_array, CertaintyStimulus, ProbabilityPieChart
import numpy as np
import os.path as op
import pandas as pd
from trial import InstructionTrial

class IntroBlockTrial(Trial):

    def __init__(self, session, trial_nr, phase_durations=[5.],
                 prob1=0.55, prob2=1.0, n_trials=16, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)

        txt = f"""
        In this block of {n_trials} trials, the first option will have a
        winning chance of {int(prob1*100):d}%.\n\n
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
            phase_durations = [.25, .3, .3, .5, .6, jitter1, .3, .3, .6, jitter2]
        elif len(phase_durations) == 12:
            phase_durations = phase_durations
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

        self.last_key_responses = dict(zip(self.buttons + [self.session.mri_trigger], [0.0] * 5))

    def draw(self):

        self.session.fixation_lines.draw()

        if self.phase == 0:
            self.session.fixation_lines.setColor((-1, 1, -1))
        elif self.phase == 1:
            self.session.fixation_lines.setColor((1, -1, -1))
        elif self.phase == 2:
            self.piechart1.draw()
        elif self.phase == 4:
            self.pile1.draw()
        elif self.phase == 6:
            self.piechart2.draw()
        elif self.phase == 8:
            self.pile2.draw()

        if self.phase == 9:
            if self.choice is not None:
                if (self.session.clock.getTime() - self.choice_time) < .5:
                    self.choice_stim.draw()
                else:
                    if (self.session.clock.getTime() - self.certainty_time) < .5:
                        self.certainty_stim.draw()


    def get_events(self):
        events = super().get_events()

        for key, t in events:
            if key not in self.last_key_responses:
                self.last_key_responses[key] = t - 0.6

            if t - self.last_key_responses[key] > 0.5:
                if self.phase > 7:
                    if self.choice is None:
                        if key in [self.buttons[0], self.buttons[1]]:
                            self.choice_time = self.session.clock.getTime()
                            if key == self.buttons[0]:
                                self.choice = 1
                            elif key == self.buttons[1]:
                                self.choice = 2
                            self.choice_stim.text = f'You chose pile {self.choice}'

                            self.log(choice=self.choice)

                    elif (self.phase > 8) & (self.certainty is None) & ((self.session.clock.getTime() - self.certainty_time) < .5):
                        if key in self.buttons:
                            self.certainty_time = self.session.clock.getTime()
                            self.certainty = self.buttons.index(key)
                            self.certainty_stim.rectangles[self.certainty].opacity = 1.0
                            self.log(certainty=self.certainty+1)

            self.last_key_responses[key] = t

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


