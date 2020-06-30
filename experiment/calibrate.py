from exptools2.core import Session
from exptools2.core import Trial
from psychopy.visual import Pie, TextStim
from trial import InstructionTrial
from session import PileSession
from utils import run_experiment
from stimuli import _create_stimulus_array, CertaintyStimulus, ProbabilityPieChart
import numpy as np
from psychopy import logging
import os.path as op
import pandas as pd

class CalibrationSession(PileSession):

    
    def create_trials(self):

        n_dummies = self.settings['mri'].get('n_dummy_scans')
        phase_durations = [np.inf] * (n_dummies + 1)

        txt = """
        In this task, you will see two piles of swiss Franc coins in
        succession. Both piles are combined with a pie chart in.
        The amount of the pile chart that is lightly colored indicates
        the probability of a lottery you will gain the amount of 
        Swiss Francs represented by the pile.
        One of the two piles has a probability of 100% for payout.
        The other probability changes every 16 trials.

        Your task is to either select the first lottery or
        the second lottery, by using your index or middle finger.
        Immediately after your choice, we ask how certain you were
        about your choice from a scale from 1 (very uncertain)
        to 4 (very certain).

        """

        trial_settings = pd.read_csv(op.abspath(op.join('settings',
            f'sub-{self.subject}_ses-calibrate.tsv')), sep='\t')

        self.trials = []

        jitter1 = self.settings['calibrate'].get('jitter1')
        jitter2 = self.settings['calibrate'].get('jitter2')

        trial_settings = trial_settings[trial_settings.trial < 100]
        for (run, p1, p2), d in trial_settings.groupby(['run', 'p1', 'p2']):
            self.trials.append(IntroBlockTrial(session=self, trial_nr=run,
                prob1=p1,
                prob2=p2))

            for ix, row in d.iterrows():
                print(row)
                self.trials.append(GambleTrial(self, row.trial,
                    prob1=row.p1, prob2=row.p2,
                    num1=int(row.n1), 
                    num2=int(row.n2),
                    jitter1=jitter1,
                    jitter2=jitter2))

class IntroBlockTrial(Trial):

    def __init__(self, session, trial_nr, phase_durations=[5.], 
            prob1=0.55, prob2=1.0, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)

        txt = f"""
        In this block of 16 trials, the first option will have a
        winning change of {int(prob1*100):d}%.\n
        The second option will have a winning chance of {int(prob2*100):d}%.
        """

        text_width = self.session.settings['various'].get('text_width')
        text_height = self.session.settings['various'].get('text_height')
        piechart_width = self.session.settings['various'].get('piechart_width')

        piechart_pos1 = .6 * -text_width - .5 * piechart_width, 1.5 * piechart_width
        piechart_pos2 = .6 * -text_width - .5 * piechart_width, -1.5 * piechart_width

        self.piechart1 = ProbabilityPieChart(self.session.win, prob1, pos=piechart_pos1, size=piechart_width)
        self.piechart2 = ProbabilityPieChart(self.session.win, prob2, pos=piechart_pos2, size=piechart_width)

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
            raise Exception("Don't directly set phase_durations for GambleTrial!")

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        self.parameters['prob1'] = prob1
        self.parameters['prob2'] = prob2

        self.parameters['n1'] = num1
        self.parameters['n2'] = num2

        self.buttons = self.session.settings['various'].get('buttons')
        piechart_width = self.session.settings['various'].get('piechart_width')

        self.piechart1 = ProbabilityPieChart(self.session.win, prob=prob1, size=piechart_width)
        self.piechart2 = ProbabilityPieChart(self.session.win, prob=prob2, size=piechart_width)

        # self.text1 = TextStim('Pie1: {int(

        self.pile1 = _create_stimulus_array(self.session.win,
            num1,
            self.session.settings['pile'].get('aperture_size'),
            self.session.settings['pile'].get('dot_size')/2.,
            image=self.session.image1)

        self.pile2 = _create_stimulus_array(self.session.win,
            num2,
            self.session.settings['pile'].get('aperture_size'),
            self.session.settings['pile'].get('dot_size')/2.,
            image=self.session.image1)


        self.choice_stim = TextStim(self.session.win)
        button_size = self.session.settings['various'].get('button_size')
        self.certainty_stim = CertaintyStimulus(self.session.win, response_size=[button_size, button_size])

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

        if self.choice is not None:
            if (self.session.clock.getTime() - self.choice_time) < .5:
                self.choice_stim.draw()
            else:
                if (self.session.clock.getTime() - self.certainty_time) < .5:
                    self.certainty_stim.draw()


    def get_events(self):
        events = super().get_events()

        for key, t in events:
            if self.phase > 4:
                if self.choice is None:
                    if key in [self.buttons[0], self.buttons[1]]:
                        self.choice_time = self.session.clock.getTime()
                        if key == self.buttons[0]:
                            self.choice = 1
                        elif key == self.buttons[1]:
                            self.choice = 2
                        
                        self.parameters['choice'] = self.choice
                        self.choice_stim.text = f'You chose pile {self.choice}'
                elif self.certainty is None:
                    if key in self.buttons:
                        self.certainty_time = self.session.clock.getTime()
                        self.certainty = self.buttons.index(key)
                        self.parameters['certainty'] = self.choice
                        self.certainty_stim.rectangles[self.certainty].opacity = 1.0
                        

        return events


if __name__ == '__main__':

    session_cls = CalibrationSession
    name = 'calibration'
    run_experiment(session_cls, name)
