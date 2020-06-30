from exptools2.core import Session
from exptools2.core import Trial
from psychopy.visual import Pie, TextStim
from trial import InstructionTrial
from session import PileSession
from utils import run_experiment
from stimuli import _create_stimulus_array, CertaintyStimulus, ProbabilityPieChart
import numpy as np
from psychopy import logging

class CalibrationSession(PileSession):

    
    def create_trials(self):

        n_dummies = self.settings['mri'].get('n_dummy_scans')
        phase_durations = [np.inf] * (n_dummies + 1)

        txt = """
        In this task, you will see two piles of swiss Franc coins in
        succession combined with a pie chart in the middle of the screen.
        The amount of the pile chart that is colored indicates
        the probability of a lottery you will gain that amount of 
        Swiss Francs.
        Note that always one of the two piles has a probability of
        100% for payout. The other probability changes every 
        16 trials.
        """
        # self.trials = [InstructionTrial(session=self, trial_nr=0,
            # phase_durations=phase_durations, txt=txt)]
        self.trials = [IntroBlockTrial(session=self, trial_nr=0),
                GambleTrial(self, 1),
                GambleTrial(self, 2),
                GambleTrial(self, 3)]


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

        piechart_pos1 = .6 * -text_width - .5 * piechart_width, .75 * piechart_width
        piechart_pos2 = .6 * -text_width - .5 * piechart_width, -.75 * piechart_width

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
            jitter1=4, jitter2=4, **kwargs):


        if phase_durations is None:
            phase_durations = [.5, .5, .3, jitter1, .5, .5, .3, jitter2] 
        else:
            raise Exception("Don't directly set phase_durations for GambleTrial!")

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        self.parameters['prob1'] = prob1
        self.parameters['prob2'] = prob2

        self.buttons = self.session.settings['various'].get('buttons')
        piechart_width = self.session.settings['various'].get('piechart_width')

        self.piechart1 = ProbabilityPieChart(self.session.win, prob=prob1, size=piechart_width)
        self.piechart2 = ProbabilityPieChart(self.session.win, prob=prob2, size=piechart_width)

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
        self.certainty_stim = CertaintyStimulus(self.session.win, response_size=[4, 1])

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
        elif self.phase == 5:
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
