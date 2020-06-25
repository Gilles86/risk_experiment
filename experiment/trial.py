from exptools2.core import Trial
from psychopy.visual import TextStim

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
