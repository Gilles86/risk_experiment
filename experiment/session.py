from stimuli import FixationLines
from exptools2.core import Session
from psychopy import visual

class PileSession(Session):
    """ Simple session with x trials. """
    def __init__(self, output_str, subject=None, output_dir=None, settings_file=None):
        """ Initializes TestSession object. """
        super().__init__(output_str, output_dir=None, settings_file=settings_file)
        self.subject = subject

        print(self.settings)

        self.fixation_lines = FixationLines(self.win,
                self.settings['pile'].get('aperture_radius')*2,
                color=(1, -1, -1))
        
        self.image1 = visual.ImageStim(self.win, 
                self.settings['pile'].get('image1'),
                texRes=32,
                size=self.settings['pile'].get('dot_radius'))

    def run(self):
        """ Runs experiment. """
        self.start_experiment()
        for trial in self.trials:
            trial.run()

        self.close()
