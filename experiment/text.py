from exptools2.core import Session
from trial import InstructionTrial
import numpy as np

class TextSession(Session):

    def __init__(self, txt, output_str, output_dir, settings_file):
        self.txt = txt
        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file)
        self.trials = [InstructionTrial(self, 0, txt=self.txt, keys='q', phase_durations=[np.inf])]

    def run(self):
        """ Runs experiment. """
        self.start_experiment()

        for trial in self.trials:
            trial.run()

        self.close()


if __name__ == '__main__':
    session = TextSession(txt='hallo', output_str='txt', output_dir='log', settings_file='settings/default.yml')
    session.run()
