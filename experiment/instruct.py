from session import PileSession
from gamble import IntroBlockTrial, GambleTrial
from utils import get_output_dir_str, create_design, sample_isis
from psychopy.visual import TextStim, ImageStim
from psychopy import logging
import os.path as op
import argparse
from trial import InstructionTrial
import numpy as np
from exptools2.core import Trial
from mapper import MapperTrial


class GambleInstructTrial(GambleTrial):

    def __init__(self, session, trial_nr, txt, bottom_txt=None, show_phase=0, keys=None, **kwargs):

        phase_durations = np.ones(12) * 1e-6
        phase_durations[show_phase] = np.inf
        self.keys = keys

        super().__init__(session, trial_nr, phase_durations=phase_durations, **kwargs)

        txt_height = self.session.settings['various'].get('text_height')
        txt_width = self.session.settings['various'].get('text_width')

        self.text = TextStim(session.win, txt,
                             pos=(0.0, 6.0), height=txt_height, wrapWidth=txt_width, color=(0, 1, 0))

        if bottom_txt is None:
            bottom_txt = "Press any button to continue"

        self.text2 = TextStim(session.win, bottom_txt, pos=(
            0.0, -6.0), height=txt_height, wrapWidth=txt_width,
            color=(0, 1, 0))

    def get_events(self):

        events = Trial.get_events(self)

        if self.keys is None:
            if events:
                self.stop_phase()
        else:
            for key, t in events:
                if key in self.keys:
                    self.stop_phase()

    def draw(self):
        if self.phase != 0:
            self.session.fixation_lines.setColor((1, -1, -1))

        if self.phase < 9:
            super().draw()
        else:
            self.session.fixation_lines.draw()
            if self.phase == 10:
                self.choice_stim.draw()
            elif self.phase == 11:
                self.certainty_stim.draw()

        self.text.draw()
        self.text2.draw()


class InstructionSession(PileSession):

    Trial = GambleTrial

    def __init__(self, output_str, subject=None, output_dir=None, settings_file=None):
        super().__init__(output_str, subject=subject,
                         output_dir=output_dir, settings_file=settings_file, run=run, eyetracker_on=False)

        logging.warn(self.settings['run'])
        self.buttons = self.settings['various'].get('buttons')
        self.image2 = ImageStim(self.win,
                                       self.settings['pile'].get('image2'),
                                       texRes=32,
                                       size=self.settings['pile'].get('dot_radius'))

    def create_trials(self):
        self.trials = []

        txt = """
        Hi!

        Welcome to the instruction part of the experiment. During this part, we will explain what a single trial in this experiment looks like, and what choices you can make.

        First of all, during the experiment, you will always have 4 buttons that you can press.

        On this computer, these are:
        * the j-key (index finger)
        * the k-key (middle finger)
        * the l-key (ring finger)
        * the ;-key (pinky)

        TIP: let your index-/middle-/ring-finger and pinky lie down on these keys during the experiment. Note that, on the computer, key 1 (j) has a little bar on top that you can feel.

        In the scanner, there will be a red, blue, yellow, and green key.

        From now on, we will call them key 1 (index finger), key 2 (middle finger), key 3 (ring finger), and key 4 (pinky).

        Press key 1 to continue.
        """

        self.trials.append(InstructionTrial(self, 1,
                                            txt=txt, keys=[self.buttons[0]]))

        for key in [2, 3, 4]:
            txt = f"""
            Press key {key} to continue.
            """

            self.trials.append(InstructionTrial(self, key,
                                                txt=txt, keys=[self.buttons[key-1]]))
        txt = """
        In this experiment, your task is to make choices between (i) getting a fixed amount of money or (ii) to participate in a lottery where you have 55% chance to win a substantially larger amount of money.

        During the course of the experiment, you will make many different choices. After each session, we will randomly select one such choice you made during the task. When you selected the 55%-lottery option, we will perform a digital lottery that determines whether you win the offered amount.

        *** WE WILL ADD UP THE AMOUNTS YOU COLLECTED ACROSS THE FOUR SESSIONS AND YOU WILL BE PAID OUT THE AVERAGE OF THOSE AMOUNTS AFTER THE FOURTH SESSION. ***

        Press key 1 to continue
        """

        self.trials.append(InstructionTrial(self, 5,
                                            txt=txt, keys=[self.buttons[0]]))

        txt = """

        Let's say you won 0 CHF on the first session, 100 CHF in the second session, 0 CHF in the third session and 100 CHF in the fourth session. How much actual money will you be paid out at the end of the fourth session, on top of your hourly 30CHF/hour rate?

        1. 100CHF
        2. 50CHF
        3. 0CHF

        Press the key that corresponds to the correct answer.
        """

        self.trials.append(InstructionTrial(self, 6,
                                            txt=txt, keys=[self.buttons[1]]))

        txt = """
        Please note that if you do not respond to a trial in time, you will earn 0 CHF if that trial gets selected. So be sure to always indicate a choice in time.

        Press key 1 to continue
        """

        self.trials.append(InstructionTrial(self, 7,
                                            txt=txt, keys=[self.buttons[0]]))

        txt = """

        Let's say that the first choice was a 55% probability to win 5CHF and the second option 100% probability to win 1CHF. You *do not respond within 3 seconds*. Now this trial gets selected after the experiment. How much money will you earn?

        1. I have a 55% probability of winning 5CHF
        2. I win 1 CHF
        3. I win 0 CHF

        Press the key that corresponds to the correct answer.
        """

        self.trials.append(InstructionTrial(self, 7,
                                            txt=txt, keys=[self.buttons[2]]))

        txt = """
        We will now take you through all the steps of a trial.

        Press key 1 to continue
        """

        self.trials.append(InstructionTrial(self, 8,
                                            txt=txt, keys=[self.buttons[0]]))

        txt = "A new trial always starts with a green fixation cross"

        self.trials.append(GambleInstructTrial(
            self, 9, txt, show_phase=0))

        txt = "After the fixation cross, you will see the probability of winning the offered amount for the FIRST option. In this case, this option offers a 100% probability of gaining the following amount in CHF."

        self.trials.append(GambleInstructTrial(
            self, 10, txt, prob1=1.0, show_phase=2))

        txt = "Now you will see the amount of money that you can potentially win when you choose the FIRST option (5 CHF), represented as the number of coins on the screen."

        self.trials.append(GambleInstructTrial(
            self, 11, txt, num1=5, show_phase=4))

        txt = "Now you will see a red cross for a while. Try to remember the amount of money that was offered in the first option!"

        self.trials.append(GambleInstructTrial(
            self, 12, txt, show_phase=5))

        txt = "Now you will see the probability of winning the offered amount for the SECOND option. In this case, this option offers a 55% probability of gaining the offered amount in CHF."

        self.trials.append(GambleInstructTrial(
            self, 13, txt, prob2=0.55, show_phase=6))

        txt = "Now you will see the amount of money that you can potentially win when you choose the SECOND option (10 CHF), represented as the number of coins on the screen"

        self.trials.append(GambleInstructTrial(
            self, 14, txt, num2=10, show_phase=8))

        txt = """
        Now it is your task to choose one of the two options.
        You can press key 1 to choose the first option or key 2 to choose the second option.
        """

        bottom_txt = "Press key 2 to choose the second option"

        self.trials.append(GambleInstructTrial(
            self, 15, txt, num2=10, show_phase=8, bottom_txt=bottom_txt, keys=[self.buttons[1]]))

        txt = "After you made your choice, you get reminded what you've chosen."

        trial16 = GambleInstructTrial(
            self, 16, txt, show_phase=10)
        trial16.choice = 2
        trial16.choice_stim.text = f'You chose pile {trial16.choice}'
        self.trials.append(trial16)

        txt = "after your choice, you should also indicate from a scale of one to four how certain you were about your choice."
        bottom_txt = "Press key 1 to indicate you were very certain of your choice"

        self.trials.append(GambleInstructTrial(
            self, 17, txt, show_phase=11, bottom_txt=bottom_txt, keys=[self.buttons[0]]))

        txt = "You now can see which certainty you've indicated"
        trial18 = GambleInstructTrial(
            self, 18, txt, show_phase=11)
        trial18.certainty = 1
        trial18.certainty_stim.rectangles[0].opacity = 1.0
        self.trials.append(trial18)

        txt = """
        We will now go through a trial again and we will ask you some questions about it at the end. So pay close attention!

        Press key 2 to continue.
        """

        self.trials.append(InstructionTrial(self, 19,
                                            txt=txt, keys=[self.buttons[1]]))

        for phase in range(8):
            txt = ''
            self.trials.append(GambleInstructTrial(
                self, 20+phase, txt, show_phase=phase, prob1=1., num1=2, prob2=.55, num2=4))

            
        txt = 'Choose the 1st option'
        bottom_txt = '(so press key 1)'
        self.trials.append(GambleInstructTrial(
            self, 28, txt, show_phase=8, prob1=1., num1=2, prob2=.55, num2=4, bottom_txt=bottom_txt, keys=self.buttons[0]))

        txt = ''
        trial29 = GambleInstructTrial(
            self, 29, txt, show_phase=10)
        trial29.choice = 1
        trial29.choice_stim.text = f'You chose pile {trial29.choice}'
        self.trials.append(trial29)


        bottom_txt = "Press key 2 to indicate you were somewhat certain of your choice"

        self.trials.append(GambleInstructTrial(
            self, 30, txt, show_phase=11, bottom_txt=bottom_txt, keys=[self.buttons[1]]))

        trial31 = GambleInstructTrial(
            self, 31, txt, show_phase=11)
        trial31.certainty = 2
        trial31.certainty_stim.rectangles[1].opacity = 1.0
        self.trials.append(trial31)


        txt = """
        Pick the correct answer:

        1. I had to choose between 4 CHF for sure,
        or a lottery with a 55% probability to win 4 CHF.

        2. I had to choose between 2 CHF for sure,
        or a lottery with a 55% probability to win 4 CHF.

        3. I had to choose between 4 CHF for sure,
        or 2 CHF for sure.

        Press the key that corresponds to the correct answer.
        """

        self.trials.append(InstructionTrial(self, 32,
                                            txt=txt, keys=[self.buttons[1]]))


        txt = """
        Pick the correct answer:

        I answered the certainty question with 2, this means that ...

        1. I was very certain about the choice I made.

        2. I was somewhat certain about the choice I made.

        3. I was very uncertain about the choice I made.

        Press the key that corresponds to the correct answer.
        """

        self.trials.append(InstructionTrial(self, 32,
                                            txt=txt, keys=[self.buttons[1]]))

        txt = """
        Well done!!

        You will now do 10 practice trials. Note that the trial now automatically goes forwad.

        You only have to respond twice:
        Once to indicate your choice between the two options and once to indicate how certain you were about your choice. 
        """

        self.trials.append(InstructionTrial(self, 33,
                                            txt=txt))



        h = np.arange(1, 9)
        fractions = 2**(h/4)
        trial_settings = create_design([.55, 1.], [1., .55], fractions=fractions, n_runs=1)
        trial_settings = trial_settings.sample(n=10)

        jitter1 = self.settings['calibrate'].get('jitter1')
        jitter2 = self.settings['calibrate'].get('jitter2')

        trial_nr = 34

        for (p1, p2), d2 in trial_settings.groupby(['p1', 'p2'], sort=False):
            n_trials_in_miniblock = len(d2)
            self.trials.append(IntroBlockTrial(session=self, trial_nr=trial_nr,
                                               n_trials=n_trials_in_miniblock,
                                               prob1=p1,
                                               prob2=p2))

            trial_nr += 1

            for ix, row in d2.iterrows():
                self.trials.append(GambleTrial(self, trial_nr,
                                               prob1=row.p1, prob2=row.p2,
                                               num1=int(row.n1),
                                               num2=int(row.n2),
                                               jitter1=jitter1,
                                               jitter2=jitter2))
                trial_nr += 1

        txt = f"""
        Well done!

        You now finished the training for the first part of the experiment.

        There is also a much simpler task that you will have to perform in the scanner
        as well.


        Press any button to continue...
        """

        self.trials.append(InstructionTrial(self, trial_nr=trial_nr, txt=txt))
        trial_nr += 1

        txt = f"""
        In this second "detection task", your job is to press KEY 1 every time that
        you see DARK coins. When you see the standard LIGHT coins, do not press
        any button.

        We will now practice this for some time...

        Press any button to continue...

        """

        self.trials.append(InstructionTrial(self, trial_nr=trial_nr, txt=txt))
        trial_nr += 1

        design = [5, 7, 10, 20, 28, 56, 28, 20, 10, 7, 5]
        n_blocks = 1
        block_length = len(design)
        n_repeats_stimulus = 6

        colors = sample_isis(n_blocks * block_length * n_repeats_stimulus)

        for block in range(n_blocks):
            for ix, n_dots in enumerate(design):
                ix += block*block_length + 1

                color_ix = (ix-1) * \
                    n_repeats_stimulus, ix*n_repeats_stimulus
                self.trials.append(
                    MapperTrial(session=self,
                               trial_nr=trial_nr,
                               phase_durations=[],
                               n_dots=n_dots,
                               colors=colors[color_ix[0]:color_ix[1]],
                               verbose=True,)
                )
                trial_nr += 1

        txt = f"""
        Well done!

        You came to the end of the instruction part of the experiment.

        You will now do the two tasks (the "gamble task" and the "detection task") in the scanner.

        In case anything is unclear. Please do not hesitate to ask the experimenters anything!

        """

        self.trials.append(InstructionTrial(self, trial_nr=trial_nr, txt=txt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default='instructee', nargs='?')
    parser.add_argument('--settings', default='instruction', nargs='?')
    cmd_args = parser.parse_args()

    subject, session, task, run = cmd_args.subject, 'instruction', 'instruction',  None
    output_dir, output_str = get_output_dir_str(subject, session, task, run)

    log_file = op.join(output_dir, output_str + '_log.txt')
    logging.warn(f'Writing results to: {log_file}')

    settings_fn = op.join(op.dirname(__file__), 'settings',
                          f'{cmd_args.settings}.yml')

    session_object = InstructionSession(output_str=output_str,
                                        output_dir=output_dir,
                                        settings_file=settings_fn, subject=subject)

    session_object.create_trials()
    print(session_object.trials)
    logging.warn(
        f'Writing results to: {op.join(session_object.output_dir, session_object.output_str)}')
    session_object.run()
    session_object.close()
