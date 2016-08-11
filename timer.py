import time
import sys
from quantum_module import sec_to_human_readable_format


class Timer():
    """
    The timer object times the operations and prints out a nice progress bar.
    At the end of the operation, the timer will print out the total time
    taken so far.
    """

    def __init__(self, total, barlength=25):
        """
        Initializes the timer object
        "total" is the total number of jobs that would take roughly
        the same amount of time to finish.
        "barlength" is the length of the bar that will be shown on screen.
        The default is 25.
        """
        self.start_time = time.time()
        self.iteration = 0
        self.total = total
        self.barlength = barlength
        self.show_progress()

    def update_progress(self):
        """
        Increments self.iteration by 1 every time this method is called.
        """
        self.iteration += 1

    def show_progress(self):
        """
        Shows the progress bar on screen. When being the called the
        after the first time, it updates the progress bar.
        """
        if self.iteration == 0:
            report_time = ""
            filledlength = 0
            percent = 0
        else:
            # Calculate time used for progress report purposes.
            elapsed = time.time() - self.start_time
            ET_sec = elapsed / (self.iteration) * (self.total - self.iteration)
            ET = sec_to_human_readable_format(ET_sec)

            report_time = "Est. time: " + ET
            filledlength = int(round(self.barlength * (self.iteration) /
                                     self.total))
            percent = round(100.00 * ((self.iteration) / self.total), 1)

        bar = '\u2588' * filledlength + '\u00B7' * (self.barlength
                                                    - filledlength)
        sys.stdout.write('\r%s |%s| %s%s %s' % ('Progress:', bar,
                                                percent, '%  ', report_time)),
        sys.stdout.flush()
        if self.iteration == self.total:
            sys.stdout.write('\n')
            sys.stdout.flush()

    def show_elapsed_time(self):
        """Prints the total elapsed time."""
        elapsed = time.time() - self.start_time
        elapsed_time = sec_to_human_readable_format(elapsed)
        print("\nTime elapsed: " + elapsed_time)

    def progress(self):
        """
        Prints the progress on screen. "start_time" is the the start time
        of the entire program. "iteration" is the job number of the
        current job. As with everything in Python, it is 0 based. For
        instance, if the current job is the first task, iteration=0.
        "total" is the total number of tasks to perform.
        The output is in the hh:mm:ss format.
        """
        # Update the progress.
        if self.iteration < self.total:
            self.update_progress()
            self.show_progress()
        if self.iteration == self.total:
            self.show_elapsed_time()
