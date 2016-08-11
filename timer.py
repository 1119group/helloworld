import time
import sys
from quantum_module import sec_to_human_readable_format


class Timer():
    """
    The timer object times the operations and prints out a nice progress bar.
    At the end of the operation, the timer will print out the total time
    taken so far.

    Usage: Initialize the class by calling timer = Timer(total_number_of_jobs),
           then at the end of each job (such as in a for loop after every
           task) invoke timer.progress() and the progress bar and the time
           estimation will be updated.

           At the end of the program, that is after all specified jobs
           are finished, the total elapsed time will be automatically shown
           unless the show_elapsed option is set to False.
    """

    def __init__(self, total, barlength=25, show_elapsed=True):
        """
        Initializes the timer object
        "total" is the total number of jobs that would take roughly
        the same amount of time to finish.
        "barlength" is the length of the bar that will be shown on screen.
        The default is 25.
        "show_elapsed" controls whether the final elapsed time is shown
        after the program finishes. The default is "True."
        """
        self.start_time = time.time()
        self.iteration = 0
        self.total = total
        self.barlength = barlength
        self.show_elapsed = show_elapsed
        self.show_progress()

    def update_progress(self):
        """
        Increments self.iteration by 1 every time this method is called.
        """
        self.iteration += 1

    def show_progress(self):
        """
        Shows the progress bar on screen. When being called after the
        first time, it updates the progress bar.
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
        """Prints the progress on screen"""
        # Update the progress.
        if self.iteration < self.total:
            self.update_progress()
            self.show_progress()
        if self.iteration == self.total and self.show_elapsed:
            self.show_elapsed_time()
