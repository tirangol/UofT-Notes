"""Cantus Firmus

Classes:
- Voice
"""

from scale import *
import math


class Voice:
    """A musical voice in first-species counterpoint.

    Lacks rhythm of any sort - all notes are equal length.

    Instance Attributes:
    - scale: the voice's key (in the form of a scale)
    - analysis: an analysis of the voice in terms of cantus firmus first species
    - as_note: the melody as a list of Notes
    - as_str: the melody as a list of string notes
    - as_int: the melody as a list of integer notes (no octave data stored)
    """
    scale: Optional[Scale]
    analysis: dict
    as_note: list[Note]
    as_str: list[str]
    as_int: list[int]

    def __init__(self, notes: list[Union[Note, str]], scale: Optional[Scale] = None) -> None:
        """Initialize this voice.

        >>> v1 = Voice(['D3', 'E3', 'F3', 'G3', 'E3', 'C3', 'D3'])
        >>> v2 = Voice(['E4', 'A4', 'B4', 'G4', 'D5', 'C5', 'F5', 'E5', 'A4', 'A5', 'G5', 'F5', \
        'E5', 'D5', 'E5', 'D5', 'B4', 'F#4', 'D5', 'C5', 'B4', 'A4', 'G#4', 'A4'])
        >>> v3 = Voice(['Bb3', 'Bb4', 'B4', 'Eb3', 'Eb3', 'A4', 'Bb4', 'B4', 'F#4', 'C#4', 'A#4'])
        """
        self.scale = scale
        self.as_note, self.as_str, self.as_int = [], [], []
        for n in notes:
            note = to_note(n)
            self.as_note.append(note)
            self.as_str.append(str(note))
            self.as_int.append(note.note_int)
        self.analysis = self.analyze()

    def __str__(self) -> str:
        """Return a string representation of the voice."""
        return str(self.as_str)

    def analyze(self) -> dict[str, Union[str, int, bool, float]]:
        """Return a summary of statistics about the voice."""
        c1, c2 = self.melody_range_calculator()
        c3, c4 = self.melody_spread_calculator()
        c5 = self.as_str.count(str(min(self.as_note)))  # occurrences of lowest note
        c6, c7, c8, c9 = self.vocal_range_calculator()
        c10, c11, c12, c13, c14, c15, c16 = self.iteration_calculator()
        c17, c18, c19, c20 = self.runs_calculator()

        return {
            'melodic range': c1,
            'sufficient melodic range': c2,
            'standard deviation of notes': round(c3, 2),
            'sufficient standard deviation of notes': c4,
            'occurrences of lowest note': c5,
            'soprano range': c6,
            'alto range': c7,
            'tenor range': c8,
            'bass range': c9,
            'repeated notes': c10,
            'conjunct intervals': c11,
            'disjunct intervals': c12,
            'predominantly conjunct intervals': c11 > c12,
            'leap interval rule violations': c13,
            'prohibited intervals': c14,
            'M6s': c15,
            'descending P5 + m3s': c16,
            'prohibited one-directional start/end intervals': c17,
            'consecutive one-directional disjunct intervals': c18,
            'consecutive m2s (chromatic progressions)': c19,
            'consecutive P4s': c20
        }

####################################################################################################
# Parameter Calculator
####################################################################################################
    def melody_range_calculator(self) -> tuple[int, bool]:
        """Return statistics related to note range."""
        total_melodic_range = interval(min(self.as_note), max(self.as_note))

        if len(self.as_note) >= 20:  # Very arbitrary decisions on "sufficient" melodic range
            sufficient_melodic_range = total_melodic_range >= 11
        elif len(self.as_note) <= 2:
            sufficient_melodic_range = total_melodic_range >= 1
        else:
            sufficient_melodic_range = total_melodic_range >= math.floor(
                10 * math.log(len(self.as_note) + 9) - 23)
        return total_melodic_range, sufficient_melodic_range

    def melody_spread_calculator(self) -> tuple[float, bool]:
        """Return statistics related to note spread."""
        mean = sum(x.absolute() for x in self.as_note) / len(self.as_note)
        sd = (sum(abs(x.absolute() - mean) ** 2 for x in self.as_note) / len(self.as_note)) ** 0.5

        if len(self.as_note) <= 10:  # Completely arbitrary
            sufficient_sd = sd >= 2
        elif len(self.as_note) >= 20:
            sufficient_sd = sd >= 3
        else:
            sufficient_sd = sd >= 2.5

        return sd, sufficient_sd

    def vocal_range_calculator(self) -> tuple[bool, bool, bool, bool]:
        """Return statistics related to the vocal singing range."""
        return all('C4' <= x <= 'A5' for x in self.as_note), all(
            'F3' <= x <= 'D5' for x in self.as_note), all(
            'C3' <= x <= 'A4' for x in self.as_note), all(
            'E2' <= x <= 'D4' for x in self.as_note)

    def iteration_calculator(self) -> tuple[int, int, int, list[tuple[str, str, str, int]], list[
            tuple[str, str, int]], list[tuple[str, str, int]], list[tuple[str, str, str, int]]]:
        """Return statistics obtained through iterating through all notes."""
        repeated_notes = 0
        conjunct_intervals = 0
        disjunct_intervals = 0
        forbidden_leap_intervals = []
        forbidden_intervals = []
        major_6ths = []
        descending_5th_3rd = []

        for i in range(len(self.as_str)):
            if i != 0:  # For statistics that compare 2 neighbouring notes
                n1, n2 = self.as_note[i - 1], self.as_note[i]

                repeated_notes += n1 == n2

                if forbidden_interval(n1, n2):
                    forbidden_intervals.append((str(n1), str(n2), i - 1))
                if abs(interval(n1, n2)) == 9:
                    major_6ths.append((str(n1), str(n2), i - 1))
                if abs(interval(n1, n2)) <= 2:
                    conjunct_intervals += 1
                else:
                    disjunct_intervals += 1
            if i >= 2:  # For statistics that compare 3 neighbouring notes
                n1, n2, n3 = self.as_note[i - 2], self.as_note[i - 1], self.as_note[i]

                if interval(n1, n2) in {12, -12, 8} and not n1 < n3 < n2 and not n2 < n3 < n1:
                    forbidden_leap_intervals.append((str(n1), str(n2), str(n3), i - 2))

                if interval(n1, n3) == -10 and abs(interval(n1, n2)) == 7:
                    descending_5th_3rd.append((str(n1), str(n2), str(n3), i - 2))

        return (repeated_notes, conjunct_intervals, disjunct_intervals, forbidden_leap_intervals,
                forbidden_intervals, major_6ths, descending_5th_3rd)

    def runs_calculator(self) -> tuple[list[tuple[list[str], int]], list[tuple[int, int]],
                                       list[tuple[list[str], int]], list[tuple[list[str], int]]]:
        """Return statistics obtained through analyzing note runs."""
        runs = [x for x in find_runs(self.as_note) if len(x) >= 2]
        forbidden_run_intervals = []
        consecutive_disjunct_intervals = []
        chromatics = []
        quartals = []

        for r in runs:
            run, run_ind = r
            # Check run's first and last notes - they can't be a compound interval, M7, or A4
            try:
                if interval_type(run[0], run[-1]) in {'M7', 'A4'} or \
                        abs(interval(run[0], run[-1])) > 12:
                    forbidden_run_intervals.append(([str(x) for x in run], run_ind))
            except ValueError:
                forbidden_run_intervals.append(([str(x) for x in run], run_ind))

            # Count number of consecutive skips in the same direction
            skips = 0
            for i in range(1, len(run)):
                if abs(interval(run[i - 1], run[i])) >= 3:
                    skips += 1
                else:
                    skips = 0
            if skips >= 3:
                consecutive_disjunct_intervals.append((skips, run_ind))

            # Count number of consecutive chromatics/quartals in the same direction
            _consecutives(r, chromatics, {1, -1})
            _consecutives(r, quartals, {5, -5})
        return forbidden_run_intervals, consecutive_disjunct_intervals, chromatics, quartals


####################################################################################################
# Helper Functions
####################################################################################################
def _consecutives(run: tuple[list, int], lst: list, s: set[int]) -> None:
    """Helper function for calculating consecutive quartals and chromatics."""
    run_lst = run[0]
    x = [str(run_lst[0])]

    for i in range(1, len(run_lst)):
        if len(x) == 0:                                           # len 0
            x.append(str(run_lst[i]))
        elif len(x) == 1 and interval(x[-1], run_lst[i]) in s:    # len 1, chromatic
            x.append(str(run_lst[i]))
        elif len(x) >= 2 and (interval(x[-2], x[-1]) == interval(x[-1], run_lst[i]) and
                              interval(x[-1], run_lst[i]) in s):  # len2, chromatic, same direction
            x.append(str(run_lst[i]))
        elif len(x) >= 2 and interval(x[-1], run_lst[i]) in s:    # len2, chromatic, wrong direction
            if len(x) >= 3:
                lst.append((x, run[1] + i - len(x)))
            x = [x[-1], str(run_lst[i])]
        else:                                                     # not chromatic
            if len(x) >= 3:
                lst.append((x, run[1] + i - len(x)))
            x = [str(run_lst[i])]
    if len(x) >= 3:
        lst.append((x, run[1] + len(run_lst) - len(x)))


def find_runs(lst: list) -> list[tuple[list, int]]:
    """Find all one-directional sequences of notes in the given voice line."""
    runs = []

    current_run = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1]:  # Next note == current
            current_run.append(lst[i])
        elif lst[i] > lst[i - 1]:  # Next note > current
            current_run = _find_runs(runs, current_run, lst, i, True)
        elif lst[i] < lst[i - 1]:  # Next note < current
            current_run = _find_runs(runs, current_run, lst, i, False)

    runs.append((current_run, len(lst) - len(current_run)))
    return runs


def _find_runs(runs: list, current_run: list, lst: list, i: int, above: bool) -> list:
    """Helper function for find_runs()."""
    if len(current_run) == 1:
        return current_run + [lst[i]]
    else:
        # If current run is previously going the opposite way, new note is bad
        for j in range(1, len(current_run)):
            if ((not above) and current_run[-1] > current_run[-j - 1]) or \
                    (above and current_run[-1] < current_run[-j - 1]):
                # Save current run and index of its first note
                runs.append((current_run, i - len(current_run)))
                return [current_run[-1], lst[i]]
        # If current run is previously going the same way
        return current_run + [lst[i]]


def forbidden_interval(n1: Union[Note, str], n2: Union[Note, str]) -> bool:
    """Return whether an interval is forbidden.

    The following intervals are foreign to the style of first species cantus firmus:
    - Descending minor 6ths
    - Any 7ths
    - Any augmented/diminished intervals
    - Any compound intervals (more than P8)
    """
    interval_int = interval(n1, n2)
    c1 = interval_int == -8             # Descending m6
    c2 = abs(interval_int) in {10, 11}  # m7 or M7
    c4 = abs(interval_int) > 12         # Compound interval

    try:
        interval_str = interval_type(n1, n2)
        c3 = interval_str[0] in {'d', 'D', 'a', 'A'}  # Augmented/diminished interval
    except ValueError:
        c3 = True

    return c1 or c2 or c3 or c4
