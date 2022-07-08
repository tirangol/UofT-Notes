"""Cantus Firmus

Classes:
- Note
- Interval

Functions:
- to_note(), to_interval()
- strnote_to_num(), num_to_strnote()
- interval(), interval_type(), absolute_letter_diff()
- strinterval_to_int(), break_strinterval_to_octaves()
- enharmonic(), find_enharmonic_equivalent()
"""

from typing import Any, Union, Optional


####################################################################################################
# Note
####################################################################################################
class Note:
    """A musical note.

    Supports the operations:
    - >, >=, <, <=
    - ==, equiv(), enharmonic()
    - +, -

    Instance Variables:
    - note: the note name
    - note_int: the note name, but converted to an integer (C = 0, C# = 1, D = 2, ..., B = 11)
    - octave: the note octave
    """
    note: str
    note_int: int
    octave: int

    def __init__(self, note: str) -> None:
        """Initialize the note.

        Preconditions:
        - 2 <= len(note) <= 4
        - note[0].isalpha()
        - all(not (x.isalpha() and x.isnumeric()) or x in {'#', 'b'} for x in note[1:-1])
        - note[-1].isnumeric()

        >>> n1 = Note('C#1')
        """
        if 'b' in note and '#' in note:
            raise ValueError("Note cannot have a sharp and flat simultaneously")
        i = 1
        try:
            while note[i] in '#b':
                i += 1
        except IndexError:
            raise ValueError("Invalid note input " + note)
        self.note = note[0].upper() + note[1:i]
        self.octave = int(note[i:])
        self.note_int = strnote_to_num(self.note)

    def __eq__(self, other: Any) -> bool:
        """Return whether a note n1 is enharmonically the same as n2.
        Automatically converts string inputs into a Note.

        >>> Note('C#1') == Note('Db1')
        True
        >>> Note('C#1') == 'Db1'
        True
        """
        if isinstance(other, Note):
            return self.octave == other.octave and self.note_int == other.note_int
        if isinstance(other, str):
            return Note(other) == self
        raise TypeError('Cannot compare Note to non-Note object ' + str(other))

    def equiv(self, other: Any) -> bool:
        """Return whether a note n1 is exactly the same as n2.
        Automatically converts string inputs into a Note.

        >>> Note('C#1') == Note('Db1')
        False
        >>> Note('C#1') == 'C#1'
        True
        """
        if isinstance(other, Note):
            return self.octave == other.octave and self.note == other.note
        if isinstance(other, str):
            return Note(other).equiv(self)
        raise TypeError('Cannot compare Note to non-Note object ' + str(other))

    def enharmonic(self, other: Any) -> bool:
        """Return whether a note n1 is enharmonic to n2."""
        return self == other and not self.equiv(other)

    def same_letter_head(self, other: Any) -> bool:
        """Return whether a note n1 has the same letter head as n2."""
        if isinstance(other, Note):
            return self.note[0] == other.note[0]
        elif isinstance(other, str):
            return Note(other).same_letter_head(self)
        raise TypeError('Cannot compare Note to non-Note object ' + str(other))

    def __lt__(self, other: Any) -> bool:
        """Return whether a note n1 < n2.
        Automatically converts string inputs into a Note.

        >>> Note('C1') < 'B1'
        True
        """
        if isinstance(other, Note):
            c1 = self.octave <= other.octave
            c2 = not other.octave == self.octave or self.note_int < other.note_int
            return c1 and c2
        if isinstance(other, str):
            return self.__lt__(Note(other))
        raise TypeError('Cannot compare Note to non-Note object ' + str(other))

    def __gt__(self, other: Any) -> bool:
        """Return whether a note n1 > n2.

        Automatically converts string inputs into a Note.
        """
        if isinstance(other, Note):
            c1 = self.octave >= other.octave
            c2 = not other.octave == self.octave or self.note_int > other.note_int
            return c1 and c2
        if isinstance(other, str):
            return self.__gt__(Note(other))
        raise TypeError('Cannot compare Note to non-Note object ' + str(other))

    def __le__(self, other: Any) -> bool:
        """Return whether a note n1 <= n2.

        Automatically converts string inputs into a Note.
        """
        return self.__eq__(other) or self.__lt__(other)

    def __ge__(self, other: Any) -> bool:
        """Return whether a note n1 >= n2.

        Automatically converts string inputs into a Note.
        """
        return self.__eq__(other) or self.__gt__(other)

    def __str__(self) -> str:
        """Return a string representation of the note."""
        return self.note + str(self.octave)

    def _increment_note(self, i: int, add: bool) -> tuple[str, int]:
        """Helper function for adding/subtracting notes. Updates octaves. Returns a note value."""
        new_note_int = self.note_int + i if add else self.note_int - i
        twelves, new_note_int = divmod(new_note_int, 12)  # If note_int >= 12, mod it by 12
        octave = self.octave + twelves  # Add any extra octaves
        return num_to_strnote(new_note_int), octave  # Convert note_int to note name, return octave

    def __add__(self, other: Any):
        """Perform addition on a note by adding intervals to it.

        If a string interval is given, the correct enharmonics will be returned.
        If an integer interval is given, notes default to [C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B]

        >>> str(Note('Ab3') + 'p4')
        'Db4'
        >>> str(Note('Ab3') + 5)
        'C#4'
        """
        if isinstance(other, int):
            note, octave = self._increment_note(other, True)
            return Note(str(note) + str(octave))

        elif isinstance(other, (Interval, str)):
            # Find identity of new note
            other = to_interval(other)
            note, octave = self._increment_note(other.as_int, True)
            x = Note(str(note) + str(octave))

            # Find identity of new letter base
            letters = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
            new_letter_base = letters[(letters.index(self.note[0]) + other.letter_diff - 1) % 7]
            return find_enharmonic_equivalent(new_letter_base, x)
        raise ValueError("You may only add integers/valid strings to a Note")

    def __sub__(self, other: Any):
        """Perform subtraction on a note by subtracting intervals from it.

        If a string interval is given, the correct enharmonics will be returned.
        If an integer interval is given, notes default to [C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B]

        >>> str(Note('C1') - 'p4')
        'G0'
        >>> str(Note('C1') - 5)
        'G0'
        """
        if isinstance(other, int):
            note, octave = self._increment_note(other, False)
            return Note(str(note) + str(octave))

        elif isinstance(other, (Interval, str)):
            # Find identity of new note
            other = to_interval(other)
            note, octave = self._increment_note(other.as_int, False)
            x = Note(str(note) + str(octave))

            # Find enharmonic equivalent
            letters = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
            new_letter_base = letters[(letters.index(self.note[0]) - other.letter_diff + 1) % 7]
            return find_enharmonic_equivalent(new_letter_base, x)
        raise ValueError("You may only add integers/valid strings to a Note")

    def absolute(self) -> int:
        """Return the absolute value of the note."""
        return self.octave * 12 + self.note_int


####################################################################################################
# Interval
####################################################################################################
class Interval:
    """A musical interval.

    Supports the operations:
    - >, >=, <, <=
    - ==, equiv(), enharmonic()
    - +, -

    Instance Attributes:
    - type: the interval type (d, m, M, a)
    - letter_diff: the number of letters to shift
    - as_int: the value of the interval in terms of semitone shifts
    """
    type: str
    letter_diff: int
    as_int: int

    def __init__(self, mus_interval: str) -> None:
        """Initialize the interval.

        Preconditions:
        - mus_interval[0] in {'d', 'D', 'm', 'M', 'a', 'A'}
        - mus_interval[1:].is_numeric()
        """
        self.type = mus_interval[0]
        self.letter_diff = int(mus_interval[1:])
        self.as_int = strinterval_to_int(mus_interval)

    def __str__(self) -> str:
        """Return a string representation of the interval."""
        return self.type + str(self.letter_diff)

    def __eq__(self, other: Any) -> bool:
        """Return whether an interval is equal to another interval."""
        if isinstance(other, (Interval, str)):
            other = to_interval(other)
            return other.type == self.type and other.as_int == self.as_int
        elif isinstance(other, int):
            return self.as_int == other
        raise TypeError('Cannot compare Interval to non-Interval object ' + str(other))

    def equiv(self, other: Any) -> bool:
        """Return whether an interval is exactly equal to another interval."""
        if isinstance(other, (Interval, str)):
            other = to_interval(other)
            return other.type == self.type and other.letter_diff == self.letter_diff
        raise TypeError('Cannot compare Interval to non-Interval object ' + str(other))

    def enharmonic(self, other: Any) -> bool:
        """Return whether two intervals are enharmonic."""
        return self == other and not self.equiv(other)

    def __gt__(self, other: Any) -> bool:
        """Return whether an interval is greater than another interval."""
        if isinstance(other, Interval):
            return self.as_int > to_interval(other).as_int
        elif isinstance(other, int):
            return self.as_int > other
        raise TypeError('Cannot compare Interval to non-Interval object ' + str(other))

    def __lt__(self, other: Any) -> bool:
        """Return whether an interval is less than another interval."""
        if isinstance(other, (Interval, str)):
            return self.as_int < to_interval(other).as_int
        elif isinstance(other, int):
            return self.as_int < other
        raise TypeError('Cannot compare Interval to non-Interval object ' + str(other))

    def __le__(self, other: Any) -> bool:
        """Return whether an interval is less than or equal to another interval."""
        return self == other or self < other

    def __ge__(self, other: Any) -> bool:
        """Return whether an interval is greater than or equal to another interval."""
        return self == other or self > other

    def __add__(self, other: Any):
        """Add this interval to another interval."""
        if isinstance(other, Note):
            return other + self
        if isinstance(other, (Interval, str)):
            return Interval(interval_type(Note('C1'), Note('C1') + self + to_interval(other)))
        raise TypeError('Cannot add an interval with non-interval object ' + str(other))

    def __sub__(self, other: Any):
        """Add this interval to another interval."""
        if isinstance(other, (Interval, str)):
            return Interval(interval_type(Note('C1'), Note('C1') - self - to_interval(other)))
        raise TypeError('Cannot add an interval with non-interval object ' + str(other))


####################################################################################################
# Utility Functions
####################################################################################################
def to_note(n: Union[Note, str]) -> Note:
    """Convert a Note or string to a Note."""
    return n if isinstance(n, Note) else Note(n)


def to_interval(i: Union[Interval, str]) -> Interval:
    """Convert an Interval or string into an Interval."""
    return i if isinstance(i, Interval) else Interval(i)


####################################################################################################
# Note Conversions
####################################################################################################
def strnote_to_num(note: str) -> int:
    """Convert a string note name into a number.

    Preconditions:
    - 1 <= len(note)
    - note[0].isalpha()
    - note[1:] in {'#', 'b', '##', 'bb'}

    >>> strnote_to_num('C#')
    1
    >>> strnote_to_num('Dbb')
    0
    >>> strnote_to_num('Cb')
    11
    """
    return ({'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
            [note[0].upper()] + note[1:].count('#') - note[1:].count('b')) % 12


def num_to_strnote(num: int, scale: Optional[list[str]] = None) -> str:
    """Convert a number into a string note name.

    Preconditions:
    - 1 <= num <= 12

    >>> num_to_strnote(3)
    'Eb'
    >>> num_to_strnote(3, ['C', 'C#', 'D#', 'Gb'])
    'D#'
    >>> num_to_strnote(5, ['C', 'C#', 'D#', 'Gbb'])
    'Gbb'
    >>> num_to_strnote(1, ['Bb', 'Cb', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A'])
    'Db'
    """
    default = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    if scale is not None:
        for i in range(len(default)):
            for y in scale:
                if enharmonic(default[i], y):
                    default[i] = y
    return default[num]


####################################################################################################
# Intervals
####################################################################################################
def interval(n1: Union[Note, str], n2: Union[Note, str]) -> int:
    """Return the interval between two notes (in number of semitones).

    >>> interval('C1', 'C3')
    24
    >>> interval(Note('C#2'), Note('C2'))
    -1
    """
    return to_note(n2).absolute() - to_note(n1).absolute()


def interval_type(n1: Union[Note, str], n2: Union[Note, str]) -> str:
    """Return the interval type between two notes.
    Automatically converts string inputs into a Note.

    >>> interval_type('C1', 'C3')
    'P15'
    >>> interval_type('B1', 'Eb2')
    'd4'
    """
    n1 = to_note(n1)
    n2 = to_note(n2)
    letters = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    letter_diff_to_semitones = {0: 0, 1: 1, 2: 3, 3: 5, 4: 7, 5: 8, 6: 10}

    # d1, P1, A1        -1, 0, 1
    # d2, m2, M2, A2     0, 1, 2, 3
    # d3, m3, M3, A3     2, 3, 4, 5
    # d4, P4, A4         4, 5, 6
    # d5, P5, A5         6, 7, 8
    # d6, m6, M7, A7     7, 8, 9, 10
    # d7, m7, M7, A7     9, 10, 11, 12
    # d8, P8, A8         11, 12, 13

    if n1 == n2:
        return 'P1'
    elif n1 < n2:
        letter_diff = (letters.index(n2.note[0]) - letters.index(n1.note[0])) % 7
        expected = letter_diff_to_semitones[letter_diff]  # Expected semitone of note

        interval_int = interval(n1, n2)
        octaves, actual = divmod(interval_int, 12)  # Actual semitone of note
        if letter_diff == 0 and (interval_int + 1) % 12 == 0:  # Handle diminished 8th
            octaves += 1
            expected += 12

        original = str(letter_diff + 1 + 7 * octaves)

        if expected == actual + 1:
            return 'd' + original
        elif letter_diff in {1, 2, 5, 6}:
            if expected == actual:
                return 'm' + original
            elif expected == actual - 1:
                return 'M' + original
            elif expected == actual - 2:
                return 'A' + original
            else:
                raise ValueError("Interval type between " + str(n1) + " and " + str(
                    n2) + " is not supported. Sorry.")
        elif letter_diff in {0, 3, 4}:
            if expected == actual:
                return 'P' + original
            elif expected == actual - 1:
                return 'A' + original
            else:
                raise ValueError("Interval type between " + str(n1) + " and " + str(
                    n2) + " is not supported. Sorry.")
        else:
            raise ValueError("Something went wrong!")
    else:
        return interval_type(n2, n1)


def strinterval_to_int(mus_interval: str) -> int:
    """Convert a musical interval name into its semitone equivalent.

    Supports only augmented (A), diminished (d), major (M), minor (m), and perfect (P) intervals.

    Preconditions:
    - mus_interval[0] in {'d', 'P', 'p', 'A', 'a', 'M', 'm'}
    - mus_interval[1:].isnumeric()

    >>> strinterval_to_int('P4')
    5
    >>> strinterval_to_int('m9')
    13
    >>> strinterval_to_int('m7')
    10
    """
    new_interval, octaves = break_strinterval_to_octaves(mus_interval)
    semitones = octaves * 12 + {1: 0, 2: 1, 3: 3, 4: 5, 5: 7, 6: 8, 7: 10}[int(new_interval[1:])]

    if mus_interval[0] in {'d', 'D'}:  # Diminished intervals
        semitones -= 1
    elif mus_interval[0] == 'M':  # Major intervals
        semitones += 1
    elif int(new_interval[1:]) in {2, 3, 6, 7} and mus_interval[0] in {'a', 'A'}:  # Aug 2, 3, 6, 7
        semitones += 2
    elif mus_interval[0] in {'A', 'a'}:  # Augmented 1, 4, 5
        semitones += 1

    # d1, P1, A1        -1, 0, 1
    # d2, m2, M2, A2     0, 1, 2, 3
    # d3, m3, M3, A3     2, 3, 4, 5
    # d4, P4, A4         4, 5, 6
    # d5, P5, A5         6, 7, 8
    # d6, m6, M7, A7     7, 8, 9, 10
    # d7, m7, M7, A7     9, 10, 11, 12
    # d8, P8, A8         11, 12, 13

    return semitones


def break_strinterval_to_octaves(n: str) -> tuple[str, int]:
    """Return an interval in it's 1-octave form, plus how many octaves are added.

    >>> break_strinterval_to_octaves('P22')
    ('P1', 3)
    >>> break_strinterval_to_octaves('m7')
    ('m7', 0)
    """
    if isinstance(n, str):
        interval_num = int(n[1:])
        octaves, interval_num = divmod(interval_num, 7)

        if interval_num == 0:
            if octaves == 0:
                raise ValueError('Interval ' + n + ' was not properly inputted.')
            interval_num += 7
            octaves -= 1

        return n[0] + str(interval_num), octaves


def absolute_letter_diff(n1: Union[str, Note], n2: Union[str, Note]) -> int:
    """Return the letter note difference between two notes, octave-sensitive."""
    n1, n2 = to_note(n1), to_note(n2)
    if n1 > n2:
        return -absolute_letter_diff(n2, n1)

    # Accounting for notational differences in notes like B#3, A###3, etc.
    c1 = 'C' + str(n1.octave)
    c2 = 'C' + str(n2.octave)
    if n1.enharmonic(c1):
        if "#" in n1.note:
            n1 = Note(n1.note[0] + str(n1.octave - 1))
        elif "b" in n1.note:
            n1 = Note(n1.note[0] + str(n1.octave + 1))
    elif n2.enharmonic(c2):
        if '#' in n1.note:
            n2 = Note(n2.note[0] + str(n2.octave - 1))
        elif 'b' in n1.note:
            n2 = Note(n2.note[0] + str(n2.octave + 1))
    # Find letter difference
    letters = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    letter_diff = letters.index(n2.note[0]) - letters.index(n1.note[0])
    # Find octave difference
    octaves = n2.octave - n1.octave

    return octaves * 7 + letter_diff


####################################################################################################
# Enharmonics
####################################################################################################
def enharmonic(n1: Union[Note, str], n2: Union[Note, str]) -> bool:
    """Return whether two notes are enharmonic."""
    if isinstance(n1, str) and isinstance(n2, str):
        if not any(x.isnumeric() for x in n1) and not any(x.isnumeric() for x in n2):
            return Note(n1 + '1').enharmonic(n2 + '1')
        if any(x.isnumeric() for x in n1) and any(x.isnumeric() for x in n2):
            return Note(n1).enharmonic(n2)
        raise TypeError("Cannot a note with octaves to a note without octaves.")
    if isinstance(n1, Note) and isinstance(n2, Note):
        return n1.enharmonic(n2)
    raise TypeError("Can only compare two strings or two notes.")


def find_enharmonic_equivalent(base: str, n: Union[str, Note]) -> Note:
    """Find an equivalent note to n with given base letter (C, D, E, F, G, A, B).
    Automatically converts string inputs into a Note.

    Preconditions:
    - base in {'C', 'D', 'E', 'F', 'G', 'A', 'B'}

    >>> str(find_enharmonic_equivalent('C', Note('Db1')))
    'C#1'
    >>> str(find_enharmonic_equivalent('E', Note('F2')))
    'E#2'
    >>> str(find_enharmonic_equivalent('F', Note('G3')))
    'F##3'
    """
    n = to_note(n)
    letters = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    n_index = letters.index(n.note[0])  # comparing note's index
    base_index = letters.index(base)    # base letter's index
    if n_index == base_index:
        return n

    if -3 <= base_index - n_index <= -1 or 4 <= base_index - n_index <= 6:
        while not enharmonic(base, n.note):
            base += '#'
    else:
        while not enharmonic(base, n.note):
            base += 'b'
    return Note(base + str(n.octave))
