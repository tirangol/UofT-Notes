"""Cantus Firmus

Classes:
- Scale

Functions:
- note_range(), scale_range()
- absolute_letter_diff()
- accidental_list()
"""

from note import *


class Scale:
    """A musical scale.

    Note that notes in Scale have no octave markings.

    Instance Attributes:
    - as_str: a list of every note in the scale as a string.
    - as_int: a list of every note in the scale as an integer.
    - chromatic_notation: a list of every chromatic note in its correct enharmonic position
    - key: the scale's key
    - scale_type: the scale's type
    - sharp_flat_count: the number of sharps (positive) or flats (negative).
    - inversion: how many inversions have been made
    """
    as_str = list[str]
    as_int = list[int]
    chromatic_notation = list[str]
    key: Optional[str]
    scale_type: Optional[str]
    sharp_flat_count: int
    inversion: int

    def __init__(self, key: Optional[str] = None, scale_type: Optional[str] = None,
                 notes: Optional[list[str]] = None):
        """Initialize the scale, either through:
        - Giving a scale type and key
        - Giving a list of notes (and optional key)
        """
        if key is None and scale_type is None and notes is None:
            raise ValueError("No arguments detected.")
        if scale_type is not None and key is None:
            raise ValueError("A scale type must be entered with a key.")
        if (scale_type is None) == (notes is None):
            raise ValueError("You must either enter a custom scale or a scale type.")

        self.scale_type = None if scale_type is None else _simplify(scale_type)
        self.inversion = 0
        if key is None:
            self.key = None
            self.as_str, self.as_int, self.chromatic_notation = [], [], []
            self.sharp_flat_count = 0
        else:
            self.key = key[0].upper() + key[1:]
            self.as_str = [self.key]
            self.as_int = [strnote_to_num(self.key)]
            self.chromatic_notation = self.get_chromatic_notation()
            self.sharp_flat_count = self.get_sharp_flat_count()

        if scale_type is None:  # Custom scale
            self.as_str = notes
            self.as_int = [strnote_to_num(x) for x in notes]
        else:  # Default scale
            as_note = [Note(self.key + '1')]
            for str_interval in get_scale_type(self.scale_type)[:-1]:
                as_note.append(as_note[-1] + str_interval)
                self.as_str.append(as_note[-1].note)
                self.as_int.append(as_note[-1].note_int)

    def get_chromatic_notation(self) -> list[str]:
        """Get the chromatic notation of the scale."""
        if self.key is None:
            return []
        note_scale = [Note(self.key + '1')]
        chromatic_notation = [self.key]
        for str_interval in ['m2', 'A1', 'm2', 'A1', 'm2', 'A1', 'm2', 'm2', 'A1', "m2", 'A1']:
            note_scale.append(note_scale[-1] + str_interval)
            chromatic_notation.append(note_scale[-1].note)
        return chromatic_notation

    def get_sharp_flat_count(self) -> int:
        """Get the number of sharps/flats in the scale.
        Sharps are 1, flats are -1. Double/triple sharps/flats are 2/-2 and 3/-3.
        """
        majors = {'major', 'ionian', 'lydian', 'mixolydian', 'pentatonic', 'major pentatonic',
                  'double harmonic major'}
        neutrals = {'whole tone', 'octatonic'}
        if (self.key == 'C' and self.scale_type in majors) or (
                self.key == 'A' and self.scale_type not in neutrals and self.scale_type not
                in majors) or self.scale_type in neutrals or self.scale_type is None:
            return 0
        else:
            circle_of_fifths = ['Fbb', 'Cbb', 'Gbb', 'Dbb', 'Abb', 'Ebb', 'Bbb', 'Fb', 'Cb', 'Gb',
                                'Db', 'Ab', 'Eb', 'Bb', 'F', 'C', 'G', 'D', 'A', 'E', 'B', 'F#',
                                'C#', 'G#', 'D#', 'A#', 'E#', 'B#', 'F##', 'C##', 'G##', 'D##',
                                'A##', 'E##', 'B##']
            for i in range(3, len(circle_of_fifths)):
                if (self.key == circle_of_fifths[i - 3] and self.scale_type in majors) or \
                        (self.key == circle_of_fifths[i] and self.scale_type not in neutrals and
                         self.scale_type not in majors):
                    return i - 18
            raise ValueError('Key not supported.')

    def __str__(self) -> str:
        """Return a string representation of the scale."""
        text = 'scale of length ' + str(len(self.as_int))
        if self.scale_type is not None:
            text = str(self.scale_type) + ' ' + text
        if self.key is not None:
            text = str(self.key) + ' ' + text
        if self.inversion != 0:
            text += ', ' + str(self.inversion) + 'th inversion'
        return text

    def __contains__(self, item: Any) -> bool:
        """Return whether item is a note inside the scale.
        Automatically converts string and integer inputs into a Note.
        """
        if isinstance(item, Note):
            return item.note in self.as_str
        elif isinstance(item, str):
            return item in self.as_str
        elif isinstance(item, int):
            return item in self.as_int
        else:
            raise TypeError('Invalid input type ' + str(item))

    def __eq__(self, other: Any) -> bool:
        """Return whether two scales contain the same notes."""
        if isinstance(other, Scale):
            return set(self.as_int) == set(other.as_int)
        elif isinstance(other, list):
            if all(isinstance(x, str) for x in other):
                return set(self.as_str) == set(other)
            elif all(isinstance(x, int) for x in other):
                return set(self.as_int) == set(other)
            raise TypeError('Cannot compare Scale to a list of multiple object types')
        raise TypeError('Cannot compare Scale to a non-Scale object ' + str(other))

    def invert(self, inv: Optional[int] = None) -> None:
        """Mutate the scale to an inversion."""
        if inv is None:
            self.inversion = (self.inversion + 1) % len(self.as_int)
            if self.scale_type is None:
                self.as_str = self.as_str[1:] + self.as_str[:1]
                self.as_int = self.as_int[1:] + self.as_int[:1]
                return
        elif 0 <= inv < len(self.as_int):
            temp = self.inversion
            self.inversion = inv
            if self.scale_type is None:
                if inv >= temp:
                    new_ind = inv - temp
                else:
                    new_ind = len(self.as_str) - temp + inv

                self.as_str = self.as_str[new_ind:] + self.as_str[:new_ind]
                self.as_int = self.as_int[new_ind:] + self.as_int[:new_ind]
                return
        else:
            raise ValueError('Entered inversion ' + str(inv) + ' is invalid')

        intervals = get_scale_type(self.scale_type)
        intervals = intervals[self.inversion:] + intervals[:self.inversion]
        as_note = [Note(self.key + '1')]
        for i in range(len(intervals)):
            as_note.append(as_note[-1] + intervals[i])
            self.as_str[i] = as_note[i].note
            self.as_int[i] = as_note[i].note_int


####################################################################################################
# Scales
####################################################################################################
def get_scale_type(scale_type: str) -> list[str]:
    """Get a list of intervals corresponding to the scale type."""
    if scale_type in {'major', 'ionian'}:
        return ['M2', 'M2', 'm2', 'M2', 'M2', 'M2', 'm2']
    elif scale_type in {'minor', 'aeolian', 'natural minor'}:
        return ['M2', 'm2', 'M2', 'M2', 'm2', 'M2', 'M2']
    elif scale_type in {'melodic minor', 'minor melodic'}:
        return ['M2', 'm2', 'M2', 'M2', 'M2', 'M2', 'm2']
    elif scale_type in {'harmonic minor', 'minor harmonic'}:
        return ['M2', 'm2', 'M2', 'M2', 'm2', 'A2', 'm2']
    elif scale_type == 'lydian':
        return ['M2', 'M2', 'M2', 'm2', 'M2', 'M2', 'm2']
    elif scale_type == 'mixolydian':
        return ['M2', 'M2', 'm2', 'M2', 'M2', 'm2', 'M2']
    elif scale_type == 'dorian':
        return ['M2', 'm2', 'M2', 'M2', 'M2', 'm2', 'M2']
    elif scale_type == 'phrygian':
        return ['m2', 'M2', 'M2', 'M2', 'm2', 'M2', 'M2']
    elif scale_type == 'locrian':
        return ['m2', 'M2', 'M2', 'm2', 'M2', 'M2', 'M2']
    elif scale_type == 'whole tone':
        return ['M2', 'M2', 'M2', 'M2', 'M2', 'd3']
    elif scale_type in {'blues', 'blue'}:
        return ['m3', 'M2', 'A1', 'm2', 'm3', 'M2']
    elif scale_type in {'major pentatonic', 'pentatonic'}:
        return ['M2', 'M2', 'm3', 'M2', 'm3']
    elif scale_type in {'minor pentatonic', 'minyo'}:
        return ['m3', 'M2', 'M2', 'm3', 'M2']
    elif scale_type in {'altered phrygian', 'freygish', 'fraygish', 'harmonic dominant',
                        'phrygian major', 'spanish phrygian', 'spanish gypsy', 'andalusian'}:
        return ['m2', 'A2', 'm2', 'M2', 'm2', 'M2', 'M2']
    elif scale_type in {'altered dorian', 'ukrainian dorian', 'romanian dorian', 'misheberak',
                        'mi sheberach', 'av horachamon', 'nikriz', 'aulos'}:
        return ['M2', 'm2', 'A2', 'm2', 'M2', 'm2', 'M2']
    elif scale_type in {'double harmonic minor', 'hungarian minor', 'gypsy minor'}:
        return ['M2', 'm2', 'A2', 'm2', 'm2', 'A2', 'm2']
    elif scale_type in {'double harmonic major', 'arabic', 'gypsy major', 'mayamalavagowla',
                        'bhairav raga', 'byzantine'}:
        return ['m2', 'A2', 'm2', 'M2', 'm2', 'A2', 'm2']
    elif scale_type == 'japanese':
        return ['M2', 'm2', 'M3', 'm2', 'M3']
    elif scale_type == 'in':
        return ['m2', 'M3', 'M2', 'm2', 'M3']
    elif scale_type in {'insen', 'revati'}:
        return ['m2', 'M3', 'M2', 'm3', 'M2']
    elif scale_type == 'iwato':
        return ['m2', 'M3', 'm2', 'M3', 'M2']
    elif scale_type == 'octatonic':
        return ['M2', 'm2', 'M2', 'A1', 'M2', 'm2', 'M2', 'm2']
    elif scale_type in {'altered dominant', 'super locrian'}:
        return ['m2', 'M2', 'm2', 'M2', 'M2', 'M2', 'M2']
    else:
        raise ValueError('Scale type ' + scale_type + ' not found.')


####################################################################################################
# Utility Functions
####################################################################################################
def _simplify(txt: str) -> str:
    """Format text properly."""
    while "  " in txt:
        txt = txt.replace('  ', ' ')
    return txt.replace('-', ' ').replace('_', ' ').lower()


def note_range(n1: Union[Note, str], n2: Union[Note, str], increment: int = 1,
               scale: Optional[Scale] = None) -> tuple:
    """Return every note between n1 and n2 with an adjustable increment.
    Adding a scale will only filter results to notes on that scale.
    Automatically converts string inputs into a Note.

    >>> [str(y) for y in note_range('Gb1', 'E2')]
    ['F#1', 'G1', 'Ab1', 'A1', 'Bb1', 'B1', 'C2', 'C#2', 'D2', 'Eb2']
    >>> s = Scale('Gb', 'major')
    >>> [str(y) for y in note_range('Gb1', 'E2', scale=s)]
    ['Gb1', 'Ab1', 'Bb1', 'Cb1', 'Db2', 'Eb2']
    >>> [str(y) for y in note_range('C#3', 'F#2', -2)]
    ['C#3', 'B2', 'A2', 'G2']
    >>> s = Scale('c', 'minor')
    >>> [str(y) for y in note_range('C1', 'E2', scale=s)]
    ['C1', 'D1', 'Eb1', 'F1', 'G1', 'Ab1', 'Bb1', 'C2', 'D2', 'Eb2']
    >>> [str(y) for y in note_range('E5', 'D4', -3, scale=s)]
    ['Bb4', 'G4']
    """
    if increment == 0:
        raise ValueError('Invalid increment 0')

    lst_so_far = []
    n1, n2 = to_note(n1), to_note(n2)
    if scale is None:
        for i in range(0, interval(n1, n2), increment):
            lst_so_far.append(n1 + i)
    else:
        for i in range(0, interval(n1, n2), increment):
            for x in scale.as_str:
                if n1 + i == Note(x + str((n1 + i).octave)):
                    lst_so_far.append(find_enharmonic_equivalent(x[0], n1 + i))
                    break
    return tuple(lst_so_far)


def scale_range(n1: Union[Note, str], n2: Union[Note, str], scale: Scale,
                increment: int = 1) -> tuple:
    """Return every note between n1 and n2 on a scale, with an adjustable increment.
    The increment will skip through scale notes.
    Automatically converts string inputs into a Note.
    """
    full_scale = note_range(n1, n2, 1, scale)
    return tuple([full_scale[i] for i in range(len(full_scale)) if i % increment == 0])


def accidental_list(notes: list[Note], scale: Scale,
                    return_acc: bool = True) -> list[Union[bool, str]]:
    """Return the accidental that should be in front of every item in the list.

    >>> accidental_list([Note('D4'), Note('E4'), Note('F4'), Note('G4'), Note('E4'), Note('C4'),\
    Note('D4')], Scale('C#', 'major'))
    [True, True, True, True, False, True, False]
    """
    as_acc = []
    for i in range(len(notes)):
        accidental = 'n' if len(notes[i].note) == 1 else notes[i].note[1:]
        same_note_detected = False

        for j in range(i, -1, -1):
            # Previous note of same note-head, same accidental
            if notes[j].equiv(notes[i]) and notes[i] is not notes[j]:
                as_acc.append(False if return_acc else '')
                same_note_detected = True
                break
            # Previous note of same note-head, different accidental
            elif notes[j].same_letter_head(notes[i]) and notes[j].octave == notes[i].octave \
                    and notes[i] is not notes[j]:
                as_acc.append(True if return_acc else accidental)
                same_note_detected = True
                break
        # No previous note of same value in voice
        if not same_note_detected:
            if notes[i].note in scale:
                as_acc.append(False if return_acc else '')
            else:
                as_acc.append(True if return_acc else accidental)
    return as_acc
