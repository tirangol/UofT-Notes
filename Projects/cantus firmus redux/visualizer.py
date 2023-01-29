"""Cantus Firmus

Classes:
- Visualizer
"""
from player import *

BG_COL = 233, 233, 233
DARK = 50, 50, 50
BLUE = 77, 120, 200
RED = 200, 120, 77
YELLOW = 120, 200, 77
DARK_RED = 150, 30, 30
DARK_YELLOW = 30, 150, 30


class Visualizer:
    """A visualizer of a voice.

    Instance Attributes:
    - voice: the voice to visualize
    - screen: the pygame screen
    - scaling: the scaling factor of the graphics on the screen
    - screen_size: the size of the pygame screen
    - player: the player of the voice
    """
    voice: Voice
    _screen: Optional[pygame.Surface]
    scaling: int
    screen_size: tuple[int, int]
    _player: Optional[Player]

    def __init__(self, voice: Voice, scaling: int = 15,
                 screen_size: tuple[int, int] = (1000, 600)) -> None:
        """Initialize the visualizer.
        >>> v = Voice(['Bb3', 'Bb4', 'B4', 'Eb3', 'Eb3', 'A4', 'Bb4', 'B4', 'F#4', 'C#4', 'A#4'])
        >>> vis = Visualizer(v, 30)
        >>> vis.display(0)
        >>> vis = Visualizer(v, 15)
        >>> vis.display()
        >>> vis = Visualizer(v, 10)
        >>> vis.display()
        """
        if screen_size[0] <= 200:
            raise ValueError("Screen size is too skinny. Make it at least 200.")
        self.voice = voice
        self._player = Player(self.voice)
        self._player.create_midi()
        self.scaling = scaling
        self.screen_size = screen_size

    def display(self, x_margin: int = 50) -> None:
        """Visualize the voice in pygame."""
        pygame.display.init()
        pygame.font.init()
        self._screen = pygame.display.set_mode(self.screen_size)
        pygame.event.clear()
        pygame.event.set_blocked(None)
        pygame.event.set_allowed([pygame.QUIT, pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN])

        music_playing = False
        meaningless_key = False
        toggle_annotations = True
        scroll = 0

        while True:
            if not music_playing and not meaningless_key:
                self.draw_frame(x_margin, scroll, toggle_annotations)
                if toggle_annotations:
                    self.annotate(x_margin)
                pygame.display.flip()
            music_playing = False
            meaningless_key = False

            event = pygame.event.wait()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 4:  # Scroll Up
                scroll += 50
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 5:  # Scroll Down
                scroll -= 50
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:  # Play music
                music_playing = True
                self._player.play_midi()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:  # Annotations
                toggle_annotations = not toggle_annotations
            elif event.type == pygame.QUIT:
                break
            else:
                meaningless_key = True
        pygame.display.quit()

####################################################################################################
# Master Functions
####################################################################################################
    def draw_frame(self, x_margin: int, scroll: int, annotations: bool = False) -> None:
        """Draw everything note-related."""
        # Calculate x-offsets
        key_sig_offset = 0 if self.voice.scale is None else min(
            abs(self.voice.scale.sharp_flat_count), 7)
        note_offset = round(key_sig_offset * self.scaling + self.scaling * 3.5)
        clef_offset = round(3.5 * self.scaling)

        # Calculate starting y
        y = 100 + scroll

        # Background
        self._screen.fill(BG_COL)

        # Staff, clef, key signature
        self.draw_staff(x_margin, y)
        clef = self.draw_clef(x_margin, y)
        self.draw_key_signature(clef, clef_offset, y, x_margin)

        # Notes
        as_note = self.voice.as_note.copy()
        scale_to_use = Scale('c', 'major') if self.voice.scale is None else self.voice.scale
        as_acc = accidental_list(self.voice.as_note, scale_to_use)

        i = 0
        as_note, as_acc, i = self.draw_notes(clef, x_margin + clef_offset + note_offset, y,
                                             as_note, as_acc, x_margin, annotations, i)

        while len(as_note) >= 1:
            y += self.scaling * 12

            # Staff, clef, key signature
            self.draw_staff(x_margin, y)
            clef = self.draw_clef(x_margin, y)
            self.draw_key_signature(clef, clef_offset, y, x_margin)

            # Notes
            as_note, as_acc, i = self.draw_notes(clef, x_margin + clef_offset + note_offset, y,
                                                 as_note, as_acc, x_margin, annotations, i)

    def annotate(self, x_margin: int) -> None:
        """Draw the text-related segment at the bottom."""
        c1 = self.voice.analysis['melodic range']
        c2 = self.voice.analysis['sufficient melodic range']
        c3 = self.voice.analysis['standard deviation of notes']
        c4 = self.voice.analysis['sufficient standard deviation of notes']
        c5 = self.voice.analysis['occurrences of lowest note']
        c6 = 'Soprano, ' if self.voice.analysis['soprano range'] else ''
        c7 = 'Alto, ' if self.voice.analysis['alto range'] else ''
        c8 = 'Tenor, ' if self.voice.analysis['tenor range'] else ''
        c9 = 'Bass, ' if self.voice.analysis['bass range'] else ''
        c10 = self.voice.analysis['repeated notes']
        c11 = self.voice.analysis['predominantly conjunct intervals']

        line = ["Melody Statistics",
                ("Melodic Range:", f"{c1} semitone{'s' if c1 > 1 else ''} {'' if c2 else '(low)'}"),
                ("Note Spread:", f"{c3} semitone{'s' if c3 > 1 else ''} {'' if c4 else '(low)'}"),
                ("Lowest Note Appears:",
                 f"{c5} {'time' if c5 == 1 else 'times'} {'' if c5 == 1 else '(should be 1)'}"),
                ("Singable for:", (c6 + c7 + c8 + c9)[:-2] if any(
                    x != '' for x in {c6, c7, c8, c9}) else 'No registers'),
                ("Consecutive Notes:", f"{c10} {'' if c10 <= 1 else '(should be at most 1)'}"),
                ("Skips vs. Steps:", f"{'Predominantly steps' if c11 else 'Predominantly skips'}")]
        red = {'low', 'should be', 'registers', 'skips'}

        if self.screen_size[0] > 600:  # Wide screens (width > 600)
            self._draw_rect((x_margin - 10, self.screen_size[1] - 135), 760, 110)
            self._draw_text(line[0], x_margin, self.screen_size[1] - 130, True, False, 24)

            for i in range(1, len(line) // 2 + 1):
                for j in {0, 3}:
                    self._draw_text(line[i + j][0], x_margin + (0 if j == 0 else 400),
                                    self.screen_size[1] - 120 + 20 * i, False, True, 18)
                    c = BLUE if not any(x in line[i + j][1] for x in red) else RED
                    self._draw_text(line[i + j][1], x_margin + (190 if j == 0 else 570),
                                    self.screen_size[1] - 120 + 20 * i, False, False, 18, c)
        else:  # Thin screens (width <= 600)
            self._draw_rect((x_margin - 10, self.screen_size[1] - 195), 400, 165)
            self._draw_text(line[0], x_margin, self.screen_size[1] - 190, True, False, 24)

            for i in range(1, len(line)):
                self._draw_text(line[i][0], x_margin,
                                self.screen_size[1] - 180 + 20 * i, False, True, 18)
                c = BLUE if not any(x in line[i][1] for x in red) else RED
                self._draw_text(line[i][1], x_margin + 190,
                                self.screen_size[1] - 180 + 20 * i, False, False, 18, c)

####################################################################################################
# Fundamentals
####################################################################################################
    def draw_staff(self, x_margin: int, y0: int) -> None:
        """Draw the staff on pygame."""
        # 5 horizontal lines
        # Height of a single note is self.scaling
        for i in range(5):
            y = y0 + i * self.scaling
            y_end = y - math.ceil(self.scaling / 10) / 2
            self._draw_line((x_margin, y_end), (self.screen_size[0] - x_margin, y_end))

        y0end = y0 - math.ceil(self.scaling / 10) / 2

        # Left vertical line
        self._draw_line((x_margin, y0end), (x_margin, y0end + 4 * self.scaling))

        # Right vertical line
        self._draw_line((self.screen_size[0] - x_margin, y0end),
                        (self.screen_size[0] - x_margin, y0end + 4 * self.scaling))

    def draw_clef(self, x_margin: int, y: int) -> str:
        """Draw the clef on pygame and return the clef type."""
        # Check number of voices above/below middle C
        c1 = sum(x > 'C4' for x in self.voice.as_note) >= sum(x < 'C4' for x in self.voice.as_note)
        # Find the sum of intervals of all notes with middle C
        c2 = sum(interval('C4', x) for x in self.voice.as_note)

        if c1 or c2 >= 0:
            y = y - self.scaling * 1.5 - math.ceil(self.scaling / 10) / 1.5
            clef = load_image(r'images\treble.png', self.scaling * 7.35)
            self._draw_image(clef, (x_margin + self.scaling * 0.75, y))
            return 'treble'
        else:
            y = y - self.scaling * 0.1 - math.ceil(self.scaling / 10) / 1.5
            clef = load_image(r'images\bass.png', self.scaling * 3.65)
            self._draw_image(clef, (x_margin + self.scaling * 0.75, y))
            return 'bass'

    def draw_key_signature(self, clef: str, x0: int, y0: int, x_margin: int) -> None:
        """Draw the key signature on pygame."""
        if self.voice.scale is None or self.voice.scale.sharp_flat_count == 0:
            return
        if abs(self.voice.scale.sharp_flat_count) > 7:
            raise ValueError("Sorry, complex key signatures are not supported.")

        y0 = y0 if clef == 'treble' else y0 + self.scaling

        if self.voice.scale.sharp_flat_count > 0:
            sharp_locations = {1: (x0,                    round(y0 + 0.4 * self.scaling)),
                               2: (x0 + self.scaling,     round(y0 + 1.9 * self.scaling)),
                               3: (x0 + 2 * self.scaling, round(y0 - 0.1 * self.scaling)),
                               4: (x0 + 3 * self.scaling, round(y0 + 1.4 * self.scaling)),
                               5: (x0 + 4 * self.scaling, round(y0 + 2.9 * self.scaling)),
                               6: (x0 + 5 * self.scaling, round(y0 + 0.9 * self.scaling)),
                               7: (x0 + 6 * self.scaling, round(y0 + 2.4 * self.scaling))}
            for i in range(1, self.voice.scale.sharp_flat_count + 1):
                self.draw_sharp(sharp_locations[i], x_margin=x_margin)
        else:
            flat_locations = {1: (x0,                    round(y0 + 2 * self.scaling)),
                              2: (x0 + self.scaling,     round(y0 + 0.4 * self.scaling)),
                              3: (x0 + 2 * self.scaling, round(y0 + 2.4 * self.scaling)),
                              4: (x0 + 3 * self.scaling, round(y0 + 0.9 * self.scaling)),
                              5: (x0 + 4 * self.scaling, round(y0 + 2.9 * self.scaling)),
                              6: (x0 + 5 * self.scaling, round(y0 + 1.4 * self.scaling)),
                              7: (x0 + 6 * self.scaling, round(y0 + 3.3 * self.scaling))}
            for i in range(1, -self.voice.scale.sharp_flat_count + 1):
                self.draw_flat(flat_locations[i], x_margin=x_margin)

####################################################################################################
# Accidentals
####################################################################################################
    def draw_sharp(self, coordinate: tuple[int, int],
                   sharp_type: Optional[str] = None, x_margin: int = 50) -> None:
        """Draw a sharp on pygame."""
        if sharp_type is None:
            sharp = load_image(r'images\sharp.png', self.scaling * 2.3)
        elif sharp_type == 'double':
            sharp = load_image(r'images\double_sharp.png', self.scaling)
        elif sharp_type == 'triple':
            sharp = load_image(r'images\triple_sharp.png', self.scaling * 2.3)
        else:
            raise ValueError('Invalid sharp type ' + sharp_type)
        x, y = coordinate
        x = x_margin + self.scaling * 0.75 + x
        y = y - self.scaling * 1.5 - math.ceil(self.scaling / 10) / 1.5
        self._draw_image(sharp, (x, y))

    def draw_flat(self, coordinate: tuple[int, int],
                  flat_type: Optional[str] = None, x_margin: int = 50) -> None:
        """Draw a flat on pygame."""
        if flat_type is None:
            flat = load_image(r'images\flat.png', self.scaling * 2.3)
        elif flat_type == 'double':
            flat = load_image(r'images\double_flat.png', self.scaling * 2.3)
        elif flat_type == 'triple':
            flat = load_image(r'images\triple_flat.png', self.scaling * 2.3)
        else:
            raise ValueError('Invalid flat type ' + flat_type)
        x, y = coordinate
        x = x_margin + self.scaling * 0.75 + x
        y = y - self.scaling * 1.5 - math.ceil(self.scaling / 10) / 1.5
        self._draw_image(flat, (x, y))

    def draw_natural(self, coordinate: tuple[int, int], x_margin: int = 50) -> None:
        """Draw a natural on pygame."""
        natural = load_image(r'images\natural.png', self.scaling * 2.3)
        x, y = coordinate
        x = x_margin + self.scaling * 0.75 + x
        y = y - self.scaling * 1.5 - math.ceil(self.scaling / 10) / 1.5
        self._draw_image(natural, (x, y))

####################################################################################################
# Notes
####################################################################################################
    def draw_notes(self, clef: str, x0: int, y: int, notes: list[Note], as_acc: list[bool],
                   x_margin: int, annotations: bool,
                   offset: int = 0) -> tuple[list[Note], list[bool], int]:
        """Draw a series of notes on pygame."""
        i = 0
        # While the current note's x isn't too far right and there're still notes left
        while (x0 + self.scaling * 5 * i + 3 * self.scaling) <= (self.screen_size[0] - x_margin) \
                and i < len(notes):
            # Draw annotations if they're turned on
            if annotations:
                self.draw_note_annotations(i, x0 + self.scaling * 5 * i, y, offset)
            # Draw notes
            self.draw_note(clef, str(notes[i]), x0 + self.scaling * 5 * i, y, as_acc[i])
            i += 1

        # Return the used up portions of the lists
        return notes[i:], as_acc[i:], i + offset

    def draw_note(self, clef: str, note_name: str, x: int, y0: int,
                  accidental: bool = True) -> None:
        """Draw a single note on pygame."""
        # Load whole note image
        note = load_image(r'images\whole_note.png', self.scaling * 1.05)

        # Calculate y position of note
        compare_note = 'E3' if clef == 'bass' else 'C5'
        interval_name_int = absolute_letter_diff(note_name, compare_note)
        y = y0 + self.scaling + (self.scaling * interval_name_int // 2)

        # Display ledger lines
        if interval_name_int <= -5:
            to_ledger_line = ((-interval_name_int + 1) // 2) - 2
            self._draw_ledger_line(to_ledger_line, x, y, interval_name_int, True)
        elif interval_name_int >= 7:
            to_ledger_line = ((interval_name_int + 1) // 2) - 3
            self._draw_ledger_line(to_ledger_line, x, y, interval_name_int, False)

        # Display note
        self._screen.blit(note, (x, y))

        # Display accidental
        if accidental:
            self._draw_note_accidental(Note(note_name).note, x, y)

####################################################################################################
# Note annotations
####################################################################################################
    def draw_note_annotations(self, i: int, x: int, y: int, offset: int) -> None:
        """Draw note-specific annotations."""
        # Above the staff, how deep the shades of red/yellow for a rectangle
        reds = 0
        yellows = 0
        # Which rules are being violated
        reds, leap = self._draw_note_annotations('leap interval rule violations',
                                                 i, 3, offset, True, reds)
        reds, p_int = self._draw_note_annotations('prohibited intervals', i, 2, offset, False, reds)
        yellows, m6 = self._draw_note_annotations('M6s', i, 2, offset, False, yellows)
        reds, desc = self._draw_note_annotations('descending P5 + m3s', i, 3, offset, True, reds)

        # Draw rectangles of different colours
        if reds != 0:
            self._draw_warning_bar(x, y, RED, reds, True)
        if yellows != 0:
            self._draw_warning_bar(x, y, YELLOW, yellows + 1, True)
        # Draw text; if multiple, then some text needs to go a line above the other
        msg_so_far = 0
        msg_dict = {0: 'LEAP', 1: 'INT.', 2: 'M6', 3: 'm7'}
        stats = [leap, p_int, m6, desc]

        for j in range(len(stats)):
            if stats[j]:
                if j != 2:
                    self._draw_warning_text(x, y - msg_so_far * self.scaling, msg_dict[j],
                                            DARK_RED, True)
                else:
                    self._draw_warning_text(x, y - msg_so_far * self.scaling, msg_dict[j],
                                            DARK_YELLOW, True)
                msg_so_far += 1

        # Below the staff, how deep the shades of red/yellow for a rectangle
        r = 0
        text = 'prohibited one-directional start/end intervals'
        r, start_end = self._draw_note_annotations(text, i, None, offset, False, r)
        skip = False
        for v in self.voice.analysis['consecutive one-directional disjunct intervals']:
            if v[0] > 2 and i in range(v[-1], v[-1] + v[0]):
                skip = True
                r += 1
                break
        text = 'consecutive m2s (chromatic progressions)'
        r, ch = self._draw_note_annotations(text, i, None, offset, True, r)
        text = 'consecutive P4s'
        r, q = self._draw_note_annotations(text, i, None, offset, True, r)
        # Draw rectangles of different colours
        if r != 0:
            self._draw_warning_bar(x, y, RED, r, False)
        # Draw text; if multiple, then some text needs to go a line above the other
        msg_so_far = 0
        msg_dict = {0: 'SEQ.', 1: 'SKIP', 2: 'm2', 3: 'P4'}
        stats = [start_end, skip, ch, q]
        for i in range(len(stats)):
            if stats[i]:
                self._draw_warning_text(x, y + msg_so_far * self.scaling,
                                        msg_dict[i], DARK_RED, False)
                msg_so_far += 1

####################################################################################################
# Helper Functions
####################################################################################################
    def _draw_note_annotations(self, rule: str, note_i: int, i: Optional[int], offset: int,
                               early_break: bool, colour_depth: int) -> tuple[int, bool]:
        """Return a new colour depth and if a rule is violated."""
        adjust_to_v = True if i is None else False

        if early_break:
            for v in self.voice.analysis[rule]:
                i = len(v[0]) if adjust_to_v else i
                if note_i in range(v[-1] - offset, v[-1] - offset + i):
                    return colour_depth + 1, note_i == v[-1] - offset
            return colour_depth, False
        else:
            rule_violated = False
            for v in self.voice.analysis[rule]:
                i = len(v[0]) if adjust_to_v else i
                if note_i in range(v[-1] - offset, v[-1] - offset + i):
                    colour_depth += 1
                    rule_violated = note_i == v[-1] - offset
            return colour_depth, rule_violated

    def _draw_warning_bar(self, x: int, y: int, colour: tuple[int, int, int],
                          alpha: int, up: bool) -> None:
        """Draw a warning bar in pygame."""
        width = self.scaling * 5
        height = round(self.scaling * 0.3)
        x = x - round(2.5 * self.scaling)
        y = round(y - 0.58 * self.scaling) if up else round(y + 4.5 * self.scaling)
        self._draw_rect((x, y), width, height, colour, alpha * 70)

    def _draw_warning_rect(self, x: int, y: int, alpha: int) -> None:
        """Draw a transparent red rectangle in pygame."""
        width = self.scaling * 5
        height = self.scaling * 4
        x = x - round(2.5 * self.scaling)
        self._draw_rect((x, y), width, height, RED, 70 * alpha)

    def _draw_warning_text(self, x: int, y: int, message: str, colour: tuple[int, int, int],
                           up: bool) -> None:
        """Draw warning text in pygame."""
        # Draw text
        x = x - round(2.4 * self.scaling)
        y = y - round(1.08 * self.scaling) if up else y + round(4.3 * self.scaling)
        text_size = round(self.scaling * 0.8)
        self._draw_text(message, x, y, True, False, text_size, colour)

    def _draw_note_accidental(self, note_name: str, x: int, y: int) -> None:
        """Draw the accidental to a note."""
        accidental = note_name[1:]
        y = y + self.scaling
        x = x - 50
        single_offset = round(1.8 * self.scaling)
        double_offset = round(2.3 * self.scaling)
        triple_offset = round(2.9 * self.scaling)

        sharp_x_offset = round(0.2 * self.scaling)
        flat_y_offset = round(0.55 * self.scaling)
        double_sharp_y_offset = round(0.58 * self.scaling)

        if accidental in {'n', ''}:
            self.draw_natural((x - single_offset, y))
        elif accidental == 'b':
            self.draw_flat((x - single_offset, y - flat_y_offset))
        elif accidental == 'bb':
            self.draw_flat((x - double_offset, y - flat_y_offset), 'double')
        elif accidental == 'bbb':
            self.draw_flat((x - triple_offset, y - flat_y_offset), 'triple')
        elif accidental == '#':
            self.draw_sharp((x - single_offset - sharp_x_offset, y))
        elif accidental == '##':
            self.draw_sharp((x - double_offset + sharp_x_offset,
                             y + double_sharp_y_offset), 'double')
        elif accidental == '###':
            self.draw_sharp((x - triple_offset - sharp_x_offset, y), 'triple')
        else:
            raise ValueError("Sorry, " + note_name + " has over 3 sharps/flats.")

    def _draw_line(self, start: tuple[float, float], end: tuple[float, float],
                   width: Optional[int] = None) -> None:
        """Draw a line in pygame."""
        if width is None:
            pygame.draw.line(self._screen, DARK, start, end, math.ceil(self.scaling / 10))
        else:
            pygame.draw.line(self._screen, DARK, start, end, width)

    def _draw_image(self, img: pygame.image, coords: tuple[float, float]) -> None:
        """Draw an image in pygame."""
        self._screen.blit(img, coords)

    def _draw_ledger_line(self, lines: int, x: float, y0: float, offset: int, up: bool) -> None:
        """Draw a ledger line on pygame."""
        for i in range(lines):
            if up:
                y = y0 + (i + 0.5) * self.scaling + (0.5 * self.scaling if offset % 2 == 0 else 0)
            else:
                y = y0 + (0.5 - i) * self.scaling - (0.5 * self.scaling if offset % 2 == 0 else 0)
            self._draw_line((x - self.scaling * 0.25, y), (x + self.scaling * 1.9, y))

    def _draw_text(self, text: str, x: int, y: int, bold: bool = False, italic: bool = False,
                   size: int = 24, colour: tuple[int, int, int] = BLUE) -> None:
        """Draw a line of text in pygame."""
        font = pygame.font.SysFont("Tahoma", size, bold=bold, italic=italic)
        text = font.render(text, True, colour)
        self._screen.blit(text, (x, y))

    def _draw_rect(self, coord: tuple[int, int], width: int, height: int,
                   colour: tuple[int, int, int] = BG_COL, alpha: int = 233) -> None:
        """Draw a rectangle in pygame."""
        s = pygame.Surface((width, height))
        s.set_alpha(alpha)
        s.fill(colour)
        self._screen.blit(s, coord)


def load_image(file: str, scaling: float) -> pygame.image:
    """Load up an image and convert it to a pygame image."""
    img = pygame.image.load(file)
    w = img.get_width()
    h = img.get_height()
    return pygame.transform.smoothscale(img.convert_alpha(), (w / h * scaling, scaling))
