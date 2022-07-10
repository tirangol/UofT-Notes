"""Cantus Firmus

There is a notational quirk between this program and regular musical notation.
- B#4 is equivalent to C5 normally
- B#4 is equivalent to C4 in this program (this is done to make octave calculations easier)

Here are some example voices you can try out:
- 'Bb3 Bb4 B4 Eb3 Eb3 A4 Bb4 B4 F#4 C#4 A#4'
- 'D3 E3 F3 G3 E3 C3 D3'
- 'E4 A4 B4 G4 D5 C5 F5 E5 A4 A5 G5 F5 E5 D5 E5 D5 B4 F#4 D5 C5 B4 A4 G#4 A4'
- 'E4 A4 B4 G4 D5 C5 F5 E5 A4 A5 G5 F5 E5 D5 E5 C5 B4 F4 D5 C5 B4 A4 G4 A4'
- 'C1 D3 D#2, F###4, G3, Ebb3, D2, D2, D2, C3, Eb3, F3, F#3, F3, Eb3, C3, Bb2, D3, C2'

The following terms in the display mean this:
- LEAP   A note after a P8/m6 leap must be between the previous two notes
- INT.   The interval is invalid (ie. compound (9th+), augmented, diminished, 7th, descending m6)
- M6     Major sixth detected (not invalid, but rare)
- m7     Descending P5 and m3 is invalid
- SEQ.   A one-directional note sequence's first/last note cannot be A4, M7, or a compound interval
- SKIP   At most 2 skips (ie. conjunct intervals) in the same direction
- m2     No chromatic progressions (ie. 3+ chromatic notes in the same direction)
- P4     A minor seventh may not consist of two perfect fourths

The different singing ranges are (inclusive):
- Soprano   C4 to A5
- Alto      F3 to D5
- Tenor     C3 to A4
- Bass      E2 to D4

You can also separate values in commands with '', "", or commas - these characters will be ignored.
"""
from visualizer import *


def simplify_input(text: str) -> str:
    """Simplify the input."""
    i = text.lower().replace(',', ' ').replace("'", ' ').replace('"', ' ').replace('[', ''). \
        replace(']', '')
    while "  " in i:
        i = i.replace("  ", " ")
    if len(i) <= 1:
        return i
    if i[0] == ' ':
        i = i[1:]
    if i[-1] == ' ':
        i = i[:-1]
    return i


if __name__ == "__main__":
    screenx, screeny = 1000, 600
    margin, textsize = 50, 15
    scale, key, voice = None, None, []
    print('Cantus Firmus (by Richard Yin)')
    print('Please type the following commands:')
    print()
    print('screen x y             - sets the screen size (default = 1000 x 600, minwidth = 200)')
    print('margin x               - sets the x margins to x (default = 50)')
    print('textsize x             - sets the text size to x (default = 15)')
    print('voice n1 n2 ...        - type in the notes of a melody/voice to analyze')
    print('scale c major          - type in the key and scale-type of the voice (default = None)')
    print('custom_scale n1 n2 ... - type in a custom scale')
    print('scale_key c            - type in the key of the scale')
    print('display                - open a window that analyzes the voice')
    print()
    print('Custom scales should not be written with octave markings/duplicate notes.')
    print('When displayed, m will play the melody and space will toggle text annotations.')
    inp = simplify_input(input())

    while True:
        while all(x not in inp for x in {"screen", "margin", "voice", "display", "scale",
                                         "textsize"}):
            print("Unrecognized input. Please try again.")
            inp = simplify_input(input())

        try:
            if inp == 'display':
                if voice == []:
                    print('Error, no voice has been created yet.')
                if scale is None:
                    v = Visualizer(voice, textsize, (screenx, screeny))
                    print("Displaying voice...")
                    v.display(margin)
                    print("Successfully displayed voice.")
                else:
                    v = Visualizer(Voice(voice.as_str, scale), textsize, (screenx, screeny))
                    print("Displaying voice...")
                    v.display(margin)
                    print("Successfully displayed voice.")

            elif inp[:6] == 'margin' and inp.count(" ") == 1:
                _, arg = inp.split(" ")
                margin = round(float(arg))
                print("Successfully changed x margin to " + str(margin))

            elif inp[:6] == 'screen' and inp.count(" ") == 2:
                _, arg1, arg2 = inp.split(" ")
                if round(float(arg1)) <= 200 or round(float(arg2)) <= 0:
                    print("Error, entered width/height is too low")
                else:
                    screenx, screeny = round(float(arg1)), round(float(arg2))
                    print("Successfully changed screen size to " + str(screenx) + " by " +
                          str(screeny))

            elif inp[:8] == 'textsize' and inp.count(' ') == 1:
                _, arg = inp.split(" ")
                textsize = round(float(arg))
                print("Successfully changed text size to " + str(textsize))

            elif inp[:5] == 'scale' and inp.count(" ") == 2:
                _, arg1, arg2 = inp.split(" ")
                scale = Scale(arg1, arg2)
                print("Successfully set scale to " + str(scale))

            elif inp[:12] == 'custom_scale':
                if inp.count(" ") == 0:
                    print("Error, no notes entered in scales")
                else:
                    args = inp.split(" ")[1:]
                    scale = Scale(key=key, notes=args)
                    print("Successfully set scale to " + str(scale))

            elif inp[:9] == 'scale_key' and inp.count(" ") == 1:
                _, arg = inp.split(" ")
                key = arg.upper() if len(arg) == 1 else arg[0].upper() + arg[1:]
                if scale is None:
                    scale = Scale(key=key, notes=[key + '3'])
                else:
                    scale.key = key
                    if scale.scale_type is not None:
                        scale = Scale(key, scale.scale_type)
                print("Successively changed key of scale to " + key)

            elif inp[:5] == 'voice':
                if inp.count(" ") == 0:
                    print("Error, no notes entered in notes")
                else:
                    args = inp.split(" ")[1:]
                    voice = Voice(args)
                    print("Successfully set voice to " + str(voice.as_str))
            else:
                print("Unrecognized input. Please try again.")
        except BaseException as e:
            print("The following error occured: " + str(e))

        inp = simplify_input(input())
