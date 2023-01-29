
# Cantus Firmus Note Visualizer and Player

Cantus Firmus ("Fixed Melody" in Latin) is a collection of Renaissance-era conventions for composing melodies for Western classical music (more info at https://en.wikipedia.org/wiki/Cantus_firmus). It is formulated as many rules, such as:

- If note A jumps an octave to note B, the following note C must be between notes A and B
- If a sequence of notes are increasing, then the distance between the first and last note of the sequence must be a certain interval

This is not exhaustive. These rules can be logically encoded, so I decided to do build a system for working with musical notes from the ground-up, and used midiutil and pygame to create a GUI for checking whether melodies follow these rules.

This program supports:
- Displaying notes on a musical staff of any size
- Downloading and playing a melody that is written
- Detect whether to display notes in bass or treble clef
- Reading key signatures and formatting notes according to them

<p align="center">
<img src="cantusfirmus.gif" width="500 px">
</p>

Upon running `main.py`, a command-line interface will allow the following commands:
- `screen x y` - set the screen size to x by y pixels (1000 by 600 by default, with a minimum with of 200)
- `margin x` - set the x-margins to x (15 pixels by default) (this is the distance between horizontal ends of the screen with staff)
- `textsize x` - set text size of error annotations to x (15 pt by default)
- `voice n1 n2 n3 ...` - type in the notes of a melody/voice to analyze. Has to be a note followed by an octave (C4 is middle C). Sharps are # and flats are b. There is GUI support for double-sharps (##) and double-flats (bb), but not trilpe sharps and flats.
- `scale c major` - type in a key and scale-type (default: c major, aka no flats or sharps). Mainly for the purpose of setting default accidentals
- `custom scale n1 n2 n3 ...` - type in the notes of a custom scale. No octave numbers and no duplicate notes allowed.
- `scale_key c` - set the key the scale is centered on (default: c major, aka no flats or sharps). Determines the key signature.
- `display` - display the window and analyze the voice.

When the displayer is open, you can scroll up and down, press m to play the melody, and press space to toggle text error annotations.
