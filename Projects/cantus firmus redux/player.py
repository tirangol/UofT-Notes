"""Cantus Firmus

Classes
- Player
"""
from voice import *
from midiutil import MIDIFile
import pygame


class Player:
    """A player of a voice.

    Instance Attributes:
    - voice: the voice to play
    - tempo: the tempo to play at
    - volume: the volume to play at
    - name: the name of the midi file
    """
    voice: Voice
    tempo: int
    volume: int
    name: str

    def __init__(self, voice: Voice, name: str = "cantus firmus", tempo: int = 120,
                 volume: int = 100) -> None:
        """Initialize the player."""
        self.voice = voice
        self.name = name
        self.tempo = tempo
        self.volume = volume

    def create_midi(self) -> None:
        """Create a midi file of the voice."""
        mf = MIDIFile(1)
        track, channel, time, note_duration = 0, 0, 0, 1
        mf.addTrackName(track, time, "voice 1")
        mf.addTempo(track, time, self.tempo)

        for i in range(len(self.voice.as_note)):
            mf.addNote(track, channel, self.voice.as_note[i].absolute() + 12,
                       i, note_duration, self.volume)

        with open(self.name + '.mid', 'wb') as outfile:
            mf.writeFile(outfile)

        pygame.mixer.init()
        pygame.mixer.music.load(self.name + '.mid')

    def play_midi(self) -> None:
        """Play the midi file on pygame."""
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()
