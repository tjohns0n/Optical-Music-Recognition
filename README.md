# OMR program
This is a Python project written to convert images of sheet music into MIDI format. Currently, it only works with single staff, single melody music, such as the examples provided in the `images` directory. Only C major and G major key signatures are supported at this point. 

## Requirements:
- FluidSynth and ALSA installed on computer
- FluidR3 soundfont installed in the location: `/usr/share/soundfonts/FluidR3_GM.sf2`
- midiutil, numpy, and OpenCV 2 Python packages installed

## Running:

```
python3 OMR.py [path to image] G (optional)
```

Only specify 'G' as an argument if the music is in G major. Otherwise, leave out the second argument. For example:

```
python3 OMR.py images/clair.png # Plays Au Clair de la Lune, a piece in C major
python3 OMR.py images/yankee.png G # Plays Yankee Doodle, a piece in G major
```