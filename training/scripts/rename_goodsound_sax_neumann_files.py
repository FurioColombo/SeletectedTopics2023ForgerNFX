import math
import shutil
from training.modules.utils.file_system import get_file_paths_in_folder
from librosa import note_to_midi

OG_DIR = r"D:/datasets/good-sounds-datataset/good-sounds/sound_files/saxo_tenor_raul_recordings/neumann-take-number/"
NOTE_NAME_DIR = r"D:/datasets/good-sounds-datataset/good-sounds/sound_files/saxo_tenor_raul_recordings/neumann-note-names/"
GUITAR_FRET_DIR = r"D:/datasets/good-sounds-datataset/good-sounds/sound_files/saxo_tenor_raul_recordings/neumann-guitar-fret/"
WAV_signature = ".wav"

notes = "G# A A# B C C# D D# E F F# G".split()


def rename_files_from_take_num2note_num(source_file_paths, target_folder):
    for file_path in source_file_paths:
        old_file_name = file_path.split('/')[-1]
        old_file_name = old_file_name.split('.')[0]
        if old_file_name.isdigit():
            old_file_name = int(old_file_name)
        else:
            print(file_path, 'not renamed')
            shutil.copy(file_path + old_file_name + WAV_signature, target_folder + old_file_name + WAV_signature)
        if math.floor(old_file_name / 33) < 7:
            sax_note_number = old_file_name % 33
            # letter name - octave
            octave_number = 2 + math.floor((sax_note_number + 7) / 12)
            note_name = notes[sax_note_number % 12] + str(octave_number)
            new_file_name = note_name + '-' + str(math.floor(old_file_name / 33)) + WAV_signature
            new_path = target_folder + new_file_name
            shutil.copy(file_path, new_path)


def rename_files_from_note_num2guitar_fret(source_file_paths, target_folder):
    for file_path in source_file_paths:
        old_file_name = file_path.split('/')[-1]
        old_file_name = old_file_name.split('.')[0]
        note_name, note_string = old_file_name.split('-')
        note_string = int(note_string)
        midi_note_num = int(note_to_midi(note_name))
        fret_number = -1
        print('note_string:', note_string)
        # high e string
        if note_string == 0:
            fret_number = midi_note_num - 64

        if note_string == 1:
            fret_number = midi_note_num - 59

        if note_string == 2:
            fret_number = midi_note_num - 55

        if note_string == 3:
            fret_number = midi_note_num - 50

        if note_string == 4:
            fret_number = midi_note_num - 45

        if note_string == 5:
            fret_number = midi_note_num - 40

        print('fret_number = midi_note_num - x:\n',
              fret_number, '=', midi_note_num, '-', str(64 - int(note_string) * 5))

        if 0 <= fret_number <= 22:
            new_file_name = str(1 + int(note_string)) + '-' + str(fret_number) + WAV_signature
            new_path = target_folder + new_file_name
            shutil.copy(file_path, new_path)


rename_files_from_note_num2guitar_fret(get_file_paths_in_folder(NOTE_NAME_DIR), GUITAR_FRET_DIR)
