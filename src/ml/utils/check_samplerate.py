import os
import wave

data_dir = ''
for file_name in os.listdir(data_dir):
    with wave.open(data_dir + '/' + file_name, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        print(frame_rate)