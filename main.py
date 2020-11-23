import sys
import musicalbeeps
from read_music import scan_image, levenshtein
import cv2
from  matplotlib import pyplot  as plt
import numpy as np

def main():
    args = sys.argv[1:]
    if len(args) != 2:
        print("Too many args!")
        return
    filename = args[0]
    num_lines = args[1]
    
    notes = scan_image(filename, int(num_lines))
    print(notes)
    img = cv2.imread(filename)
    plt.imshow(img)
    plt.show()
    player = musicalbeeps.Player(volume = 0.3, mute_output = False)
    note_map = {'d1': 'D', 'e1': 'E', 'f1': 'F', 'g1': 'G', 'a1': 'A', 'b1': 'B', 'c1': 'C5', 'd2': 'D5', 'e2': 'E5', 'f2': 'F5', 'g2': 'G5'}
    
    for note in notes:
        player.play_note(note_map[note])
    

def evaluate():
    img_folder = "music_no_clef"
    files = ["music1.png", "music2.png", "music3.png", "music4.png", "music5.png", "two_lines.jpg"]

    total_dist = 0
    for f in files:
        fname = f.split('.')[0]
        correct_notes = np.load("music_no_clef/" + fname + ".npy")
        num_lines = 1
        if (fname == "two_lines"):
            num_lines = 2
        predicted_notes = scan_image("music_no_clef/" + f, num_lines)
        edit_distance = levenshtein(correct_notes, predicted_notes) / num_lines
        total_dist += edit_distance
        print(fname + ": " + str(edit_distance))
    print("Average Edit Distance : " + str(total_dist / len(files)))
main()

        
    
