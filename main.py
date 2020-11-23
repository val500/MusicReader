import sys
import musicalbeeps
from read_music import scan_image
import cv2
from  matplotlib import pyplot  as plt

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
    
main()


