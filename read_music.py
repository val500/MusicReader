import numpy as np
import cv2
import math
from  matplotlib import pyplot  as plt
from sklearn.cluster import KMeans

def scan_image(imgName, num_lines):
    #imgName = 'music.png'
    img = cv2.imread(imgName)
    lines = find_lines(imgName, num_lines)
    circles = find_circles(imgName)
    note_arr = find_notes(circles, lines, num_lines)
    return note_arr

def find_lines(imgName, num_lines):
    img = cv2.imread(imgName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 120)
    #plt.imshow(edges)
    line_pts = []
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 100  # minimum number of votes (intersections in Hough grid cell)

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]))
    lines = lines.reshape((lines.shape[0], lines.shape[2]))
    for line in lines:
        pt1 = (line[0], line[1])
        pt2 = (line[2], line[3])
        img = cv2.line(img, pt1, pt2, (0,0,255), 3)
        line_pts.append((pt1, pt2))
    plt.imshow(img)
    hor_vals = []
    for pt1, pt2 in line_pts:
        if pt1[1] == pt2[1]:
            hor_vals.append(pt1[1])
    hor_vals = np.array(hor_vals)
    model = KMeans(num_lines * 5).fit(hor_vals.reshape((-1, 1)))
    return model.cluster_centers_.reshape(num_lines * 5)

def find_circles(imgName):
    img = cv2.imread(imgName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
    img = cv2.drawKeypoints(img, keypoints, img)
    pts = cv2.KeyPoint_convert(keypoints)
    sort_pts = pts[pts[:,0].argsort()]
    return sort_pts[:,1].reshape(-1)

def find_notes(notes, lines, num_lines):
    all_line_pos = []
    lines = np.sort(lines)[::-1]
    bare_notes = ['d1', 'e1', 'f1', 'g1', 'a1', 'b1', 'c1', 'd2', 'e2', 'f2', 'g2']
    note_vals = []
    for l_n in range(num_lines, 0, -1):
        for b in bare_notes:
            note_vals.append(b + str(l_n))
    
    for line_num in range(num_lines):
        all_line_pos.append(lines[line_num * 5] + (lines[line_num * 5] - ((lines[line_num * 5] + lines[line_num*5 + 1]) / 2)))
        for i in range(4):
            all_line_pos.append(lines[line_num*5 + i])
            all_line_pos.append((lines[line_num*5 + i] + lines[line_num*5 + i+1]) / 2)
        all_line_pos.append(lines[line_num * 5 + 4])
        all_line_pos.append(lines[line_num * 5 + 4] + (lines[line_num * 5 + 4] - ((lines[line_num * 5 + 3] + lines[line_num * 5 + 4]) / 2)))
        
    note_arr = []
    for n in notes:
        note_arr.append(note_vals[np.argmin(np.abs(n - all_line_pos))])

    new_note_arr = sorted(note_arr, key=lambda a: a[2])
    new_note_arr = [a[0:2] for a in new_note_arr]
    return np.array(new_note_arr)


# Taken from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python, calculates edit distance
def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]
