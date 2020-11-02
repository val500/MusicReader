import numpy as np
import cv2
import math
from  matplotlib import pyplot  as plt
from sklearn.cluster import KMeans

def scan_image():
    imgName = 'music.png'
    img = cv2.imread(imgName)
    lines = find_lines(imgName)
    circles = find_circles(imgName)
    note_arr = find_notes(circles, lines)
    return note_arr

def find_lines(imgName):
    img = cv2.imread(imgName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 120)
    #plt.imshow(edges)
    line_pts = []
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]))
    lines = lines.reshape((lines.shape[0], lines.shape[2]))
    for line in lines.reshape((3180, 4)):
        pt1 = (line[0], line[1])
        pt2 = (line[2], line[3])
        cv2.line(img, pt1, pt2, (0,0,255), 3)
        line_pts.append((pt1, pt2))
    hor_vals = []
    for pt1, pt2 in line_pts:
        if pt1[1] == pt2[1]:
            hor_vals.append(pt1[1])
    hor_vals = np.array(hor_vals)
    model = KMeans(5).fit(hor_vals.reshape((-1, 1)))
    return model.cluster_centers_.reshape(5)
    
def find_circles(imgName):
    img = cv2.imread(imgName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
    pts = cv2.KeyPoint_convert(keypoints)
    sort_pts = pts[pts[:,0].argsort()]
    return sort_pts[:,1].reshape(-1)

def find_notes(notes, lines):
    all_line_pos = []
    lines = np.sort(lines)[::-1]
    note_vals = ['d1', 'e1', 'f1', 'g1', 'a1', 'b1', 'c1', 'd2', 'e2', 'f2', 'g2']
    all_line_pos.append(lines[0] + (lines[0] - ((lines[0] + lines[1]) / 2)))
    for i in range(0, len(lines) - 1):
        all_line_pos.append(lines[i])
        all_line_pos.append((lines[i] + lines[i+1]) / 2)
    all_line_pos.append(lines[-1])
    all_line_pos.append(lines[-1] + (lines[-1] - ((lines[-2] + lines[-1]) / 2)))
    all_line_pos = np.array(all_line_pos)
    note_arr = []
    for n in notes:
        note_arr.append(note_vals[np.argmin(np.abs(n - all_line_pos))])
    return np.array(note_arr)
