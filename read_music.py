import numpy as np
import cv2
import math
from  matplotlib import pyplot  as plt
from sklearn.cluster import KMeans

def find_lines():
    img = cv2.imread('music.png')
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
    print(line_pts)
    hor_vals = []
    for pt1, pt2 in line_pts:
        if pt1[1] == pt2[1]:
            hor_vals.append(pt1[1])
    hor_vals = np.array(hor_vals)
    model = KMeans(5).fit(hor_vals.reshape((-1, 1)))
    print(model.cluster_centers_)
    return model.cluster_centers_.reshape(5)

def find_circles():
    img = cv2.imread('music.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,1,param1=100,param2=100,minRadius=0,maxRadius=100)
    print(circles)

find_circles()
