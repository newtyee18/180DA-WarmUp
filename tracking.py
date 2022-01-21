# References used:
# 1) https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
# 2) https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
# 3) https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
# 4) https://stackoverflow.com/questions/51699162/im-having-error-with-cv2-findcontours
# 5) https://gist.github.com/bigsnarfdude/d811e31ee17495f82f10db12651ae82d
# 6) https://stackoverflow.com/questions/51699162/im-having-error-with-cv2-findcontours
# 7) https://stackoverflow.com/questions/22588146/tracking-white-color-using-python-opencv
# 8) https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097

from __future__ import print_function
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

cap = cv.VideoCapture(0)

lower1 = np.array([70,60,160])          # lower HSV threshold for cyan
upper1 = np.array([135,255,255])        # upper HSV threshold for cyan
lower2 = np.array([220,220,220])
upper2 = np.array([255,255,255])
kernelOpen = np.ones((7,7))
kernelClose = np.ones((20,20))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Implementing Kmeans
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_mask = cv.inRange(hsv, lower1, upper1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    rgb_mask = cv.inRange(rgb, lower2, upper2)
    
    mask3 = cv.morphologyEx (hsv_mask, cv.MORPH_OPEN, kernelOpen)
    mask4 = cv.morphologyEx (mask3, cv.MORPH_CLOSE, kernelClose)

    mask5 = cv.morphologyEx (rgb_mask, cv.MORPH_OPEN, kernelOpen)
    mask6 = cv.morphologyEx (mask5, cv.MORPH_CLOSE, kernelClose)
    
    # Thresholding the frame for grayscale
    ret,thresh1 = cv.threshold(gray,127,255,cv.THRESH_BINARY)

    # Kmeans for rgb
    # reshaped_rgb = rgb.reshape((rgb.shape[0] * rgb.shape[1], 3)) # comment out if want to get bounding box video
    # clt = KMeans(n_clusters=5) # comment out if want to get bounding box video; kmeans is REALLY SLOW AND WILL LAG VIDEO
    # clt.fit(reshaped_rgb) # comment out if want to get bounding box video

    contours, h = cv.findContours(hsv_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv.rectangle(hsv, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv.drawContours(hsv, contours, -1, (255, 255, 0), 1)
    # Display the resulting frame
    # hist = find_histogram(clt) # comment out if want to get bounding box video
    # bar = plot_colors2(hist, clt.cluster_centers_) # comment out if want to get bounding box video
    cv.imshow("frame", hsv) # change rgb to hsv to get bounding box video
   # plt.axis("off") # comment out if want to get bounding box video
   # plt.imshow(bar) # comment out if want to get bounding box video
   # plt.show() # comment out if want to get bounding box video
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()