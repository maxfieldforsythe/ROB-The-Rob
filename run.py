import os
import subprocess
import glob
from math import radians
import cv2
import numpy as np

# Setup Variables -----------------------------------------------------------------------------------------------------
from numpy import float32
# Maze file (Choose from Maze1.blend, Maze2.blend, Maze3.blend)
maze = "Maze1.blend"
# Prints Blender Output when set to True (We recommend it remains False)
verbose = False
# Fast Render (Reduces the number of frames rendered)
# When Fast Render is set to True, the execution time is about 5x faster
fastRender = True

# Program -------------------------------------------------------------------------------------------------------------
# Aruco setup
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
MARKER_LENGTH = 0.25  # 0.25 Meters

# Facial Recognition Setup
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Camera Matrix Setup
focal = 0.08  # Focal length is 80 mm
sensorWidth = 0.1  # Sensor width is 100 mm
fx = focal / sensorWidth * 1920
fy = focal / sensorWidth * 1920
cx = 1920 / 2
cy = 1080 / 2
k = np.array([[focal / sensorWidth * 1920, 0., 1920 / 2],
              [0., focal / sensorWidth * 1920, 1080 / 2],
              [0., 0., 1.]])

# Simulation Starting Variables
imgNum = 0
distance = 0
rotate = 0

# ROB's starting position
x, y, z = 13.931, -7.037, 1.1274
startRotate = radians(0)

# Ensures simulation does not get stuck in a loop
hasId = True
count = 0


# Creates an output video of the run using the rendered images
def createVideo(folder, outputName, show_frameNum=True, show_Item=True, show_tvecs=False, show_rvecs=False):
    # Setup video output
    frameNum = len(glob.glob(str(folder)+"/*.png"))
    img = cv2.imread(str(folder)+"/0.png")
    size = img.shape
    out = cv2.VideoWriter(str(outputName)+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 22, (size[1], size[0]))

    print("Creating", str(outputName)+".avi", "using", str(frameNum), "Rendered Images from", str(folder),
          "\nThis process may take several seconds...")

    # Iterate through Image Renders and alter images as necessary, then add the images to the output video
    for i2 in range(0, frameNum):
        frame = cv2.imread(str(folder)+"/"+str(i2)+".png")
        if show_Item:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces2 = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
            # detect markers
            corners2, ids2, _ = cv2.aruco.detectMarkers(image=frame, dictionary=arucoDict)
            faceId2 = 2
            idsTemp2 = None
            for (x2, y2, w2, h2) in faces2:
                corners2.append(np.array([[[x2, y2], [x2+w2, y2], [x2+w2, y2+h2], [x2, y2+h2]]], dtype=float32))
                if ids2 is None and idsTemp2 is None:
                    idsTemp2 = []
                if idsTemp2 is None:
                    ids2 = np.append(ids2, [faceId2])
                else:
                    idsTemp2.append([faceId2])
                faceId2 += 1
            if ids2 is None and idsTemp2 is not None:
                ids2 = np.array(idsTemp2)
            if ids2 is not None:
                # Draw detected markers and calculate pose
                cv2.aruco.drawDetectedMarkers(image=frame, corners=corners2, ids=ids2, borderColor=(0, 0, 255))
                rvecs2, tvecs2, _ = cv2.aruco.estimatePoseSingleMarkers(corners=corners2, markerLength=MARKER_LENGTH,
                                                                        cameraMatrix=k, distCoeffs=None)
                cv2.aruco.drawAxis(image=frame, cameraMatrix=k, distCoeffs=None, rvec=rvecs2[0], tvec=tvecs2[0],
                                   length=MARKER_LENGTH)
                # Adds tvecs information to video output
                if show_tvecs:
                    text = tvecs2[0][0]
                    for m in range(len(text)):
                        text[m] = "{:.3f}".format(text[m])
                    cv2.putText(frame, "tvecs: "+str(text), (20, 1000), fontScale=1, color=(0, 255, 255), thickness=2,
                                fontFace=cv2.FONT_HERSHEY_DUPLEX)
                # Adds rvecs information to video output
                if show_rvecs:
                    text = rvecs2[0][0]
                    for m in range(len(text)):
                        text[m] = "{:.3f}".format(text[m])
                    cv2.putText(frame, "rvecs: "+str(text), (20, 1050), fontScale=1, color=(0, 255, 255), thickness=2,
                                fontFace=cv2.FONT_HERSHEY_DUPLEX)
        # Add frame number to video
        if show_frameNum:
            cv2.putText(frame, str(i2), (20, 40), fontScale=1, color=(0, 255, 255), thickness=2,
                        fontFace=cv2.FONT_HERSHEY_DUPLEX)
        # Write images to video output
        out.write(frame)
    print("\nVideo Creation Completed!\nOutput saved to", str(outputName)+".avi\n")


if True:
    print("Beginning Traversal of", str(maze), "\nThis process may take several Minutes. Please be Patient...\n")

    # Remove old Renders from OverheadImages and OutputImages files
    files = glob.glob("OverheadImages/*")
    for f in files:
        os.remove(f)
    files = glob.glob("OutputImages/*")
    for f in files:
        os.remove(f)

    # Run the Maze Traversal
    while hasId and count < 600:
        # Prevents endless traversal
        count += 1
        # Create a Valid argument for
        args = "BlenderFiles\\Blender\\blender.exe --background --python main.py "+str(imgNum)+" "+str(x)+" "+str(
            y)+" "+str(
            z)+" "+str(startRotate)+" "+str(distance)+" "+str(rotate)+" "+str(os.getcwd())+" "+str(maze)+" "+str(
            fastRender)
        # Prints all output from the Blender Rendering Process (Not Recommended)
        if verbose:
            subprocess.call(args, shell=False)
        # Prints only necessary outputs
        else:
            FNULL = open(os.devnull, 'w')
            subprocess.call(args, stdout=FNULL, stderr=FNULL, shell=False)

        # Gets ROB's new position before executing the next leg of the traversal
        f = open("position.txt", "r")
        pos = f.read().split()
        x, y, z, startRotate = pos[0], pos[1], pos[2], pos[3]

        # Find most recent Render for use in CV techniques
        files = glob.glob("OutputImages/*.png")
        imgNum = files.index(max(files))
        render = cv2.imread("OutputImages/"+str(imgNum)+".png")
        imgNum += 1

        # Convert the image to grayscale for face detection
        grayRender = cv2.cvtColor(render, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(grayRender, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        # detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(image=render, dictionary=arucoDict)

        # Adds found faces to the ArUco corners / ids variables so a pose estimate can be performed
        faceId = 2
        idsTemp = None
        for (xFace, yFace, w, h) in faces:
            # Add face corners to ArUco corners
            corners.append(
                np.array([[[xFace, yFace], [xFace+w, yFace], [xFace+w, yFace+h], [xFace, yFace+h]]], dtype=np.float32))
            if ids is None and idsTemp is None:
                idsTemp = []
            # Add found face to id list
            if idsTemp is None:
                ids = np.append(ids, [faceId])
            else:
                idsTemp.append([faceId])
            faceId += 1
        if ids is None and idsTemp is not None:
            ids = np.array(idsTemp)
        # Determine how to traverse the maze based on detected markers
        if ids is not None:
            # Draw detected markers and calculate pose
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners=corners, markerLength=MARKER_LENGTH,
                                                                  cameraMatrix=k, distCoeffs=None)
            # Determine rotation and translation based on id and pose
            maxD = 0
            for i in range(0, len(tvecs)):
                idDistance = (ids[i][0] - (ids[i][0] % 2))/2
                if ids[i] > 500:
                    idDistance = 1

                if tvecs[i][0][2] <= idDistance:
                    print("ROB has detected an ArUco at Position: ("+str(round(float(x), 2))+", "+str(
                        round(float(y), 2))+", "+str(round(float(z), 2))+")")
                    print("Traversing to next position...")
                    distance = 0
                    if ids[i] % 2 == 0:
                        rotate = 90
                    else:
                        rotate = -90
                    if ids[i] > 500:
                        hasId = False
                else:
                    if fastRender:
                        if maxD == 0:
                            distance = (tvecs[i][0][2] - idDistance) + 0.3
                            maxD = distance
                        else:
                            if maxD > (tvecs[i][0][2] - idDistance) + 0.3:
                                distance = (tvecs[i][0][2]-idDistance)+0.3
                                maxD = distance
                    else:
                        distance += 0.3
                    rotate = 0
if count >= 600:
    print("Maze traversal failed :(")
else:
    print("Traversal of", str(maze), "Completed!\n")

    # Create a video output based on the Program Renders
    print("Creating Video Outputs:\n")
    createVideo("OutputImages", "cvBlenderRobot", show_tvecs=True, show_rvecs=True)
    createVideo("OverheadImages", "cvBlenderOverhead", show_Item=False)
