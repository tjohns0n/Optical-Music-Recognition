# Messy code ahead!
# This code was written when I was still learning Python and is pretty gnarly. 
# In the future, I'd like to expand on this project to work on a greater variety
# of sheet music. I'll clean up the library calls, refactor heavily, and rewrite
# everything to conform to PEP8 at that point. 

from midiutil.MidiFile import MIDIFile
import numpy as numpy
from subprocess import call
from operator import itemgetter

import cv2
import statistics
import math
import sys

# Methodically remove lines if they are not part of a staff on a page
# lines: [(magnitude, direction in radians), ...] format list
# distances: the distances between 0 and 1, 1 and 2, and so on in lines
# median: the median distance in distances
def deleteLines(lines, distances, median):
    temp = []
    median_low = median * .9
    median_high = median * 1.1 
    counter = 0
    for i in range(0, len(distances)):
        inc = False
        if distances[i] > median_low and distances[i] < median_high:
            counter += 1
            inc = True
        if inc == False:
            if counter != 4:
                if i < len(lines):
                    temp.append(lines[i])
            counter = 0 
    return [element for element in lines if element not in temp]

def getStaffImage(image, lineGroup, median):
    top = int(round(lineGroup[0] - (median * 2)))
    bottom = int(round(lineGroup[4] + (median * 2)))
    return image[top:bottom, 150:len(image[1]) - 120]

def getNotes(images, templates, eighths, rests, threshold, method):
    notes = []
    tempNotes = []
    for staffImage in images:
        q = []
        for template in templates:
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(staffImage, template, method)
            loc = None 
            if method == cv2.TM_SQDIFF_NORMED:
                loc = numpy.where(res <= threshold)
            else:
                loc = numpy.where(res >= threshold)
            zipped = zip(*loc[::-1])
            zipped = sorted(zipped)
            lastX = -1 * w
            for pt in zipped:
                if pt[0] > lastX + w:
                    #rectangle(staffImage, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
                    #lastX = pt[0]
                    q.append(pt)
        tempNotes.append(q)

    for staffImage, q in zip(images, tempNotes):
        qs = []
        q = sorted(q)
        w, h = templates[0].shape[::-1]
        lastX = -1 * w
        for pt in q:
            if pt[0] > lastX + w * 1.2:
                qs.append((pt, 4))
                lastX = pt[0]
        notes.append(qs)
    
    return notes


def getBlackRows(images):
    black = []
    rows, cols = images.shape
    for i in range(0, cols):
        sum = 0
        for j in range(0, rows):
            pixel = images[j,i]
            if pixel < 127:
                sum += 1
        black.append(sum)
    return black

def getBlackCols(image):
    black = []
    rows, cols = image.shape
    for i in range(0, rows):
        sum = 0
        for j in range(0, cols):
            pixel = image[i, j]
            if pixel < 127:
                sum += 1
        black.append(sum)
    return black

def getNotePosition(black, quarter):
    eighth = False
    candidates = []
    counter = 0
    start = 0
    end = 0
    for i in range(0, len(black)):
        if black[i] > .1:
            if counter == 0:
                start = i
            counter += 1
        else:
            if counter > 3:
                end = i
                candidates.append((start, end))
                start = 0
                end = 0
            counter = 0

    if (quarter):
        print(candidates)
        if len(candidates) > 1:
            eighth = True

    candidates = max(candidates, key=(lambda x: x[1] - x[0]))
    candidates = int(candidates[0] + ((candidates[1] - candidates[0]) / 2))
    return candidates, eighth

def analyzeBlack(blackPerCol, ignore, ignore2, ignoreWidth):
    quarters = [list(range(q[0][0], q[0][0] + ignoreWidth)) for q in ignore]
    quarters.extend([list(range(q[0][0], q[0][0] + ignoreWidth)) for q in ignore2])
    quarters = [item for sublist in quarters for item in sublist]
    halves = []
    wholes = []
    hRests = []

    blackPerCol = [b / max(blackPerCol) for b in blackPerCol]

    noiseLevel = min(blackPerCol)
    
    stem = False 
    start = -1
    end = -1
    tallCounter = 0
    counter = 0
    consistent = False
    height = 0
    for col, i in zip(blackPerCol, list(range(0, len(blackPerCol)))):
        if col > noiseLevel:
            if counter == 0:
                height = col
                consistent = True
                start = i
            if (abs(height - col) > 0):
                consistent = False
            counter += 1
        if col > .4:
            tallCounter += 1
        if col <= noiseLevel: 
            if counter > 5 and tallCounter < 4:
                end = i 
                if tallCounter >= 1:
                    halves.append((start, end))
                elif consistent:
                    hRests.append(start)
                else:
                    wholes.append((start, end)) 
            consistent = False
            counter = 0
            tallCounter = 0
            start = 0
            end = 0
    
    validWholes = []
    halves = [h for h in halves if h[0] not in quarters and h[1] not in quarters and h[0] > 10]

    deletions = []
    for i in range(0, len(wholes)):
        if i + 1 < len(wholes) and wholes[i + 1][0] - wholes[i][0] < 15:
            deletions.append(wholes[i])

    wholes = [w for w in wholes if w not in deletions]
    return wholes, halves, hRests


# Read the image:
image = cv2.imread(sys.argv[1], 1)

# Resize the image (not necessary, as far as I can tell)
width = 1800
originalHeight, originalWidth = image.shape[:2]
aspectRatio = originalWidth / originalHeight
height = int(width * aspectRatio)
image = cv2.resize(image, (height, width))

# Run an edge detector on the image and find its straight lines
result = cv2.Canny(image, 100, 300, 3)
cdst = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
lines = cv2.HoughLines(result, 1, math.pi/180, 300, srn=0, stn=0)

# Toss out lines that aren't horizontal enough, add the rest to sortedLines
sortedLines = []
for i in range(0,len(lines)):
    if (abs(lines[i,0,1] - (math.pi / 2)) < 0.09):
        sortedLines.append((lines[i,0,0], lines[i,0,1]))
# Sort the acceptable lines
sortedLines = sorted(sortedLines, key=itemgetter(0))

deletions = []
linePoints = []
distances = []

# Throw out lines if the distance between them is too small, to remove duplicate line detection
for i in range(0, len(sortedLines) - 1):
    if sortedLines[i + 1][0] - sortedLines[i][0] < 3:
        deletions.append(sortedLines[i + 1])
sortedLines = [x for x in sortedLines if x not in deletions]

# Now that most bad lines have been thrown out, find the distance between each line
for i in range(0, len(sortedLines) - 1):
    distances.append(int(sortedLines[i + 1][0] - sortedLines[i][0]))

# Calculate the median distance between lines to find the distance between staves
median = statistics.median(distances)

# Remove lines that aren't part of the staves:
improvement = True 
while (improvement == True):
    length = len(sortedLines)
    sortedLines = deleteLines(sortedLines, distances, median)
    if len(sortedLines) == length:
        improvement = False

lines = []

for line in sortedLines:
    rho = line[0]
    theta = line[1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho 
    y0 = b * rho 
    pt1 = (int(round(x0 + 5000*(-b))), int(round(y0 + 5000*(a))))
    pt2 = (int(round(x0 - 5000*(-b))), int(round(y0 - 5000*(a))))
    # TODO: transform lines if skewed  
    lines.append(pt1[1])
    #cv2.line(image, pt1, pt2, (125,0,125), 1)
    
cv2.namedWindow("detected lines", cv2.WINDOW_NORMAL)
cv2.imshow('detected lines', image)
cv2.resizeWindow('detected lines', 700,700)
keyPress = cv2.waitKey(0)

staffImages = []
staffLines = []
for i in range(0, len(lines), 5):
    if i + 4 < len(lines):
        img = getStaffImage(image, lines[i:i+5], median)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        staffImages.append(img)
        baseLine = lines[i]
        for j in range (i, i + 5):
            lines[j] = int(median * 2 + lines[j] - baseLine)


for i in range(0, len(lines), 5):
    if (i + 4 < len(lines)):
        staffLines.append(lines[i:i+5])


HALF = [cv2.imread('images/templates/half_1.png', 0), cv2.imread('images/templates/half_2.png', 0), cv2.imread('images/templates/half_3.png', 0), cv2.imread('images/templates/half_4.png', 0)]

RESTS = [cv2.imread('images/templates/hrest.png', 0), cv2.imread('images/templates/qrest.png', 0)]

QUARTER = [cv2.imread('images/templates/quarter_1.png', 0), cv2.imread('images/templates/quarter_3.png', 0), cv2.imread('images/templates/quarter_2.png', 0), cv2.imread('images/templates/quarter_4.png', 0)]

SHARP = [cv2.imread('images/templates/sharp.png', 0)]

WHOLE = [cv2.imread('images/templates/whole.png', 0), cv2.imread('images/templates/whole2.png', 0), cv2.imread('images/templates/whole3.png', 0)]

symbolLocations = []
quarterAndEighths = []
halves = []
wholes = []
sharps = []
rests = []

quarterAndEighths = getNotes(staffImages, QUARTER, False, False, .79, cv2.TM_CCOEFF_NORMED)
quarterRests = getNotes(staffImages, [RESTS[1]], False, True, .75, cv2.TM_CCOEFF_NORMED)
print("Quarter rests", quarterRests)
w, h = QUARTER[0].shape[::-1]
# for staff, img in zip(quarterAndEighths, staffImages):
#     for note in staff:
#         point = note[0]
#         cv2.rectangle(img, point, (point[0] + w, point[1] + h), (0,0,255), 2)




# tempQuarters = sorted(tempQuarters)
# w, h = quarter[0].shape[::-1]
# lastX = -1 * w
# for note in tempQuarters:
#     if note[0] > lastX + w:
#         quarterAndEighths.append(pt)


blackPerCol = []
for staffImage in staffImages:
    blackPerCol.append(getBlackRows(staffImage))

hRests = []
for bpc, qe, qr in zip(blackPerCol, quarterAndEighths, quarterRests):
    whole, half, hrest = analyzeBlack(bpc, qe, qr, len(QUARTER[0]))
    wholes.append(whole)
    halves.append(half)
    hRests.append(hrest)


h, w = staffImages[0].shape 


h, w = QUARTER[0].shape    

hrests = []
for hrest, i in zip(hRests, list(range(0, len(hRests)))):
    hrest = [(h + i * 3000, -1, 2) for h in hrest]
    hrests.extend(hrest)

qrests = []
for qrest, i in zip(quarterRests, list(range(0, len(quarterRests)))):
    qrest = [(q[0][0] + i * 3000, -1, 4) for q in qrest]
    qrests.extend(qrest)


noteMatches = []
for staff, notes, img, i in zip(staffLines, quarterAndEighths, staffImages, list(range(0, len(staffLines)))):
    matches = []
    # staff lines:
    F2 = staff[0]
    D2 = staff[1]
    B2 = staff[2]
    G1 = staff[3]
    E1 = staff[4]
    # spaces between:
    E2 = int(round(staff[0] + .5 * median))
    C2 = int(round(staff[1] + .5 * median))
    A1 = int(round(staff[2] + .5 * median))
    F1 = int(round(staff[3] + .5 * median))
    # above staff:
    G2 = int(round(staff[0] - .5 * median))
    A2 = int(round(staff[0] - median))
    # below:
    D1 = int(round(staff[4] + .5 * median))
    linePositions = [D1, E1, F1, G1, A1, B2, C2, D2, E2, F2, G2, A2]
    for note, val in notes:
        segments = img[0:len(img[0]), note[0]:note[0]+w]
        black = getBlackCols(segments)
        rows, cols = segments.shape 
        black = [l / max(black) for l in black]
        candidate, eighth = getNotePosition(black, True)
        distances = [abs(candidate - x) for x in linePositions]
        print(eighth)
        if eighth:
            noteMatches.append((note[0] + i * 3000, distances.index(min(distances)), 8))
        else:
            noteMatches.append((note[0] + i * 3000, distances.index(min(distances)), 4))


halfMatches = []

for staff, notes, img, i in zip(staffLines, halves, staffImages, list(range(0, len(halves)))):
    matches = []
    # staff lines:
    F2 = staff[0]
    D2 = staff[1]
    B2 = staff[2]
    G1 = staff[3]
    E1 = staff[4]
    # spaces between:
    E2 = int(round(staff[0] + .5 * median))
    C2 = int(round(staff[1] + .5 * median))
    A1 = int(round(staff[2] + .5 * median))
    F1 = int(round(staff[3] + .5 * median))
    # above staff:
    G2 = int(round(staff[0] - .5 * median))
    A2 = int(round(staff[0] - median))
    # below:
    D1 = int(round(staff[4] + .5 * median))
    linePositions = [D1, E1, F1, G1, A1, B2, C2, D2, E2, F2, G2, A2]
    for note, val in notes:
        segments = img[0:len(img[0]), note:note+w]
        black = getBlackCols(segments)
        rows, cols = segments.shape 
        # cv2.rectangle(segments, (0, 0), (cols, rows), (255,255,255), -1)
        # for i in range(0, rows):
            # cv2.line(segments, (0, i), (black[i], i), (0,0,0), 1)
        # cv2.imshow('a', segments)
        # cv2.waitKey(0)
        black = [l / max(black) for l in black]
        candidate, _ = getNotePosition(black, False)
        distances = [abs(candidate - x) for x in linePositions]
        halfMatches.append((note + i * 3000, distances.index(min(distances)), 2))

wholeMatches = []

for staff, notes, img, i in zip(staffLines, wholes, staffImages, list(range(0, len(wholes)))):
    matches = []
    # staff lines:
    F2 = staff[0]
    D2 = staff[1]
    B2 = staff[2]
    G1 = staff[3]
    E1 = staff[4]
    # spaces between:
    E2 = int(round(staff[0] + .5 * median))
    C2 = int(round(staff[1] + .5 * median))
    A1 = int(round(staff[2] + .5 * median))
    F1 = int(round(staff[3] + .5 * median))
    # above staff:
    G2 = int(round(staff[0] - .5 * median))
    A2 = int(round(staff[0] - median))
    # below:
    D1 = int(round(staff[4] + .5 * median))
    linePositions = [D1, E1, F1, G1, A1, B2, C2, D2, E2, F2, G2, A2]
    for note, val in notes:
        segments = img[0:len(img[0]), note:note+w]
        black = getBlackCols(segments)
        rows, cols = segments.shape 
        # cv2.rectangle(segments, (0, 0), (cols, rows), (255,255,255), -1)
        # for i in range(0, rows):
            # cv2.line(segments, (0, i), (black[i], i), (0,0,0), 1)
        # cv2.imshow('a', segments)
        # cv2.waitKey(0)
        black = [l / max(black) for l in black]
        candidate, _ = getNotePosition(black, False)
        distances = [abs(candidate - x) for x in linePositions]
        wholeMatches.append((note + i * 3000, distances.index(min(distances)), 1))

noteMatches.extend(wholeMatches)
noteMatches.extend(halfMatches)
noteMatches.extend(hrests)
noteMatches.extend(qrests)

noteMatches = sorted(noteMatches)

noSharpDict = {
    -1: 0,
    0: 62,
    1: 64,
    2: 65,
    3: 67,
    4: 69,
    5: 71,
    6: 72,
    7: 74,
    8: 76,
    9: 77,
    10: 79,
    11: 81
}

sharpDict = {
    -1: 0,
    0: 62,
    1: 64,
    2: 66,
    3: 67,
    4: 69,
    5: 71,
    6: 72,
    7: 74,
    8: 76,
    9: 78,
    10: 79,
    11: 81
}

noSharps = True

if (len(sys.argv) > 2):
    noSharps = False
mf = MIDIFile(1, adjust_origin="none")  # only 1 track
track = 0  # the only track
time = 0  # start at the beginning
mf.addTrackName(track, time, "Piano")
mf.addTempo(track, time, 40)
time = 0
channel = 0
volume = 100
for pos, note, value in noteMatches:
    pitch = 0
    if noSharps:
        pitch = noSharpDict.get(note)
    else:
        pitch = sharpDict.get(note)
    
    if pitch != 0:
        duration = 1 / value  # 1 beat long
        mf.addNote(track, channel, pitch, time, duration, volume)
    
    time = time + 1 / value

with open("output.mid", 'wb') as outf:
    mf.writeFile(outf)
call(["fluidsynth", "-a",  "alsa" ,"-m", "alsa_seq", "-l", "-i", "/usr/share/soundfonts/FluidR3_GM.sf2", "output.mid"])
