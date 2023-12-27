import cv2 as cv
import math
import numpy as np
import timeit
import pieceMover

class PuzzlePiece:
    number = 0
    basePath = "/home/mrmuffinswigle/Desktop/programming/python/eget/Puzzle/"

    def __init__(self,pieceImg): #top left point på skärmen
        self.topLeftPoint =  () #tuple, xy
        self.pieceImg = pieceImg #np.array
        self.imgHeight, self.imgWidth = self.pieceImg.shape
        self.bgrImg = []
        self.center = ()
        self.width = None
        self.height = None
        self.sideType = [None,None,None,None]
        self.sideMatchings = [None,None,None,None]
        self.nbr = PuzzlePiece.number
        PuzzlePiece.number +=1

    def __eq__(self,other):
        if other != None and self.nbr == other.nbr:
            return True
        else:
            return False
        
    def __hash__(self):
        return self.nbr
        
    def getImg(self):
        return self.pieceImg
    
    def getHeight(self):
        return self.imgHeight
    
    def getWidth(self):
        return self.imgWidth
    
    def sideMatch(self,puzzlePiece,targetSide): #targetSide is int
        if self.sideType[(targetSide+2)%4] + puzzlePiece.sideType[targetSide] == 0:
            return True
        else:
            return False
        
    def getSideMatch(self,side):
        return -self.sideType[side]
        
    def showPiece(self):
        cv.namedWindow("es")
        cv.moveWindow("es",400,400)

        cv.imshow("es",self.pieceImg)
        k = cv.waitKey(0) & 0xFF
        cv.destroyAllWindows()

    def findCenter(self): #bitens dimensioner och center
        img = self.pieceImg
        contour,hierarchy = cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

        hull = cv.convexHull(contour[0])
        pts = self.squareFinder(hull,1000,1000)

        width = int(round((abs(pts[0][0][0]-pts[1][0][0]) + abs(pts[2][0][0]-pts[3][0][0]))/2))
        height = int(round((abs(pts[1][0][1]-pts[2][0][1]) + abs(pts[0][0][1]-pts[3][0][1]))/2))

        center = (pts[0][0][0]+width//2,pts[0][0][1]-height//2) #x,y
        return center,width,height,pts

    def squareFinder(self,points,pieceWidth,pieceHeight): #lägg till minsta pusselbitsdimensionen för urvalskriterie
        pointList = []
        decimals = 5
        angleOffset = 5
        minError = 1000000
        
        for p1 in points:
            for p2 in points:
                if (p1 == p2).all():
                    continue
                v12 = (p2[0][0]-p1[0][0],p2[0][1]-p1[0][1])
                ang1 = 180/math.pi*math.acos(round(np.dot(v12,[1,0])/(np.linalg.norm(v12)*np.linalg.norm([1,0])),decimals))
                if ang1 > angleOffset:
                    continue

                for p3 in points:
                    if (p1 == p3).all() or (p2 == p3).all():
                        continue
                    v23 = (p3[0][0]-p2[0][0],p3[0][1]-p2[0][1])
                    ang2 = 180/math.pi*math.acos(round(np.dot([0,-1],v23)/(np.linalg.norm([0,-1])*np.linalg.norm(v23)),decimals))
                    if ang2 > angleOffset:
                        continue

                    for p4 in points:
                        if (p1 == p4).all() or (p2 == p4).all() or (p3 == p4).all():
                            continue
                        v34 = (p4[0][0]-p3[0][0],p4[0][1]-p3[0][1])
                        v41 = (p1[0][0]-p4[0][0],p1[0][1]-p4[0][1])

                        ang3 = 180/math.pi*math.acos(round(np.dot([-1,0],v34)/(np.linalg.norm([-1,0])*np.linalg.norm(v34)),decimals))
                        ang4 = 180/math.pi*math.acos(round(np.dot([0,1],v41)/(np.linalg.norm([0,1])*np.linalg.norm(v41)),decimals))
                        if ang3 > angleOffset:
                            continue
                        if ang4 > angleOffset:
                            continue
                        
                        lengthError = abs(np.linalg.norm(v12)-pieceWidth) + abs(np.linalg.norm(v23)-pieceHeight) + abs(np.linalg.norm(v34)-pieceWidth) + abs(np.linalg.norm(v41) - pieceHeight)
                        if lengthError < minError:
                            pointList = [p1,p2,p3,p4]
                            minError = lengthError

        return pointList
    
    def showSegment(self,segment): #från piecesegmenter
        imTest = np.zeros((500,500))
        cv.drawContours(imTest,self.convertPointListToContour(segment),-1,255,1)
        cv.namedWindow("es")
        cv.moveWindow("es",400,400)

        cv.imshow("es",imTest)
        k = cv.waitKey(0) & 0xFF
        cv.destroyAllWindows()

    def pieceSegmenter(self,height = 1000,width = 1000, targetWidth = 200):
        imC = np.copy(self.pieceImg)
        contours, hierarchy = cv.findContours(imC,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

        hull = cv.convexHull(contours[0])
        cornerPoints = self.squareFinder(hull,height,width)

        initialSeg = []
        first = np.where(np.equal(contours[0],cornerPoints[0]).all(2))[0][0]
        second = np.where(np.equal(contours[0],cornerPoints[1]).all(2))[0][0]
        third = np.where(np.equal(contours[0],cornerPoints[2]).all(2))[0][0]
        fourth = np.where(np.equal(contours[0],cornerPoints[3]).all(2))[0][0]

        initialSeg.append(contours[0][second:third]) #konstig ordning eftersom konturerna börjar nedifrån och moturs
        initialSeg.append(np.concatenate((contours[0][third:],contours[0][:fourth+1])))
        initialSeg.append(contours[0][fourth:first])
        initialSeg.append(contours[0][first:second])

        processingSeg = [[],[],[],[]]
        rotatedSeg = [[],[],[],[]]
        rotMatrix = [[0,-1],[1,0]]
        shiftedSeg = [[],[],[],[]]
        scaledSeg = [[],[],[],[]]
        finalSegments = [[],[],[],[]]

        for i in range(4):
            for j in range(len(initialSeg[i])):
                processingSeg[i].append(initialSeg[i][j][0])
        
            for j in range(len(processingSeg[i])):
                point = processingSeg[i][j]
                if i == 0:
                    for k in range(3):
                        point = np.matmul(rotMatrix,point)
                elif i == 1:
                    for k in range(0):
                        point = np.matmul(rotMatrix,point)
                elif i == 2:
                    for k in range(1):
                        point = np.matmul(rotMatrix,point)
                elif i == 3:
                    for k in range(2):
                        point = np.matmul(rotMatrix,point)

                rotatedSeg[i].append(point)

            smallestX = 100000
            smallestY = 100000
            for j in range(len(rotatedSeg[i])):
                if rotatedSeg[i][j][0] < smallestX:
                    smallestX = rotatedSeg[i][j][0]
                if rotatedSeg[i][j][1] < smallestY:
                    smallestY = rotatedSeg[i][j][1]
        
            for j in range(len(rotatedSeg[i])):
                point = rotatedSeg[i][j]
                point = [point[0]-smallestX,point[1]-smallestY]
                shiftedSeg[i].append(point)

            scaledSeg[i] = self.rescalePointList(shiftedSeg[i],targetWidth)

        return scaledSeg
    
    def rescalePointList(self,pointList,targetWidth):
        largest = 0
        scaledPointList = []
        for j in range(len(pointList)):
            if pointList[j][0] > largest:
                largest = pointList[j][0]

        scalingFactor = targetWidth/largest
        for j in range(len(pointList)):
            point = np.multiply(pointList[j],scalingFactor)
            point = np.round(point).astype(int)
            scaledPointList.append(point)
        return scaledPointList

    def convertPointListToContour(self,list):
        contourList = []
        for i in range(len(list)):
            contourList.append([list[i]])

        return np.array(contourList)
    
    def savePointList(self,pointList,path):
        with open(path,"w",encoding="utf-8") as f:
            for i in pointList:
                f.write(str(i) + "\n")

    def loadPointList(self,path):
        pointList = []
        with open(path,"r",encoding="utf-8") as f:
            endOfFile = False
            while endOfFile == False:
                s = f.readline()
                if "\n" not in s:
                    endOfFile = True
                    break
                nbrs = s.strip("[]\n").split(" ")
                newNbrs = []
                for i in nbrs:
                    if i != "":
                        newNbrs.append(i)
                pointList.append([int(newNbrs[0]),int(newNbrs[1])])

        return pointList
    
    def showStandardSides(self):
        sideTypes = np.linspace(-8,8,17)
        path = self.basePath + "sideTypeStandards/"
        for s in sideTypes:
            print(s)
            segment = self.loadPointList(path + str(int(s)))
            self.showSegment(segment)
    
    def determineSegmentSideType(self,pointList):
        basePath = self.basePath + "sideTypeStandards/"
        sideTypes = np.linspace(-8,8,17)
        matches = []

        for j in sideTypes:
            sideCnt = self.convertPointListToContour(self.loadPointList(basePath + str(int(j))))
            distanceSq = 0
            for i in pointList:
                x,y = i
                c = (int(x),int(y))
                distanceSq += (cv.pointPolygonTest(sideCnt,c,True))**2

            i = 0
            while i < len(matches) and matches[i][1] < distanceSq:
                i +=1
            matches.insert(i,[int(j),distanceSq])
                
        return matches
    
    def determineSides(self):
        segments = self.pieceSegmenter(targetWidth=200)
        sideTypes = []
        for i in range(len(segments)):
            side = self.determineSegmentSideType(segments[i])
            sideTypes.append(side)
        return sideTypes




    

    



















        # def determineSides(self,maxSqDifference): #används senare
    
    
    
    
    
    
    #     segments = self.pieceSegmenter(targetWidth=200)
    #     sideTypes = []
    #     for i in range(len(segments)):
    #         side = []
    #         matches = self.determineSegmentSideType(segments[i])
    #         for j in range(len(matches)):
    #             if matches[j][1] < maxSqDifference:
    #                 side.append(matches[j][0])
    #         sideTypes.append(side)
    #     return sideTypes

    # def determineSegmentSideType(self,pointList):
    #     basePath = self.basePath + "sideTypeStandards/"
    #     sideTypes = np.linspace(-8,8,17)
    #     currType = None
    #     lastMatch = 1000000

    #     for j in sideTypes:
    #         sideCnt = self.convertPointListToContour(self.loadPointList(basePath + str(int(j))))
    #         match = cv.matchShapes(sideCnt,self.convertPointListToContour(pointList),1,0.0)
    #         if match < lastMatch:
    #             lastMatch = match
    #             currType = int(j)
                
    #     return currType
    
    # def determineSegmentSideType(self,pointList):
    #     basePath = self.basePath + "sideTypeStandards/"
    #     sideTypes = np.linspace(-8,8,17)
    #     currType = None
    #     lastDist = 100000000

    #     for j in sideTypes:
    #         sideCnt = self.convertPointListToContour(self.loadPointList(basePath + str(int(j))))
    #         distanceSq = 0
    #         for i in pointList:
    #             x,y = i
    #             c = (int(x),int(y))
    #             distanceSq += (cv.pointPolygonTest(sideCnt,c,True))**2
    #         if distanceSq < lastDist:
    #             lastDist = distanceSq
    #             currType = int(j)
                
    #     return currType

    # #height first
        # h = self.height
        # w = self.width
        # distancePercent = 0.2
        # vDistance = math.floor(h*distancePercent)
        # hDistance = math.floor(w*distancePercent)

        # pointT = 0
        # pointB = h
        # pointL = 0
        # pointR = w

        # #höjden
        # täljare = self.pieceImg[0:vDistance,0:w] 
        # TopP = cv.countNonZero(täljare)/(vDistance*w)
        # if TopP < 0.35:
        #     pointT += 20
        #     print("top is outie")

        # #botten
        # täljare = self.pieceImg[h-vDistance:h,0:w] 
        # botP = cv.countNonZero(täljare)/(vDistance*w)
        # if botP < 0.35:
        #     pointB -= 20
        #     print("down is outie")

        # #vänster
        # täljare = self.pieceImg[0:h,0:hDistance] 
        # leftP = cv.countNonZero(täljare)/(vDistance*h)
        # if leftP < 0.35:
        #     pointL += 20
        #     print("left is outie")

        # #höger
        # täljare = self.pieceImg[0:h,w-hDistance:w] 
        # rightP = cv.countNonZero(täljare)/(vDistance*h)
        # if rightP < 0.35:
        #     pointR -= 20
        #     print("right is outie")
            
        
        # centre = ((pointL + pointR)//2,(pointT + pointB)//2) #x,y

    # def cropSides(self, min, max):
    #     centre,w,h = self.getCentreWithLineCounting(min, max)
    #     img = self.pieceImg
    #     y = centre[1]
    #     x = centre[0]

    #     #print(w//2)
    #     segment = math.floor((max-min)*1.3/2)

    #     # leftCrop = img[y-h//2:y+h//2,0:x-extra]
    #     # rightCrop = img[y-h//2:y+h//2,x+extra:]
    #     # topCrop = img[0:y-extra,x-w//2:x+w//2]
    #     # botCrop = img[y+extra:,x-w//2:x+w//2] #sätt dem som fix storlek
    #     leftCrop = img[y-segment:y+segment,0:segment]
    #     rightCrop = img[y-segment:y+segment,self.imgWidth-segment:]
    #     topCrop = img[0:segment,x-segment:x+segment]
    #     botCrop = img[self.imgHeight-segment:,x-segment:x+segment] #sätt dem som fix storlek

    #     cropArray = [rightCrop,topCrop,leftCrop,botCrop] #som enhetscirkeln

    #     img = self.bgrImg
    #     leftCrop = img[y-segment:y+segment,0:segment]
    #     rightCrop = img[y-segment:y+segment,self.imgWidth-segment:]
    #     topCrop = img[0:segment,x-segment:x+segment]
    #     botCrop = img[self.imgHeight-segment:,x-segment:x+segment] #sätt dem som fix storlek

    #     bgrArray = [rightCrop,topCrop,leftCrop,botCrop]
    #     for i in range(4):
    #         if i == 0:
    #             cropArray[i] = cv.rotate(cropArray[i],cv.ROTATE_90_COUNTERCLOCKWISE)
    #             bgrArray[i] = cv.rotate(bgrArray[i],cv.ROTATE_90_COUNTERCLOCKWISE)
    #         elif i == 2:
    #             cropArray[i] = cv.rotate(cropArray[i],cv.ROTATE_90_CLOCKWISE)
    #             bgrArray[i] = cv.rotate(bgrArray[i],cv.ROTATE_90_CLOCKWISE)
    #         elif i == 3:
    #             cropArray[i] = cv.rotate(cropArray[i],cv.ROTATE_180)
    #             bgrArray[i] = cv.rotate(bgrArray[i],cv.ROTATE_180)

    #     for i in range(len(cropArray)):
    #         cropArray[i] = cv.resize(cropArray[i],(200,100),interpolation=cv.INTER_LINEAR)
    #         cropArray[i] = cv.cvtColor(cropArray[i],cv.COLOR_GRAY2RGB)
    #         bgrArray[i] = cv.resize(bgrArray[i],(200,100),interpolation=cv.INTER_LINEAR)

    #     return cropArray, bgrArray


        # def getCentreWithLineCounting(self,min,max): #tar första och sista linjen vars pixelsumma överstiger ett visst värde, !!!måste förmodligen göra om denna med convex hull istället
    #     img = self.pieceImg
        
    #     bestHorOne = 0
    #     bestHorTwo = 0

    #     threshold = min//2 #måste ändra till procentsats
    #     satisfied = False

    #     for h in range(self.imgHeight):
    #         line = img[h:h+1,0:self.imgWidth]
    #         value = cv.countNonZero(line)
    #         if value > threshold and satisfied == False:
    #             bestHorOne = h
    #             satisfied = True
    #         if value > threshold:
    #             bestHorTwo = h

    #     bestVerOne = 0
    #     bestVerTwo = 0

    #     satisfied = False

    #     for w in range(self.imgWidth):
    #         line = img[0:self.imgHeight,w:w+1]
    #         value = cv.countNonZero(line)
    #         if value > threshold and satisfied == False:
    #             bestVerOne = w
    #             satisfied = True
    #         if value > threshold:
    #             bestVerTwo = w
        
    #     width = bestVerTwo-bestVerOne
    #     height = bestHorTwo-bestHorOne
        
    #     imgCentre = ((bestVerTwo+bestVerOne)//2,(bestHorTwo+bestHorOne)//2)
    #     x,y = self.topLeftPoint
    #     self.centre = x+(bestVerTwo+bestVerOne)//2,y +(bestHorTwo+bestHorOne)//2
    #     self.diff = self.centre[0] - self.topLeftPoint[0],self.centre[1] - self.topLeftPoint[1]
    #     return imgCentre,width,height