import sys
from PIL import Image,ImageGrab, ImageDraw
import pyautogui as pg
import numpy as np
import time
import cv2 as cv
#from matplotlib import pyplot as plt
import PieceCollection
import PuzzlePiece
import PileHandler
import Pile
import math
import copy
import pieceMover
import os
import logging


#puzzlepiece class with attributes: img of self, boundary pos, central position and sides which will it will get itself
#imagescan class that outputs an array of puzzlepieces
#puzzlesolver class that solves the puzzle and outputs new puzzlepiece position


class imagescanner:
    screenWidth = 1920
    screenHeight = 1080
    topBarHeight = 36 #36 för scale100, 10 för scale20
    savedImagesNbr = 0
    basePath = "/home/mrmuffinswigle/Desktop/programming/python/eget/Puzzle/"

    def deleteFolderContents(self, folderDir) -> None:
        for filename in os.listdir(folderDir):
            os.remove(os.path.join(folderDir,filename))
    
    def saveBackgroundScreenshots(self,scale) -> None:
        imagePathBase = self.basePath + "puzzleimages/pythonimgbackground"
        theme = 0
        seam = 1000
        for i in range(12):
            self.changeBackground(theme,scale)

            imgLeft = ImageGrab.grab((0,self.topBarHeight,seam,1080))
            pieceMover.pieceMover.moveTo(1830,525)
            pieceMover.pieceMover.click()
            pieceMover.pieceMover.moveTo(10,536) #drag to left
            imgRight = ImageGrab.grab((seam,self.topBarHeight,1920,1080))
            pieceMover.pieceMover.moveTo(1830,525) #drag to right
            pieceMover.pieceMover.click()

            img = Image.new("RGB",(1920,1080-self.topBarHeight))
            img.paste(imgLeft,(0,0))
            img.paste(imgRight,(seam,0))
            img.save(imagePathBase + str(theme) + ".png")
            theme = theme+1

    def saveBackgroundScreenshot(self,theme) -> None:
        imagePathBase = self.basePath + "puzzleimages/pythonimgbackground"
        seam = 1000
        imgLeft = ImageGrab.grab((0,self.topBarHeight,seam,1080))
        print("waiting for input")
        input()
        print("continuing")
        imgRight = ImageGrab.grab((seam,self.topBarHeight,1920,1080))
        
        img = Image.new("RGB",(1920,1080-self.topBarHeight))
        img.paste(imgLeft,(0,0))
        img.paste(imgRight,(seam,0))
        img.save(imagePathBase + str(theme) + ".png")

    def changeBackground(self,themeNbr,scale) -> None:
        
        threeBars100 = (38,24)
        threeBars20 = (8,4)
        modify100 = (112,55)
        modify20 = (18,14)
        theme100 = (935,537)
        theme20 = (953,539)
        ok100 = (1102,524)
        ok20 = (994,539)

        themeStart100 = (54,85)
        themeStart20 = (10,17)
        themeOffsetY100 = 50
        themeOffsetY20 = 20
        themeOffsetX100 = 70
        themeOffsetX20 = 20

        if scale == 20:
            threeBars = threeBars20
            modify = modify20
            theme = theme20
            ok = ok20
            themeStart = themeStart20
            themeOffsetY = themeOffsetY20
            themeOffsetX = themeOffsetX20

        elif scale == 100:
            threeBars = threeBars100
            modify = modify100
            theme = theme100
            ok = ok100
            themeStart = themeStart100
            themeOffsetY = themeOffsetY100
            themeOffsetX = themeOffsetX100

        pieceMover.pieceMover.moveTo(threeBars[0],threeBars[1]) #three bars
        time.sleep(0.5)
        pieceMover.pieceMover.moveTo(modify[0],modify[1]) #modify
        pieceMover.pieceMover.click()
        pieceMover.pieceMover.moveTo(theme[0],theme[1]) #theme
        pieceMover.pieceMover.click()

        pieceMover.pieceMover.moveTo(themeStart[0]+(themeNbr%3)*themeOffsetX,themeStart[1]+(themeNbr//3)*themeOffsetY)
        pieceMover.pieceMover.click()
        pieceMover.pieceMover.moveTo(ok[0],ok[1]) #ok
        pieceMover.pieceMover.click()
        time.sleep(0.5)

    def savePuzzleScreenshot(self,nbr) -> None:
        imagePathBase = self.basePath + "puzzleimages/pythonimgPuzzle"
        self.changeBackground(nbr)
        img = ImageGrab.grab((0,self.topBarHeight,1920,1080))
        img.save(imagePathBase + str(nbr) + ".png")

    def getFieldScreenshot(self):
        img = np.array(ImageGrab.grab((0,self.topBarHeight,1920,1080)))
        return cv.cvtColor(img,cv.COLOR_RGB2BGR)

    def savePuzzleScreenshots(self):
        imagePathBase = self.basePath + "puzzleimages/pythonimgPuzzle"
        theme = 0
        for i in range(12):
            self.changeBackground(theme)

            img = ImageGrab.grab((0,self.topBarHeight,1920,1080))

            img.save(imagePathBase + str(theme) + ".png")
            theme = theme+1

    def getImg(self,nbr):
        return cv.imread(self.basePath + "puzzleimages/pythonimgPuzzle" + str(nbr) + ".png")
    
    def getDimensions(self,nbr):
        return cv.imread(self.basePath + "puzzleimages/pythonimgPuzzle" + str(nbr) + ".png").shape

    def getBackground(self,nbr):
        return cv.imread(self.basePath + "puzzleimages/pythonimgbackground" + str(nbr) + ".png")
    
    def filterImage3(self,nbr,img):
        background = self.getBackground(nbr)

        colorPieces = cv.bitwise_xor(img,background)
        bwPieces = cv.cvtColor(colorPieces,cv.COLOR_BGR2GRAY)
        ret, threshPieces = cv.threshold(bwPieces,1,255,cv.THRESH_BINARY)
        #kernel = np.ones((3,3),np.uint8)
        #finalPieces = cv.morphologyEx(threshPieces,cv.MORPH_OPEN,kernel,iterations = 1)
        #finalPieces = cv.morphologyEx(finalPieces,cv.MORPH_CLOSE,kernel,iterations=1)

        return threshPieces

    def getClearingDimensions(self,nbr: int) -> (int,int):
        img = self.filterImage3(nbr,self.getFieldScreenshot())
        center = (self.screenWidth//2, (self.screenHeight-self.topBarHeight)//2)
        
        chunk = []
        extraUp = 10
        while np.count_nonzero(chunk) == 0: #ger en rektangel med höjd extraUp och bredd 2*extraUp
            extraUp += 1
            chunk = img[center[1]-extraUp:center[1],center[0]-extraUp//2:center[0]+extraUp//2]
        extraUp -= 1

        chunk = []
        extraDown = 10
        while np.count_nonzero(chunk) == 0:
            extraDown += 1
            chunk = img[center[1]:center[1]+extraDown,center[0]-extraDown//2:center[0]+extraDown//2]
        extraDown -= 1

        return min(extraUp,extraDown,(self.screenHeight-self.topBarHeight)//2-20) #x,y,w,h
    
    def photographEveryPiece(self,nbr: int) -> PieceCollection.PieceCollection:
        """Goes over every piece on the field and fits them one by one into
        the photosquare before moving the pieces into a pile. Returns a collection with
        the pieces.
        
        Args:
            nbr (int): The number of the background that the photos should be filtered against.

        Returns:
            Collection.Collection: A collection with pieces added to it's attribute pieceList
            and with pileDim dimensions used by the method.        
        """

        collection = PieceCollection.PieceCollection()
        side = self.getClearingDimensions(nbr)
        photoDim = ((self.screenHeight-self.topBarHeight)//2-side,(self.screenHeight-self.topBarHeight)//2,self.screenWidth//2-side//2,self.screenWidth//2+side//2)#y,y+h,x,x+w
        pileDim = ((self.screenHeight-self.topBarHeight)//2,(self.screenHeight-self.topBarHeight)//2+side,self.screenWidth//2-side//2,self.screenWidth//2+side//2)#kan bli problem senare (y,y+h,x,x+w)
        cropDim = ((self.screenHeight-self.topBarHeight)//2-side,(self.screenHeight-self.topBarHeight)//2+side,self.screenWidth//2-side//2,self.screenWidth//2+side//2)

        backgroundPart = self.getBackground(nbr)[cropDim[0]:cropDim[1],cropDim[2]:cropDim[3]]

        fieldAll = self.getFieldScreenshot()
        fieldAll[cropDim[0]:cropDim[1],cropDim[2]:cropDim[3]] = backgroundPart
        fieldFiltered = self.filterImage3(nbr,fieldAll)

        locations = np.where(fieldFiltered>0)
        locationBlackout = []

        folderDir = self.basePath + "puzzleimages/debug/"
        tempNbr = 0 ####
        for filename in os.listdir(folderDir):
            os.remove(os.path.join(folderDir,filename))

        while len(locations[0]) > 0:
            pieceMover.pieceMover.moveTo(locations[1][0]+1,locations[0][0]+1+self.topBarHeight)
            pieceMover.pieceMover.click()
            pieceMover.pieceMover.moveTo(self.screenWidth//2,(self.screenHeight-self.topBarHeight)//2+self.topBarHeight-side*4//5)
            pieceMover.pieceMover.click()
            pieceMover.pieceMover.moveTo(100,0)

            pieceImg = self.getFieldScreenshot()
            pieceFiltered = self.filterImage3(nbr,pieceImg)[photoDim[0]:photoDim[1],photoDim[2]:photoDim[3]]

            if len(np.where(pieceFiltered>0)[0]) == 0:
                locationBlackout.append((locations[0][0],locations[1][0]))
            else:
                contours,_ = cv.findContours(pieceFiltered,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                x,y,w,h = cv.boundingRect(contours[0])

                tries = 0
                while (photoDim[3] == photoDim[2]+x+w or photoDim[0] == photoDim[0]+y or photoDim[1] == photoDim[0]+y+h or photoDim[2] == photoDim[2]+x):
                    moments = cv.moments(pieceFiltered)
                    cX = moments["m10"]//moments["m00"]
                    cY = moments["m01"]//moments["m00"]
                    pieceMover.pieceMover.moveTo(photoDim[2]+cX,photoDim[0]+cY)
                    pieceMover.pieceMover.click()
                    pieceMover.pieceMover.moveTo(self.screenWidth//2,(self.screenHeight-self.topBarHeight)//2+self.topBarHeight-side//2)
                    pieceMover.pieceMover.click()
                    pieceImg = self.getFieldScreenshot()
                    pieceFiltered = self.filterImage3(nbr,pieceImg)[photoDim[0]:photoDim[1],photoDim[2]:photoDim[3]]
                    contours,_ = cv.findContours(pieceFiltered,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                    x,y,w,h = cv.boundingRect(contours[0])
                    tries +=1
                    if tries > 4:
                        raise Exception("couldnt fit piece in area")

                part = pieceFiltered
                cv.imwrite(self.basePath + "puzzleimages/debug/pieceFiltered" + str(tempNbr)+ ".png", pieceFiltered)
                tempNbr +=1

                pieceFinal = pieceFiltered[y:y+h,x:x+w]
                colorImg = pieceImg[photoDim[0]:photoDim[1],photoDim[2]:photoDim[3]][y:y+h,x:x+w]
                piece = PuzzlePiece.PuzzlePiece(pieceFinal)
                piece.topLeftPoint = (x+self.screenWidth//2-side//2,y+self.screenHeight//2+self.topBarHeight-side)
                piece.bgrImg = colorImg

                center,w,h,_ = piece.findCenter()
                pieceMover.pieceMover.pieceMove(piece, piece.topLeftPoint[0]+center[0],piece.topLeftPoint[1]+center[1],self.screenWidth//2,self.screenHeight//2+self.topBarHeight+side//2)
                collection.addPiece(piece)

            fieldAll = self.getFieldScreenshot()
            fieldAll[cropDim[0]:cropDim[1],cropDim[2]:cropDim[3]] = backgroundPart
            fieldFiltered = self.filterImage3(nbr,fieldAll)
            for i in range(len(locationBlackout)):
                fieldFiltered[locationBlackout[i][0]][locationBlackout[i][1]] = 0
                
                locations = np.where(fieldFiltered>0)
        collection.pileDim = pileDim
        return collection

    def scale20(self) -> None: #start non fullscreen
        scale20 = (1696,1025)
        fullscreen20 = (1898,71)
        ok20 = (994,539)
        resetZoom = (1660,1022)

        pieceMover.pieceMover.moveTo(resetZoom[0],resetZoom[1])
        pieceMover.pieceMover.click()
        print("waitLoad")
        self.waitLoad()
        pieceMover.pieceMover.moveTo(scale20[0],scale20[1])
        pieceMover.pieceMover.click()
        self.waitLoad()
        pieceMover.pieceMover.moveTo(fullscreen20[0],fullscreen20[1])
        pieceMover.pieceMover.click()
        self.waitLoad()
        pieceMover.pieceMover.moveTo(ok20[0],ok20[1])
        pieceMover.pieceMover.click()
        time.sleep(0.1)

    def waitLoad(self) -> None:
        side = 300
        center = (960,553)

        centerImg1 = cv.cvtColor(self.getFieldScreenshot()[center[1]-side//2-self.topBarHeight:center[1]+side//2-self.topBarHeight,center[0]-side//2:center[0]+side//2], cv.COLOR_BGR2GRAY)
        time.sleep(0.1)
        centerImg2 = cv.cvtColor(self.getFieldScreenshot()[center[1]-side//2-self.topBarHeight:center[1]+side//2-self.topBarHeight,center[0]-side//2:center[0]+side//2], cv.COLOR_BGR2GRAY)
        while len(np.nonzero(cv.bitwise_xor(centerImg1,centerImg2))[0]) != 0:
            centerImg1 = centerImg2
            time.sleep(0.1)
            print("in loop")
            centerImg2 = cv.cvtColor(self.getFieldScreenshot()[center[1]-side//2-self.topBarHeight:center[1]+side//2-self.topBarHeight,center[0]-side//2:center[0]+side//2], cv.COLOR_BGR2GRAY)
        print("waitLoad finished")
        return
    
    def posInArea(self,area,coordinates): #area as (x,x+w,y,y+h), coordinates as (x,y)
        if (area[0]<coordinates[0]<area[1]) and (area[2]<=coordinates[1]<=area[3]):
            return True
        else:
            return False
        
    def areaOverlap(self,area1,area2): #area as (x,x+w,y,y+h)
        pts1 = []
        pts2 = []
        for i in range(2):
            for j in range(2):
                pts1.append((area1[i],area1[j+2]))
                pts2.append((area2[i],area2[j+2]))
        for i in range(4):
            if (self.posInArea(area2,pts1[i]) or self.posInArea(area1,pts2[i])):
                return True
        return False

    def spreadPiecesOut2(self,collection):
        print("spreading pieces out")
        collection.setMinPieceDimensions()
        extraRatio = 1.2
        pieceWidth,pieceHeight = round(collection.minWidth*extraRatio),round(collection.minHeight*extraRatio)
        puzzleStart = (0,0)
        puzzleClearingMargin = 50
        puzzleHeight,puzzleWidth = collection.puzzleDimensions()
        puzzleClearing = (puzzleStart[0],puzzleStart[0] +puzzleWidth*pieceWidth+puzzleClearingMargin,puzzleStart[1]+self.topBarHeight,puzzleStart[1]+pieceHeight*puzzleHeight+puzzleClearingMargin+self.topBarHeight) #x,x+w,y,y+h
        _, distributionW, _, distributionH = collection.getImgDimMinMax()

        pileHandler = PileHandler.PileHandler()

        pileDim = collection.pileDim
        newPileDim = (pileDim[2],pileDim[3],pileDim[0],pileDim[1]) #x, x+w, y, y+h

        pieceList = list(reversed(collection.pieces))
        amount = len(pieceList)

        yTopBound = self.topBarHeight #topBarHeight kommer vara inbakat då loopen fortsätter
        xLeftBound = 0
        while yTopBound+distributionH<self.screenHeight:
            #print("starting loop at:" + str((xLeftBound,yTopBound)))
            if xLeftBound+distributionW > self.screenWidth:
                xLeftBound = 0
                yTopBound = yTopBound + distributionH
                continue
            if self.areaOverlap((xLeftBound,xLeftBound+distributionW,yTopBound,yTopBound+distributionH),newPileDim):
                xLeftBound = newPileDim[1]
                continue
            if self.areaOverlap((xLeftBound,xLeftBound+distributionW,yTopBound,yTopBound+distributionH),puzzleClearing):
                xLeftBound = puzzleClearing[1]
                continue
            #print("added pile at:" + str((xLeftBound+distributionW//2,yTopBound + distributionH//2)))
            pile = Pile.Pile((xLeftBound+distributionW//2,yTopBound + distributionH//2))
            pileHandler.addPile(pile)
            xLeftBound = xLeftBound +distributionW

        while len(pieceList) > 0:
            piece = pieceList.pop(0)
            pileHandler.distributePiece(piece) #coords (x,y)
        collection.pHandler = pileHandler

    def initializeGrid(self,height,width):
        puzzleGrid = []
        for i in range(width):
            for j in range(height):
                if i == 0:
                    puzzleGrid.append([None])
                else:
                    puzzleGrid[j].append(None)
        return puzzleGrid

    def solvePuzzle(self,collection,nbr):
        self.deleteFolderContents(self.basePath + "puzzleimages/hornpunkter")
        height,width = collection.puzzleDimensions()
        _, maxWidth ,_ ,maxHeight = collection.getImgDimMinMax()
        collection.setMinPieceDimensions()
        pHandler = collection.pHandler

        puzzleGrid = self.initializeGrid(height,width)
        pieceList = copy.copy(collection.pieces) #för att flytta ner bitar efter hand
        remainingPieces = set(pieceList)
        puzzleStart = (20,self.topBarHeight+20)

        toleranceIncrease = 100
        baseTolerance = 200
        prevPiece = None

        matchNotFound = True
        tolerance = baseTolerance - toleranceIncrease
        triedCombinations = []
        side = True
        topPiece = None

        imgList = []
        while matchNotFound == True:
            tolerance += toleranceIncrease

            leftPiece = None
            edges = [1,2]
            availablePieces1 = remainingPieces
            firstMatches = collection.matchFinder(leftPiece,topPiece,edges,availablePieces1,tolerance)
            #print("potential firstPieceMatches found: " + str(len(firstMatches)))
            for f in firstMatches:
                if matchNotFound == False:
                    break
                edges = [1]
                leftPiece = f
                availablePieces = remainingPieces
                matches = collection.matchFinder(leftPiece,topPiece,edges,availablePieces,tolerance)
                #print("potential pieceMatches found: " + str(len(matches)))
                for p in matches:
                    if f == p:
                        #print("skipped same piece combination")
                        triedCombinations.append([f,p])
                        continue
                    if [f,p] in triedCombinations:
                        continue

                    pHandler.makePieceAvailable(p)

                    x,y = p.topLeftPoint[0] + p.center[0][0],p.topLeftPoint[1] + p.center[0][1]
                    if self.pieceTesterBasic(f,p,side,collection.minWidth,maxWidth,collection.minHeight,maxHeight,nbr):
                        #print("moving chunk")
                        time.sleep(1)
                        pHandler.removePiece(p)
                        puzzleGrid[0][0] = f
                        puzzleGrid[0][1] = p
                        self.moveGrid(puzzleGrid,puzzleStart[0]+f.center[0][0],self.topBarHeight+puzzleStart[1]+f.center[0][1])
                        remainingPieces.remove(p)
                        remainingPieces.remove(f)
                        pieceList.remove(p)
                        pieceList.remove(f)
                        prevPiece = p
                        # print("filling empty spot")
                        # self.pieceMove2(pieceList[0],x,y)
                        matchNotFound = False
                        break
                    else:
                        triedCombinations.append([f,p])
            #print("increasing tolerance to " + str(tolerance+toleranceIncrease))

        #print("finished TL corner")

        for i in range(height):
            for j in range(width):
                if i == 0 and j in [0,1]:
                    continue
                #print(i,j,height,width)
                time.sleep(0.5)
                matchNotFound = True
                tolerance = baseTolerance - toleranceIncrease
                triedPieces = set()
                topPiece = None
                leftPiece = None
                while matchNotFound == True:
                    tolerance += toleranceIncrease
                    edges = []
                    if i == 0:
                        topPiece = None
                    else:
                        topPiece = puzzleGrid[i-1][j]
                    if j == 0:
                        leftPiece = None
                    else:
                        leftPiece = puzzleGrid[i][j-1]

                    if i == 0:
                        edges.append(1)
                    elif i == height-1:
                        edges.append(3)

                    if j == 0:
                        edges.append(2)
                    elif j == width-1:
                        edges.append(0)

                    availablePieces = remainingPieces - triedPieces

                    if len(availablePieces) == 0:
                        raise Exception("No pieces match")

                    matches = collection.matchFinder(leftPiece,topPiece,edges,availablePieces,tolerance) #set                
                    #print("potential pieceMatches found: " + str(len(matches)))
                    if len(matches) != 0:
                        if j == 0:
                            side = False
                            prevPiece = puzzleGrid[i-1][j]
                        else:
                            side = True
                        for p in matches:
                            pHandler.makePieceAvailable(p)
                                    
                            x,y = p.topLeftPoint[0] + p.center[0][0],p.topLeftPoint[1] + p.center[0][1]
                            index = pieceList.index(p)
                            if self.pieceTesterWithContour(prevPiece,p,side,collection.minWidth,maxWidth,collection.minHeight,maxHeight,nbr):
                                #print(p.nbr)
                                pHandler.removePiece(p)
                                remainingPieces.remove(p)
                                pieceList.remove(p)
                                puzzleGrid[i][j] = p
                                prevPiece = p
                                matchNotFound = False
                                break
                            else:
                                triedPieces.add(p)
                    else:
                        #print("No pieceMatches found for tolerance: " + str(tolerance))
                        pass

    def pieceTesterWithContour(self,piece1, piece2,horizontalWise,minWidth,maxWidth,minHeight,maxHeight,nbr):
        waitTime = 1
        screenImg = self.getFieldScreenshot()
        filteredScreen = self.filterImage3(nbr,screenImg)

        if horizontalWise == True:
            targetPoint = [piece1.topLeftPoint[0] + piece1.center[3][1][0][0],piece1.topLeftPoint[1] + piece1.center[3][1][0][1]]
            closestPoint = self.TLcontour(nbr,horizontalWise,targetPoint)
            vectorCH = [piece2.center[3][0][0][0]-piece2.center[0][0],piece2.center[3][0][0][1]-piece2.center[0][1]]
            xDest = closestPoint[0] - vectorCH[0]
            yDest = closestPoint[1] - vectorCH[1]+ self.topBarHeight
            y = piece1.topLeftPoint[1]-self.topBarHeight
            h = maxHeight
            x = piece1.topLeftPoint[0]+piece1.imgWidth//2
            w = piece1.imgWidth//2+piece2.imgWidth//2
        else:
            targetPoint = [piece1.topLeftPoint[0] + piece1.center[3][0][0][0],piece1.topLeftPoint[1] + piece1.center[3][0][0][1]]
            closestPoint = self.TLcontour(nbr,horizontalWise,targetPoint)
            vectorCH = [piece2.center[3][3][0][0]-piece2.center[0][0],piece2.center[3][3][0][1]-piece2.center[0][1]]
            xDest = closestPoint[0] - vectorCH[0]
            yDest = closestPoint[1] - vectorCH[1] + self.topBarHeight
            y = piece1.topLeftPoint[1]-self.topBarHeight+piece1.imgHeight//2
            h = piece1.imgHeight//2+piece2.imgHeight//2
            x = piece1.topLeftPoint[0]
            w = maxWidth

        cv.circle(filteredScreen,closestPoint,1,100,1)
        cv.circle(filteredScreen,targetPoint,1,100,1)
        cv.imwrite( self.basePath + "puzzleimages/hornpunkter/hornpunkter" + str(self.savedImagesNbr) + ".png",filteredScreen)
        self.savedImagesNbr +=1

        if piece1 == None:
            return True

        lastPos = (piece2.center[0][0]+piece2.topLeftPoint[0],piece2.center[0][1]+piece2.topLeftPoint[1])

        pieceMover.pieceMover.pieceMoveAndHover2(piece2,xDest,yDest)
        #print("testing piece")
        if self.detectChange(waitTime,y,h,x,w):
            pieceMover.pieceMover.click()
            #print("Match!")
            return True
        else:
            pieceMover.pieceMover.pieceMoveAndDrop2(piece2,lastPos[0],lastPos[1])
            return False
        
    def pieceTesterBasic(self,piece1, piece2,horizontalWise,minWidth,maxWidth,minHeight,maxHeight,nbr):
        waitTime = 1
        
        if horizontalWise == True:
            xDest = piece1.topLeftPoint[0] + piece1.center[3][2][0][0] + (piece2.center[0][0]-piece2.center[3][3][0][0])
            yDest = piece1.topLeftPoint[1] + piece1.center[3][2][0][1] + (piece2.center[0][1]-piece2.center[3][3][0][1])
            y = piece1.topLeftPoint[1]-self.topBarHeight
            h = maxHeight
            x = piece1.topLeftPoint[0]+piece1.imgWidth//2
            w = piece1.imgWidth//2+piece2.imgWidth//2
        else:
            xDest = piece1.topLeftPoint[0] + piece1.center[3][0][0][0] + (piece2.center[0][0]-piece2.center[3][3][0][0])
            yDest = piece1.topLeftPoint[1] + piece1.center[3][0][0][1] + (piece2.center[0][1]-piece2.center[3][3][0][1])
            y = piece1.topLeftPoint[1]-self.topBarHeight+piece1.imgHeight//2
            h = piece1.imgHeight//2+piece2.imgHeight//2
            x = piece1.topLeftPoint[0]
            w = maxWidth

        if piece1 == None:
            return True

        lastPos = (piece2.center[0][0]+piece2.topLeftPoint[0],piece2.center[0][1]+piece2.topLeftPoint[1])

        pieceMover.pieceMover.pieceMoveAndHover2(piece2,xDest,yDest)
        #print("testing piece")
        if self.detectChange(waitTime,y,h,x,w):
            #print("Match!")
            pieceMover.pieceMover.click()
            return True
        else:
            pieceMover.pieceMover.pieceMoveAndDrop2(piece2,lastPos[0],lastPos[1])
            return False
        
    def detectChange(self, t, y,h,x,w) -> bool:
        frequency = 5
        period = 0
        while period < t:
            img1 = cv.cvtColor(self.getFieldScreenshot(),cv.COLOR_BGR2GRAY)[y:y+h,x:x+w]
            time.sleep(1/frequency)
            period += 1/frequency
            img2 = img1
            img1 = cv.cvtColor(self.getFieldScreenshot(),cv.COLOR_BGR2GRAY)[y:y+h,x:x+w]
            if len(np.nonzero(cv.bitwise_xor(img1,img2))[0]) != 0:
                return True
        return False
    
    def gridCoverage(self,grid):
        dimensions = [] #y1,y2,x1,x2
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                p = grid[i][j]
                if p == None:
                    break
                if p.topLeftPoint[0] < x1:
                    x1 = p.topLeftpoint[0]
                if p.topLeftPoint[1] < y1:
                    y1 = p.topLeftpoint[1]
                if p.topLeftPoint[0] + p.imgWidth > x2:
                    x2 = p.topLeftPoint[0] + p.imgWidth
                if p.topLeftPoint[1] + p.imgHeight > y2:
                    y2 = p.topLeftPoint[1] + p.imgHeight

        return (y1,y2,x1,x2)

    def moveGrid(self,grid,xDest,yDest):
        p = grid[0][0]
        x1,y1 = p.topLeftPoint
        xDiff = xDest-x1-p.center[0][0]
        yDiff = yDest-y1-p.center[0][1]
        pieceMover.pieceMover.pieceMove2(p,xDest,yDest)
        for i in range(len(grid)):
            for j in range(1,len(grid[0])):
                if grid[i][j] == None:
                    continue
                x,y = grid[i][j].topLeftPoint
                x +=xDiff
                y +=yDiff
                grid[i][j].topLeftPoint = x,y
        #print("finished moving grid")

    def TLcontour(self,nbr,horizontal,targetPoint): #point är lista [x,y], horizontal boolean
        screen = self.getFieldScreenshot()
        filteredScreen = self.filterImage3(nbr,screen)

        contours, hierarchy = cv.findContours(filteredScreen,cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        
        TLcontour = []
        minDistance = 100000
        for c in contours:
            dist = abs(cv.pointPolygonTest(c,[0,0],True))
            if dist < minDistance and len(c) > 1:
                TLcontour = c
                minDistance = dist
        
        cornerPoint = []
        x,y,w,h = cv.boundingRect(TLcontour)
        boundingImg = filteredScreen[y:y+h,x:x+w]

        if horizontal == True:
            minDist = 1000000
            for p in TLcontour: #avstånd från bottombound till punkt på konturen + från konturen till målpunkten
                dist = math.hypot(abs(p[0][0]-targetPoint[0]),abs(p[0][1]-targetPoint[1])) + abs(p[0][0]-x-w)+abs(p[0][1]-y-h)
                if dist < minDist:
                    cornerPoint = p[0]
                    minDist = dist
                
            cornerPoint = list(map(int,cornerPoint))
            inContour = True
            while inContour == True: #hittar punkten längst ner på konturen från tidigare hittad hörnpunkt
                cornerPoint[1] +=1
                posValue = cv.pointPolygonTest(TLcontour,cornerPoint,False)
                if posValue == -1:
                    inContour = False
            cornerPoint[1] -=1
            cornerPoint[1] +=9 #test med artificiell ökning

        else:
            cornerPoint = [x,y]
            extraHeight = 0 #hittar lägsta punkten längst vänstra sidan, oberoende av targetpoint
            for i in range(h):
                if boundingImg[i][0] == 255:
                    extraHeight = i
            #print(extraHeight)
            cornerPoint[1] += extraHeight
            cornerPoint[1] +=12

            # cv.circle(filteredScreen,cornerPoint,1,100,1)
            # cv.circle(filteredScreen,targetPoint,1,100,1)

            # cv.namedWindow("es")
            # cv.moveWindow("es",400,400)

            # cv.imshow("es",filteredScreen)
            # k = cv.waitKey(0) & 0xFF
            # cv.destroyAllWindows()

            # minDist = 1000000   #
            # for p in TLcontour: #avstånd från leftbound till punkt på konturen + från konturen till målpunkten
            #     dist = math.hypot(abs(p[0][0]-targetPoint[0]),abs(p[0][1]-targetPoint[1]))
            #     if dist < minDist:
            #         cornerPoint = p[0]
            #         minDist = dist
        
        return cornerPoint
            

        # self.savedImagesNbr +=1
        # cv.circle(filteredScreen,closestContourPoint,3,100,3)

        # cv.imwrite(self.basePath + "puzzleimages/punktförsök" + str(self.savedImagesNbr) + ".png",filteredScreen)














        # def spreadPiecesOut(self,collection):
    
    
    
    
    
    
    
    
    
    
    
    
    
    #     print("spreading pieces out")
    #     collection.setMinPieceDimensions()
    #     extraRatio = 1.2
    #     pieceWidth,pieceHeight = round(collection.minWidth*extraRatio),round(collection.minHeight*extraRatio)
    #     maxWidthSideRoom = 0
    #     maxHeightSideRoom = 0
    #     puzzleStart = (0,0)
    #     puzzleClearingMargin = 50
    #     margin = 3 #antal pixlar som är extra marginal runt varje centrum
    #     puzzleHeight,puzzleWidth = collection.puzzleDimensions()
    #     for p in collection.pieces:
    #         if p.center[0][0] > maxWidthSideRoom or p.imgWidth - p.center[0][0] > maxWidthSideRoom:
    #             maxWidthSideRoom = max(p.center[0][0], p.imgWidth - p.center[0][0])
    #         if p.center[0][1] > maxHeightSideRoom or p.imgHeight - p.center[0][1] > maxHeightSideRoom:
    #             maxHeightSideRoom = max(p.center[0][1], p.imgHeight - p.center[0][1])

    #     puzzleClearing = (puzzleStart[0],puzzleStart[0] +puzzleWidth*pieceWidth+puzzleClearingMargin,puzzleStart[1]+self.topBarHeight,puzzleStart[1]+pieceHeight*puzzleHeight+puzzleClearingMargin+self.topBarHeight) #x,x+w,y,y+h
        
    #     pileDim = collection.pileDim
    #     newPileDim = (pileDim[2],pileDim[3],pileDim[0],pileDim[1])

    #     nbr = 1
    #     prevBoundary = 0
    #     extra = 5

    #     xPositions = (self.screenWidth-2*extra-2*margin)//(2*margin+maxWidthSideRoom)
    #     yPositions = (self.screenHeight-2*extra-2*margin)//(2*margin+maxHeightSideRoom)
    #     currX = 0
    #     currY = 0
    #     pieceList = list(reversed(collection.pieces))
    #     amount = len(pieceList)
    #     i = 0
    #     while i <= amount-1:
    #         if currX == xPositions-1 and currY == yPositions-1:
    #             raise Exception("Not enough space to spread out pieces")

    #         xDest = extra + margin + currX *(2*margin + maxWidthSideRoom)
    #         yDest = extra + margin + currY *(2*margin + maxHeightSideRoom)+self.topBarHeight

    #         if xDest + extra > self.screenWidth:
    #             currX = 0
    #             currY += 1
    #             continue

    #         if self.posInArea(newPileDim,(xDest,yDest)) or self.posInArea(puzzleClearing,(xDest,yDest)):
    #             currX += 1
    #             print("skipped over area")
    #             continue
    #         else:
    #             self.pieceMove2(pieceList[i],xDest,yDest)
    #             currX += 1
    #             i +=1

    # def separatePieces(self,nbr):
    #     img = self.filterImage3(nbr)

    #     contours, _ = cv.findContours(img,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #     minWidth, minHeight = self.randomPieceDim(nbr)
    #     maxVerStacked = (self.screenHeight-self.topBarHeight)//minHeight
    #     maxHorStacked = (self.screenWidth)//minWidth
    #     smallerMargin = 0.8
    #     largerMargin = 1.5

    #     for i in range(len(contours)):
    #         boundRect = cv.boundingRect(contours[i])
    #         if boundRect[2] > minWidth*smallerMargin and boundRect[3] > minHeight*smallerMargin:
    #             cv.rectangle(img,(boundRect[0],boundRect[1],boundRect[2],boundRect[3]),100,3)

    #     return img

        # def collectionCreate(self,nbr):
        # collection = PieceCollection.PieceCollection()
        # eroded = self.filterImage(nbr)

        # cnts1 = cv.findContours(eroded.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # cnts2 = imutils.grab_contours(cnts1)

        # i = 0
        # for c in cnts2:
        #     (x,y,w,h) = cv.boundingRect(c)
        #     if (w > 60 and h>60):
        #         height,width,_ = self.getDimensions(nbr)
        #         blankImg = np.zeros((height,width,1),np.uint8)
        #         cv.drawContours(blankImg,cnts2,i,255,-1)
        #         crop = blankImg[y:y+h,x:x+w]
                

        #         piece = PuzzlePiece.PuzzlePiece((x,y),w,h,crop,collection)
        #         collection.addPiece(piece)
        #     i += 1

        # return collection

    # def puzzleImgDifferenceSum(self):
    #     imagePathBase = self.basePath + "puzzleimages/pythonimgbackground"
    #     imagePuzzleBase = self.basePath + "puzzleimages/pythonimgPuzzle"
    #     theme = 1
    #     for i in range(1):
    #         imgPuzzle = cv.imread(imagePuzzleBase + str(theme) + ".jpg")
    #         imgBackground = cv.imread(imagePathBase + str(theme) + ".jpg")
    #         imgDiff = cv.absdiff(imgPuzzle,imgBackground)
    #         if i == 0:
    #             gray = cv.cvtColor(imgDiff,cv.COLOR_RGB2GRAY)
    #             ret,res = cv.threshold(gray,0,255,cv.THRESH_BINARY)
    #             imgSum = res
    #         else:
    #             gray = cv.cvtColor(imgDiff,cv.COLOR_RGB2GRAY)
    #             ret,res = cv.threshold(gray,0,255,cv.THRESH_BINARY)
    #             imgSum = cv.bitwise_or(imgSum,res)
        
    #     return imgSum

    

    # def getImgArray(self):
    #     return self.imgArray

    # def getCurrentArray(self):
    #     return self.currArray
    
    # def getShape(self):
    #     return self.shape

    # def showCurrentArray(self):
    #     z = self.currArray.astype(np.uint8)
    #     im = Image.fromarray(z)
    #     im.show()

        # def backHsvHistRange(self, nbr):
    #     if nbr == 1:
    #         histrange = [[102,106],[69,98],[126,196]]
    #         return histrange
    #     else:
    #         return None

        # def filterImage(self,nbr):
        # hsvRange = self.backHsvHistRange(nbr)

        # hsvMin = np.array([hsvRange[0][0],hsvRange[1][0],hsvRange[2][0]],np.uint8)
        # hsvMax = np.array([hsvRange[0][1],hsvRange[1][1],hsvRange[2][1]],np.uint8)
        # filtered = cv.inRange(cv.cvtColor(self.getImg(nbr),cv.COLOR_BGR2HSV),hsvMin,hsvMax)
        # filtered = cv.bitwise_not(filtered)

        # kernel = np.ones((3,3),np.uint8)
        # eroded = cv.erode(filtered,kernel,iterations=1)

        # return eroded