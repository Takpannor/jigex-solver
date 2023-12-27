import cv2 as cv
import PuzzlePiece
#from matplotlib import pyplot as plt
import os
import numpy as np
import copy
import shutil

# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# from keras.models import Sequential
# import keras_cv

import pathlib



class PieceCollection:
    basePath = "/home/mrmuffinswigle/Desktop/programming/python/eget/Puzzle/"

    def __init__(self) -> None:
        self.nbrPieces = 0
        self.pieces = []
        self.pileDim = () #y, y+h, x, x+w
        self.minWidth = 0
        self.minHeight = 0
        self.cornerPts = []
        self.pHandler = None

    def addPiece(self, piece) -> None:
        self.nbrPieces +=1
        self.pieces.append(piece)

    def getPieceImg(self,i):
        return self.pieces[i].pieceImg
    
    def showPieces(self):
        for i in range(self.nbrPieces):
            self.pieces[i].showPiece()
    
    def deleteFolderContents(self, folderDir) -> None:
        for filename in os.listdir(folderDir):
            os.remove(os.path.join(folderDir,filename))
    
    def saveCollection(self) -> None:
        if len(self.pieces) == 0:
            raise Exception("Collection is empty")
        self.deleteFolderContents(self.basePath + "puzzleimages/collection")
        self.deleteFolderContents(self.basePath + "puzzleimages/collection")
        f = open(self.basePath + "collectionText.txt","w")
        f.write(str(self.pileDim) + "\n")
        f.write(str(self.nbrPieces)+"\n")
        for i in range(self.nbrPieces):
            piece = self.pieces[i]
            cv.imwrite(self.basePath + "puzzleimages/collection/piece" + str(i) + ".png",piece.pieceImg)
            cv.imwrite(self.basePath + "puzzleimages/bgrcollection/piece" + str(i) + ".png",piece.bgrImg)
            f.write(str(piece.topLeftPoint)+"\n")
        f.close()

    def loadCollection(self) -> None: #(read from file
        f = open(self.basePath + "collectionText.txt","r")
        pileStr = f.readline()
        pileStr = pileStr.strip(" ()\n")
        self.pileDim = tuple(map(int,pileStr.split(",")))
        nbr = int(f.readline())
        for i in range(nbr):
            coordStr = f.readline()
            coordStr = coordStr.strip("()\n")
            coords = tuple(map(int,coordStr.split(",")))
            topLeftPoint = coords
            img = cv.imread(self.basePath + "puzzleimages/collection/piece" + str(i) + ".png",cv.IMREAD_GRAYSCALE)
            piece = PuzzlePiece.PuzzlePiece(img)
            piece.topLeftPoint = topLeftPoint
            piece.bgrImg = cv.imread(self.basePath + "puzzleimages/bgrcollection/piece" + str(i) + ".png")
            self.addPiece(piece)
        f.close()

    def getImgDimMinMax(self):
        widthArr = []
        heightArr = []
        for i in range(self.nbrPieces):
            h,w = self.pieces[i].pieceImg.shape
            widthArr.append(w)
            heightArr.append(h)
        minWidth = min(widthArr)
        maxWidth = max(widthArr)
        minHeight = min(heightArr)
        maxHeight = max(heightArr)

        return minWidth,maxWidth,minHeight,maxHeight
    
    def setMinPieceDimensions(self) -> None:
        width = 1000000
        height = 10000000
        for p in self.pieces:
            _, pWidth, pHeight,_ = p.findCenter()
            if pWidth < width:
                width = pWidth
            if pHeight < height:
                height = pHeight
        
        self.minHeight = height
        self.minWidth = width

    def setCenterForAllPieces(self) -> None:
        for p in self.pieces:
            center = p.findCenter()
            p.center = center

    def determineAllSides(self) -> None:
        for p in self.pieces:
            sides = p.determineSides()
            p.sideMatchings = sides

    def puzzleDimensions(self) -> None:
        self.determineAllSides()
        topLength = 0
        leftLength = 0
        botLength = 0
        rightLength = 0

        for p in self.pieces:
            for j in range(4):
                if p.sideMatchings[j][0][0] == 0:
                    if j == 0:
                        rightLength += 1
                    if j == 1:
                        topLength += 1
                    if j == 2:
                        leftLength += 1
                    if j == 3:
                        botLength += 1

        horizontalOptions = []
        verticalOptions = []

        horizontalOptions.append(topLength)
        horizontalOptions.append(botLength)
        verticalOptions.append(leftLength)
        verticalOptions.append(rightLength)

        for h in horizontalOptions:
            for v in verticalOptions:
                if h*v == len(self.pieces):
                    return v,h

        raise Exception("puzzle dimensions unclear:" + str((topLength,botLength,leftLength,rightLength,len(self.pieces))))
    
    def matchPiece(self, piece, sideRequirements, tolerance) -> bool: #requirements counter clockwise, same as piece sides, returns boolean
        for i in range(4):
            if sideRequirements[i] == None:
                continue
            for j in range(17):
                if piece.sideMatchings[i][j][1] > tolerance:
                    if len(piece.sideMatchings[i][0:j]) == 0:
                        return False
                    for s in piece.sideMatchings[i][0:j]:
                        if sideRequirements[i] == s[0]:
                            break
                        else: 
                            return False
                    break
        return True

    def matchFinder(self, leftPiece, topPiece, edges ,availablePieces, tolerance) -> list: #
        matchingPieces = set()
        leftIterations = 0
        topIterations = 0
        #fixa ifall left/toppiece = None

        sideRequirements = [None,None,None,None]
        for i in edges:
            sideRequirements[i] = 0

        if leftPiece == None:
            leftIterations = 1
        else:
            for s in leftPiece.sideMatchings[0]:
                if s[1] <= tolerance:
                    leftIterations +=1
                else:
                    break
        
        if topPiece == None:
            topIterations = 1
        else:
            for s in topPiece.sideMatchings[2]:
                if s[1] <= tolerance:
                    topIterations +=1
                else:
                    break

        for i in range(leftIterations):
            if leftPiece != None:
                sideRequirements[2] = -leftPiece.sideMatchings[0][i][0]
            for j in range(topIterations):
                if topPiece != None:
                    sideRequirements[1] = -topPiece.sideMatchings[3][j][0]
                for p in availablePieces:
                    if self.matchPiece(p,sideRequirements,tolerance):
                        matchingPieces.add(p)

        return matchingPieces

    def checkCollectionSideSum(self) -> None: #väldigt användbar
        totSum = 0
        occurenceList = []

        for j in range(4):
            occurenceList.append([])
            for i in range(17):
                occurenceList[j].append(0)

        for p in self.pieces:
            nbrs = p.sideType
            for i in range(4):
                occurenceList[i][nbrs[i]+8] +=1
            totSum += sum(nbrs)

        print("Training sum: " + str(totSum))
        print("horizontal")
        print(occurenceList[0])
        occurenceList[2].reverse()
        print(occurenceList[2])
        print("vertical")
        print(occurenceList[1])
        occurenceList[3].reverse()
        print(occurenceList[3])

    def showSpecificSides(self,side,sideType) -> None:
        for c,p in enumerate(self.pieces):
            if p.sideType[side] == sideType:
                print(c)
                p.showPiece()

    def accuracyTest(self) -> None:
        for p in self.pieces:
            sides = p.determineSides(1000,1000)
            if not np.array_equal(sides,p.sideType):
                print("pieceNbr: " + str(p.nbr))
                print("Expected values: " + str(p.sideType))
                print("Recieved values: " + str(sides) + "\n")
                p.showPiece()

    def saveStandardSides(self) -> None:
        sideTypes = self.types
        path = self.basePath + "sideTypeStandards/"

        for s in sideTypes:
            for p in self.pieces:
                for i in range(4):
                    if p.sideType[i] == s:
                        segment = p.pieceSegmenter(height = 1000,width = 1000,targetWidth = 200)[i]
                        p.savePointList(segment,path + str(s))

    def getPiece(self,pieceNbr) -> PuzzlePiece.PuzzlePiece:
        for p in self.pieces:
            if p.nbr == pieceNbr:
                return p
            
    def matchRequiredDistStatistics(self) -> int: #hur nära rätt svar är ifrån bästa matchningen
        requiredDist = 0
        for p in self.pieces:
            segments = p.pieceSegmenter(targetWidth=200)
            for i in range(4):
                matches = p.determineSegmentSideType(segments[i])
                j = 0
                while matches[j][0] != p.sideType[i]:
                    if p.sideType[i] == 0:
                        print("error for flat")
                    j +=1
                if matches[j][1] > requiredDist:
                    requiredDist = matches[j][1]

        return requiredDist

    def distErrors(self,maxSqDiff) -> int:
        errors = 0
        for p in self.pieces:
            sides = p.determineSides(maxSqDiff)
            for i in range(4):
                for s in sides[i]:
                    if len(sides[i]) > 0 and s != p.sideType[i]:
                        errors +=1
        return errors
    











































    # def firstMatchStatistics(self) -> None: #ger statistik över vilken tolerans 
    #     y = []
    #     for p in self.pieces:
    #         sideMatches = p.determineSides()
    #         for i in range(4):
    #             if sideMatches[i][0][0] == p.sideType[i]:
    #                 y.append(sideMatches[i][0][1])

    #     fig, axs = plt.subplots()

    #     print(y)
    #     axs.hist(y)
    #     plt.show(block = False)
        
    # def distErrorStatistics(self) -> None: #antalet fel svar som erhålls med ett visst avstånd i kvadrat
    #     x = list(range(700,1500,100))
    #     y = []
    #     for k in x:
    #         y.append(self.distErrors(k))

    #     print(y)
    #     fig, ax = plt.subplots()

    #     ax.bar(x,y,width = 4)
    #     plt.show(block = False)













    # def saveTrainingImgs(self,folderName): # separat funktion för kategorisering
    #     if len(self.pieces) == 0:
    #         raise Exception("Collection is empty")
    #     basePath = self.basePath + "puzzleimages/traningspussel/"

    #     if folderName in os.listdir(basePath):
    #         shutil.rmtree(basePath+folderName)
    #     os.mkdir(basePath + folderName)
    #     os.mkdir(basePath + folderName + "/bgr")
    #     os.mkdir(basePath + folderName + "/bw")
    #     os.mkdir(basePath + folderName + "/kategoriserat bw")
    #     with open(basePath + folderName + "/collectionText.txt","w") as f:
    #         f.write(str(self.nbrPieces) + "\n")
    #         for c,p in enumerate(self.pieces):
    #             cv.imwrite(basePath + folderName + "/bw/" + str(c) + ".png",p.pieceImg)
    #             cv.imwrite(basePath + folderName + "/bgr/" + str(c) + ".png",p.bgrImg)
    #             f.write(str(p.topLeftPoint)+"\n")

    # def categorizeTrainingPieces(self,folderName): #foldername pusselnamnet
    #     basePath = self.basePath + "puzzleimages/traningspussel/"
    #     letters = {"a": 1, "s": -2, "d": 2, "f": -4, "j": 4, "k": 6, "l": -6, "ö": 0, "q": -1, "w": -3, "e": 3, "r": -5, "u":5, "i": 7,"o":-7, "p": "okategoriserat"}
    #     keys = {97: "a", 115: "s", 100: "d", 102: "f", 106: "j", 107: "k", 108: "l", 246:"ö", 113: "q", 119: "w", 101: "e", 114: "r", 117: "u", 105: "i", 111: "o", 112: "p"}
    #     for fileName in os.listdir(basePath + folderName + "/bw"):
    #         imgBgr = cv.imread(basePath + folderName + "/bgr" + "/" + fileName)
    #         imgBw = cv.imread(basePath + folderName + "/bw" + "/" + fileName)

    #         pieceNbr = int(fileName.split(".")[0])

    #         img = imgBgr #np.hstack((imgBgr,imgBw))
    #         plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
    #         plt.show(block = False)

    #         pieceName = str(pieceNbr) + "_"
    #         for j in range(4):
    #             s = input("Enter side " + str(j) + ": ")
    #             sideType = letters[s]
    #             print("You entered: " + str(s))
    #             pieceName += str(sideType)
    #             if j != 3:
    #                 pieceName += ","
    #         plt.close("all")
    #         if not cv.imwrite(basePath + folderName + "/kategoriserat bw/" + pieceName + ".png",imgBw):
    #             raise Exception("Could not write bwimage")

    # def loadTrainingPieces(self,folderName):
    #     basePath = self.basePath + "puzzleimages/traningspussel/"
        
    #     for fileName in os.listdir(basePath + folderName + "/kategoriserat bw 2"):
    #         name = fileName[:-4]
    #         chunk = name.split("_")
    #         sideNbrs = list(map(int,chunk[1].split(",")))
    #         bwImg = cv.imread(basePath + folderName + "/kategoriserat bw 2/" + fileName)
    #         bgrImg = cv.imread(basePath + folderName + "/bgr/" + chunk[0] + ".png")

    #         piece = PuzzlePiece.PuzzlePiece(cv.cvtColor(bwImg,cv.COLOR_BGR2GRAY))
    #         piece.bgrImg = bgrImg
    #         piece.sideType = sideNbrs
    #         piece.nbr = int(chunk[0])
    #         self.addPiece(piece)

        # def trainingTransfer(self) -> None:
        # path = self.basePath + "puzzleimages/traningspussel/50 green/kategoriserat bw/"
        # for fileName in os.listdir(path):
        #     img = cv.imread(path + fileName)
        #     chunk = fileName[:-4].split("_")
        #     nbr = chunk[0]
        #     sides = list(map(int,chunk[1].split(",")))
        #     for i in range(len(sides)):
        #         if sides[i] == 1 or sides[i] == -1:
        #             plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
        #             plt.show(block = False)
        #             print(str(nbr) + " looking for side: " + str(i))
        #             side = input("enter sideType: ")
        #             sides[i] = int(side)

        #             plt.close("all")
        #         elif sides[i] > 0:
        #             sides[i] = sides[i] +1
        #         elif sides[i] < 0:
        #             sides[i] = sides[i] -1

        #     pieceName = nbr + "_"
        #     for i in range(4):
        #         pieceName += str(sides[i])
        #         if i != 3:
        #             pieceName += ","
        #     cv.imwrite(self.basePath + "puzzleimages/traningspussel/50 green/kategoriserat bw 2/" + pieceName + ".png",img)























    # def getIdenticalPieces(self):
    #     for p1 in self.pieces:
    #         for p2 in self.pieces:
    #             if p1.sideType == p2.sideType and p1 != p2:
    #                 print("found identical pieces")

    # def matchFinder(self,grid,pieceList,xPos,yPos): #ska ge array med matchningar som sedan testas i imagescanner
    #     height,width = len(grid),len(grid[0])

    #     matches = 0
    #     matchList = []
    #     requiredSideType = self.getRequiredSideTypesForPos(grid,xPos,yPos,width,height)
    #     for p in pieceList:
    #         if self.matchPiece(p,requiredSideType):
    #             matches +=1
    #             matchList.append(p)

    #     if matches == 0:
    #         raise Exception("didnt find a match for position: (" + str(xPos) + "," + str(yPos) + "), h,w: " + str(height) + "," + str(width))
    #     return matchList

    # #     return sideLength,topLength #dvs höjd och bredd
    
    # # def matchPiece(self,piece,requiredSideType):
    # #     sideType = piece.sideType
    # #     for i in range(4):
    # #         if sideType[i] != requiredSideType[i] and requiredSideType[i] != None:
    # #             return False
    # #     return True
        
    # # def getRequiredSideTypesForPos(self,grid,xpos,ypos,width,height):
    # #     requiredSideType = [None,None,None,None]
    # #     sidesLeft = [0,1,2,3]
    # #     if xpos == 0:
    # #         requiredSideType[2] = 0
    # #         sidesLeft.remove(2)
    # #     elif xpos == width-1:
    # #         requiredSideType[0] = 0
    # #         sidesLeft.remove(0)
    # #     if ypos == 0:
    # #         requiredSideType[1] = 0
    # #         sidesLeft.remove(1)
    # #     elif ypos == height-1:
    # #         requiredSideType[3] = 0
    # #         sidesLeft.remove(3)

    # #     for side in sidesLeft:
    # #         if side == 0:
    # #             p = grid[ypos][xpos+1]
    # #             if p != None:
    # #                 requiredSideType[0] = p.getSideMatch(2)
    # #         if side == 1:
    # #             p = grid[ypos-1][xpos]
    # #             if p != None:
    # #                 requiredSideType[1] = p.getSideMatch(3)
    # #         if side == 2:
    # #             p = grid[ypos][xpos-1]
    # #             if p != None:
    # #                 requiredSideType[2] = p.getSideMatch(0)
    # #         if side == 3:
    # #             p = grid[ypos+1][xpos]
    # #             if p != None:
    # #                 requiredSideType[3] = p.getSideMatch(1)

    # #     return requiredSideType

    # def saveCrops(self): #funkar ej för tillfället
    #     #self.deleteFolderContents(self.basePath + "puzzleimages/testsides")
    #     min, max = self.getImgDimMinMax()
    #     for i in range(self.nbrPieces):
    #         sideImgs,_ = self.pieces[i].cropSides(min,max)
    #         for j in range(4):
    #             cv.imwrite(self.basePath + "puzzleimages/testsides/piece_" + str(i) + "_side_" + str(j) + ".png",sideImgs[j])

    # def showCrops(self):
    #     cv.namedWindow("es")
    #     cv.moveWindow("es",400,400)
    #     min, max = self.getImgDimMinMax()
    #     for i in range(self.nbrPieces):
    #         sideImgs,bgrImgs = self.pieces[i].cropSides(min,max)
    #         for j in range(4):
    #             bwImg = cv.cvtColor(sideImgs[j],cv.COLOR_GRAY2BGR)
    #             newImg = np.concatenate((bgrImgs[j],bwImg),axis=1)
    #             cv.imshow("es",sideImgs)
    #             k = cv.waitKey(0) & 0xFF
    #             cv.destroyAllWindows()




    # def saveTrainingImgs(self):
    #     folderName = "green puzzle 144 pieces"
    #     min, max = self.getImgDimMinMax()
    #     cv.namedWindow("es")
    #     cv.moveWindow("es",400,400)
    #     letters = {"a": 1, "s": -2, "d": 2, "f": -4, "j": 4, "k": 6, "l": -6, "ö": 0, "q": -1, "w": -3, "e": 3, "r": -5, "u":5, "i": 7,"o":-7, "p": "okategoriserat"}
    #     keys = {97: "a", 115: "s", 100: "d", 102: "f", 106: "j", 107: "k", 108: "l", 246:"ö", 113: "q", 119: "w", 101: "e", 114: "r", 117: "u", 105: "i", 111: "o", 112: "p"}
    #     nbr = 0
    #     for i in range(self.nbrPieces):
    #         plt.imshow(cv.cvtColor(self.pieces[i].bgrImg,cv.COLOR_BGR2RGB))
    #         plt.show(block = False)
    #         pieceName = ""
    #         for j in range(4):
    #             pressed = input("Enter side " + str(j) + ": ")
    #             try: 
    #                 pieceCat = letters[pressed]
    #             except:
    #                 continue
    #             print("Entered side type: " + pressed + ", " + str(pieceCat))
    #             pieceName += str(pieceCat) + ","
    #         plt.close("all")

    #         pieceName = pieceName[:-1]
    #         piecesPath = self.basePath + "puzzleimages/traningsbitar/" + folderName + "/classifiedPieces/"+ str(i) + "_"+ pieceName +".png"
    #         bgrPath = self.basePath + "puzzleimages/traningsbitar/"+ folderName + "/bgrPieces/" + str(i) + ".png"
    #         if not cv.imwrite(piecesPath,self.pieces[i].pieceImg):
    #             raise Exception("Could not write bwimage")
    #         if not cv.imwrite(bgrPath,self.pieces[i].bgrImg):
    #             raise Exception("Could not write bgrimage")

    # def checkTrainingPieces(self): #hor is boolean
    #     wantedType = int(input("Enter wantedType: "))
    #     sideNbr = int(input("Enter side to check: "))
    #     # if sideNbr == 2 or 3:
    #     #     wantedType *= -1
    #     for i in range(len(self.pieces)):
    #         piece = self.pieces[i]
    #         if piece.sideType[sideNbr] == wantedType:
    #             plt.imshow(cv.cvtColor(piece.bgrImg,cv.COLOR_BGR2RGB))
    #             print(i)
    #             print(piece.sideType)
    #             plt.show(block = False)
    #             input("press enter:")

    # def burnTrainingFolders(self):
    #     letters = {"a": 1, "s": -2, "d": 2, "f": -4, "j": 4, "k": 6, "l": -6, "ö": 0, "q": -1, "w": -3, "e": 3, "r": -5, "u":5, "i": 7,"o":-7}
    #     basePath = self.basePath + "puzzleimages/traningsbitar/"
    #     for key in letters:
    #         self.deleteFolderContents(basePath + str(letters[key]))    

    # def trainCNN(self):
    #     dataset_path = self.basePath + "puzzleimages/traningsbitar"
    #     data_dir = pathlib.Path(dataset_path)
    #     batch_size = 32
    #     img_height = 100
    #     img_width = 200

    #     model_path = self.basePath + "model.keras"

    #     train_ds = tf.keras.utils.image_dataset_from_directory(
    #         data_dir,
    #         validation_split=0.2,
    #         subset="training",
    #         seed=123,
    #         image_size=(img_height,img_width),
    #         batch_size=batch_size
    #     )

    #     val_ds = tf.keras.utils.image_dataset_from_directory(
    #     data_dir,
    #     validation_split=0.2,
    #     subset="validation",
    #     seed=123,
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size
    #     )
        
    #     class_names = train_ds.class_names
    #     num_classes = len(class_names)

    #     model = Sequential([keras_cv.layers.Grayscale(output_channels=1),
    #         layers.Rescaling(1./255),
    #         layers.Conv2D(16,3,padding="same",activation="relu"),
    #         layers.MaxPooling2D(),
    #         layers.Conv2D(32, 3, padding='same', activation='relu'),
    #         layers.MaxPooling2D(),
    #         layers.Conv2D(64, 3, padding='same', activation='relu'),
    #         layers.MaxPooling2D(),
    #         layers.Flatten(),
    #         layers.Dense(128, activation='relu'),
    #         layers.Dense(num_classes)])
        
    #     model.compile(optimizer="adam",
    #                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #                   metrics=["accuracy"])
        
    #     epochs = 10
    #     history = model.fit(
    #         train_ds,
    #         validation_data=val_ds,
    #         epochs=epochs,
    #     )

    #     model.save(model_path)

    #     acc = history.history['accuracy']
    #     val_acc = history.history['val_accuracy']

    #     loss = history.history['loss']
    #     val_loss = history.history['val_loss']

    #     epochs_range = range(epochs)

    #     plt.figure(figsize=(8, 8))
    #     plt.subplot(1, 2, 1)
    #     plt.plot(epochs_range, acc, label='Training Accuracy')
    #     plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    #     plt.legend(loc='lower right')
    #     plt.title('Training and Validation Accuracy')

    #     plt.subplot(1, 2, 2)
    #     plt.plot(epochs_range, loss, label='Training Loss')
    #     plt.plot(epochs_range, val_loss, label='Validation Loss')
    #     plt.legend(loc='upper right')
    #     plt.title('Training and Validation Loss')
    #     plt.show()

    # def applyCNN(self):
    #     min, max = self.getImgDimMinMax()
    #     model_path = self.basePath + "model.keras"
    #     class_list = os.listdir(self.basePath + "puzzleimages/traningsbitar")

    #     model = keras.models.load_model(model_path)
    #     allCrops = []

    #     for i in range(len(self.pieces)):
    #         pieceCrops,colorCrops = self.pieces[i].cropSides(min,max)
    #         for j in range(4):
    #             pieceCrops[j] = np.expand_dims(pieceCrops[j],axis=0)
    #             allCrops.append(pieceCrops[j])

    #     allCrops = np.vstack(allCrops)
    #     prediction = model.predict(allCrops)

    #     sum = 0
    #     for i in range(len(prediction)):
    #         pieceNbr = i//4
    #         sideNbr = i%4
    #         sideInt = int(class_list[np.argmax(prediction[i])])

    #         self.pieces[pieceNbr].sideType[sideNbr] = sideInt
    #         sum += sideInt
        
    #     if sum != 0:
    #         raise Exception("sideSum = " + str(sum))





        

    # def getPieceWithRule(self,sideType):
    #     for p in self.pieces:
    #         match = True
    #         for i in range(4):
    #             if p.sideType[i] != sideType[i] and sideType[i] != None:
    #                 match = False
    #         if match == True:
    #             print("Match actually found")
    
    # def matchSurroundings(self,piece,grid,xpos,ypos,width,height):
    #     #inga sidor på insidan av puzzlet
    #     if xpos != 0 and piece.sideType[2] == 0:
    #         return False
    #     if xpos != width-1 and piece.sideType[0] == 0:
    #         return False
    #     if ypos != 0 and piece.sideType[1] == 0:
    #         return False
    #     if ypos != height-1 and piece.sideType[3] == 0:
    #         return False
        
    #     #sidorna ska vara sidor
    #     if xpos == 0 and piece.sideType[1] != 0:
    #         return False
    #     if xpos == width-1 and piece.sideType[0] != 0:
    #         return False
    #     if ypos == 0 and piece.sideType[2] != 0:
    #         return False
    #     if ypos == height-1 and piece.sideType[3] != 0:
    #         return False

    #     #se så att sidor passar i resten av puzzlet
    #     if ypos-1 > 0 and grid[ypos-1][xpos] != 0 and not piece.sideMatch(grid[ypos-1][xpos],3):
    #         return False
    #     if ypos+1 < height-1 and grid[ypos+1][xpos] != 0 and not piece.sideMatch(grid[ypos+1][xpos],1):
    #         return False
    #     if xpos-1 > 0 and grid[ypos][xpos-1] != 0 and not piece.sideMatch(grid[ypos][xpos-1],0):
    #         return False
    #     if xpos+1 < width-1 and grid[ypos][xpos+1] != 0 and not piece.sideMatch(grid[ypos][xpos+1],2):
    #         return False
    #     else:
    #         return True