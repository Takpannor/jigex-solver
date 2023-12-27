

import time
import pieceMover

class PileHandler:
    
    def __init__(self):
        self.piles = []

    def addPile(self,pile):
        self.piles.append(pile)

    #takes piece and finds smallest pile and returns the piles' coordinates for piecemove
    def distributePiece(self,piece,exceptionNbr = None):
        smallestPile = 1000
        currS = 0
        for i in range(len(self.piles)):
            if len(self.piles[i].pile) < smallestPile and (exceptionNbr != self.piles[i].nbr or exceptionNbr == None):
                smallestPile = len(self.piles[i].pile)
                currS = i
        self.piles[currS].addToPile(piece)
        pieceMover.pieceMover.pieceMove2(piece, self.piles[currS].coordinates[0], self.piles[currS].coordinates[1])
    
    def makePieceAvailable(self,piece):
        #print("looking for piece: " + str(piece.nbr))
        for pil in self.piles:
            if pil.pieceIsInPile(piece):
                while not pil.pieceIsOnTop(piece):
                    #print("moving pieces around")
                    p = pil.removeTopPiece()
                    #print("moving piece: " + str(p.nbr))
                    self.distributePiece(p,pil.nbr)
                return True
        raise Exception("couldnt make piece available")

    def removePiece(self,piece): #removes the top piece after it has matched etc.
        for pil in self.piles:
            for p in pil.pile:
                if p == piece:
                    pil.removeTopPiece()