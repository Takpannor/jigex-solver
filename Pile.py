

class Pile:
    number = 0

    def __init__(self, coordinates):
        self.coordinates = coordinates #x,y för bitcentrum
        self.pile = []
        self.nbr = Pile.number
        Pile.number +=1
    
    def addToPile(self,piece):
        self.pile.insert(0,piece)

    #removes top piece in list and returns it
    def removeTopPiece(self):
        return self.pile.pop(0)
    
    def pieceIsOnTop(self,piece):
        if self.pile[0] == piece:
            return True
        else:
            return False
        
    def pieceIsInPile(self,piece):
        for p in self.pile:
            if piece == p:
                return True
        return False