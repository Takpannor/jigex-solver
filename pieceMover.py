import pyautogui as pg


class pieceMover:
    """Class containing static functions for moving pieces around"""
    pg.PAUSE = 0.08
    
    @staticmethod
    def pieceMove2(piece,xDest,yDest) -> None:
        """Moves center of piece to coordinates given by xDest and yDest and drops it."""

        pg.moveTo(piece.center[0][0]+ piece.topLeftPoint[0],piece.center[0][1]+piece.topLeftPoint[1])
        pieceMover.click()
        pieceMover.moveTo(xDest,yDest)
        xDiff = xDest - piece.center[0][0] - piece.topLeftPoint[0]
        yDiff = yDest - piece.center[0][1] - piece.topLeftPoint[1]
        piece.topLeftPoint = (piece.topLeftPoint[0]+xDiff,piece.topLeftPoint[1]+yDiff)
        pieceMover.click()

    @staticmethod
    def pieceMoveAndHover2(piece,xDest,yDest) -> None:
        """Moves center of piece to xDest and yDest still holding the piece.
        
        Args:
            piece (PuzzlePiece.PuzzlePiece): The piece that should be moved
            xDest (int): The x coordinate of the destination
            yDest (int): The y coordinate of the destination
        
        """

        pieceMover.moveTo(piece.center[0][0]+ piece.topLeftPoint[0],piece.center[0][1]+piece.topLeftPoint[1])
        pieceMover.click()
        pieceMover.moveTo(xDest,yDest)
        xDiff = xDest - piece.center[0][0] - piece.topLeftPoint[0]
        yDiff = yDest - piece.center[0][1] - piece.topLeftPoint[1]
        piece.topLeftPoint = (piece.topLeftPoint[0]+xDiff,piece.topLeftPoint[1]+yDiff)

    @staticmethod
    def pieceMoveAndDrop2(piece,xDest,yDest) -> None:
        """Moves the mouse with piece in hand to xDest and yDest and drops it."""

        pieceMover.moveTo(xDest,yDest)
        xDiff = xDest - piece.center[0][0] - piece.topLeftPoint[0]
        yDiff = yDest - piece.center[0][1] - piece.topLeftPoint[1]
        piece.topLeftPoint = (piece.topLeftPoint[0]+xDiff,piece.topLeftPoint[1]+yDiff)
        pieceMover.click()

    @staticmethod
    def pieceMove(piece, xCurr, yCurr, xDest,yDest) -> None:
        """Moves mouse to xCurr, yCurr, picks up piece there and 
        moves mouse to xDest and yDest and drops piece."""

        pieceMover.moveTo(xCurr,yCurr)
        pieceMover.click()
        pieceMover.moveTo(xDest,yDest)
        xDiff = xDest - xCurr
        yDiff = yDest - yCurr
        x,y = piece.topLeftPoint
        piece.topLeftPoint = (x+xDiff,y+yDiff)
        pieceMover.click()

    @staticmethod
    def click() -> None:
        """Clicks once at current position"""

        pg.click()
    
    @staticmethod
    def moveTo(x,y) -> None:
        """ Moves the cursor to the position given by the parameters x and y
        
        Parameters:
            x (int): The x-coordinate of the destination
            y (int): The y-coordinate of the destination
        
        """

        pg.moveTo(x,y)

