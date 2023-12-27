from PIL import Image, ImageGrab, ImageFilter,ImageMath
import ImageScanner
import numpy as np
import cv2 as cv
import pyautogui
#from matplotlib import pyplot as plt
import PieceCollection
import PuzzlePiece
import os
import copy
import pynput
import logging

#TODO:
# Chunk sometimes not being moved (possibly due to being too fast)
# Pieces sometimes wrongly being matched
# Chunk sometimes being moved into pile area (maybe piece is not removed from set?) (kanske att clickningar är för snabba) (förmodligen så märks inte den senaste matchningen)

#Mouse suppressor

if __name__ == '__main__':
    scan = ImageScanner.imagescanner()
    collection = PieceCollection.PieceCollection()
    nbr = 0

    #scan.TLcontour(1,False,(100,100))

    #scan.saveBackgroundScreenshot(0)

    collection = scan.photographEveryPiece(nbr)
    collection.saveCollection()
    print(len(collection.pieces))
    #collection.loadCollection()

    collection.setCenterForAllPieces()
    scan.spreadPiecesOut2(collection)
    collection.saveCollection()
    scan.solvePuzzle(collection,nbr)


































#def on_press(key):
#     print("trying to exit")
#     os._exit(1)

# def win32_event_filter(msg, data):
#     if msg == 512:
#         listener.suppress_event()

#     listener = pynput.mouse.Listener(
#                            win32_event_filter=win32_event_filter,
#                            suppress=False)

#     listener1 = pynput.keyboard.Listener(
#                             on_press=on_press
#     )
#     listener.start()
#     listener1.start()

#     listener.stop()
#     listener1.stop()
