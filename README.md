# jigex-solver

The program is intended for personal use only and will likely not function on other systems without additional effort. The code is therefore only posted on github as a complement to the demonstration.

Many functions are obsolete and may not be used at all during normal operation. However, most are out-commented at the end of each file in case of changes during development.

Program operation description: <br />
The program is split into 3 main separate processes: Scanning, Separation and Solving. These processes each have their own subprocesses which often tie into a loop for the main process. Descriptions below omit certain details for simplification.

The scanning process separates the puzzle pieces from the known background using a XOR operator and moves every piece into a clearing before saving an image of the piece, creating a PuzzlePiece object, using the image to find the corners, center and edges of the piece and moving the piece into a temporary storage pile below. The scanned pieces are objects in the PieceCollection object "pieces" list for the next process.

The separation process involves fuzzily matching every scanned edge against a classification of possible different sides. These are stored as a list of points in the sideTypeStandards folder. The matches are stored in order of the smallest sum of distances between the contour points of the piece edge and the standardized edges. Puzzle dimensions are then calculated using the amount of flat edges on each face, as pieces have a fixed rotation, in order to distribute pieces into piles at appropriate positions without disturbing the solving process later. The pieces are then distributed into the separate piles using the PileHandler and Pile objects.

The solving process constructs the puzzle using the previously stored matches for the current position and also for the surrounding pieces. Each position starts with a low tolerance for matchings, increasing the tolerance if no matches are found. Matches are detected from a visual response from the website which initiates placing the piece. If a needed piece is not topmost in a pile, the pieces above are moved into the smallest pile in order to minimize unnecessary delays.

The program exits after the puzzle is finished.
