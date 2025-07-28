Python scripts for modelling of curve shortening flow (CSF).

Now most of for-loops refactored to use numpy arrays operations.

Command line examples:

Original image file is c:\\work\\images\\test1.png
Output images will be stored in c:\\work\\images\\test1 (should exist!)
-i50 (--iterations 50) is for max. 50 iterations.
-s1 (--save\_every\_n 1) is for saving image at every iteration.
-p (--preserve\_area) is for area preserving.
-n3 (--number\_curves 3) is for extracting 3 curves from original image file.
-v contour (--view contour) is for displaying curves together with vectors of flow.

py shorten\_curve.py c:\\work\\images\\test1.png c:\\work\\images\\test1 -i50 -s1 -p -n3 -v contour
py shorten\_curve.py c:\\work\\images\\test1.png c:\\work\\images\\test1 --iterations 50 --save\_every\_n 1 --preserve\_area --number\_curves 3 --view contour

py shorten\_curve.py c:\\work\\images\\test1.png c:\\work\\images\\test1 -i50 -s1 -p -n3 -v solid
py shorten\_curve.py c:\\work\\images\\test1.png c:\\work\\images\\test1 --iterations 50 --save\_every\_n 1 --preserve\_area --number\_curves 3 --view solid

-c red (--color\_palette red) is for using red colors to fill curves.
py shorten\_curve.py c:\\work\\images\\test1.png c:\\work\\images\\test1 -i50 -s1 -p -n3 -v solid -c red
py shorten\_curve.py c:\\work\\images\\test1.png c:\\work\\images\\test1 --iterations 50 --save\_every\_n 1 --preserve\_area --number\_curves 3 --view solid --color\_palette red

py shorten\_curve.py c:\\work\\images\\test1.png c:\\work\\images\\test1 -i50 -m3 -s1 -n3 -v vector
py shorten\_curve.py c:\\work\\images\\test1.png c:\\work\\images\\test1 --iterations 50 --median\_filter 3 --save\_every\_n 1 --number\_curves 3 --view vector

