Python scripts for modelling of curve shortening flow (CSF).

Now most of for-loops refactored to use numpy arrays operations.

Command line examples:

Original image file is c:\work\images\test1.png
Output images will be stored in c:\work\images\test1 (should exist!)
-i50 (--iterations 50) is for max. 50 iterations.
-s1 (--save_every_n 1) is for saving image at every iteration.
-p (--preserve_length) is for length preserving.
-n3 (--number_curves 3) is for extracting 3 curves from original image file.
-v contour (--view contour) is for displaying curves together with vectors of flow.

py shorten_curve.py c:\work\images\test1.png c:\work\images\test1 -i50 -s1 -p -n3 -v contour
py shorten_curve.py c:\work\images\test1.png c:\work\images\test1 --iterations 50 --save_every_n 1 --preserve_length --number_curves 3 --view contour

py shorten_curve.py c:\work\images\test1.png c:\work\images\test1 -i50 -s1 -p -n3 -v solid
py shorten_curve.py c:\work\images\test1.png c:\work\images\test1 --iterations 50 --save_every_n 1 --preserve_length --number_curves 3 --view solid

-c red (--color_palette red) is for using red colors to fill curves.
py shorten_curve.py c:\work\images\test1.png c:\work\images\test1 -i50 -s1 -p -n3 -v solid -c red
py shorten_curve.py c:\work\images\test1.png c:\work\images\test1 --iterations 50 --save_every_n 1 --preserve_length --number_curves 3 --view solid --color_palette red

py shorten_curve.py c:\work\images\test1.png c:\work\images\test1 -i50 -m3 -s1 -n3 -v vector
py shorten_curve.py c:\work\images\test1.png c:\work\images\test1 --iterations 50 --median_filter 3 --save_every_n 1 --number_curves 3 --view vector
