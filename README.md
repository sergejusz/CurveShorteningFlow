Python scripts for modelling of curve shortening flow (CSF).

Now most of for-loops refactored to use numpy arrays operations.

Command line examples:



Create folder for output images, for example, "c:\\Work\\images\\output".

extract curves from images

py shorten\_curve.py data\\images\\five\_curves.png c:\\Work\\images\\output -i 1000 -m3 -p -s10 -n5 -v solid -c red

py shorten\_curve.py data\\images\\five\_curves.png c:\\Work\\images\\output -i 1000 -m3 -p -s10 -n5 -v vector --bg\_color "5089a8" --fg\_color "ffffcc"

py shorten\_curve.py data\\images\\five\_curves.png c:\\Work\\images\\output -i 1000 -m3 -p -s10 -n5 -v contour --bg\_color "5089a8" --fg\_color "ffffcc"

py shorten\_curve.py data\\images\\five\_curves.png c:\\Work\\images\\output -i 1000 -m3 -p -s10 -n5 -v history --bg\_color "5089a8" --fg\_color "ffffcc"

py shorten\_curve.py data\\images\\curve13.png c:\\Work\\images\\output -i 1000 -p -s10 -v history --bg\_color "5089a8" --fg\_color "ffffcc"

py shorten\_curve.py data\\images\\curve13.png c:\\Work\\images\\output -i 1000 -p -s10 -v vector --bg\_color "5089a8" --fg\_color "ffffcc"

py shorten\_curve.py data\\images\\curve12.png c:\\Work\\images\\output -i 1000 -s10 -v solid -c blue

py shorten\_curve.py data\\images\\curve14.png c:\\Work\\images\\output -i 1000 -p -s10 -v vector --bg\_color "5089a8" --fg\_color "ffffcc"

py shorten\_curve.py data\\images\\curve11.png c:\\Work\\images\\output -i 1000 -p -s10 -v vector

py shorten\_curve.py data\\images\\test\_6.png c:\\Work\\images\\output -i 1000 -m5 -p -s10 -n5 -v history --bg\_color "5089a8" --fg\_color "ffffcc"

py shorten\_curve.py data\\images\\test\_6.png c:\\Work\\images\\output -i 1000 -m5 -p -s10 -n5 -v vector --bg\_color "5089a8" --fg\_color "ffffcc"

py shorten\_curve.py data\\images\\test\_2.png c:\\Work\\images\\output -i 100 -m3 -p -s1 -n2 -v solid -c red

py shorten\_curve.py data\\images\\embed1.png c:\\Work\\images\\output -i 100 -m5 -p -s1 -n4 -v solid -c green



generate geometric figures

py shorten\_curve.py circle c:\\Work\\images\\output -i 10 -p -s1 -v vector --radius 100

py shorten\_curve.py ellipse c:\\Work\\images\\output -i 10 -p -s1 -v vector --radius\_x 100 --radius\_y 60

py shorten\_curve.py paperclip c:\\Work\\images\\output -i 10 -p -s1 -v vector --radius 100

py shorten\_curve.py rectangle c:\\Work\\images\\output -i 100 -p -s1 -v vector --side\_x 100 --side\_y 60



py shorten\_curve.py EIGHT\_SHAPE c:\\Work\\images\\output -i 4 -s 1 -v vector --num\_points 600 --height 240 --width 300 -p --radius 50 --bg\_color "0000ff" --fg\_color "ffffcc"

py shorten\_curve.py TOUCH\_EIGHT\_SHAPE c:\\Work\\images\\output -i 4 -s 1 -v vector --num\_points 600 --height 240 --width 300 -p --radius 50 --bg\_color "0000ff" --fg\_color "ffffcc"

py shorten\_curve.py LISSAJOUX c:\\Work\\images\\output -i 2 -s 1 -v vector --num\_points 1000 --radius\_x 120 --radius\_y 90 --freq\_x 3  --freq\_y 2 --height 240 --width 300 -p  --bg\_color "0000ff" --fg\_color "ffffcc"

