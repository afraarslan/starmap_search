#  Locating a Defined Area in the Star Map

It is a python program that finds the location of the given area in the Star Map.

Run the program as follows:

    python solution.py --small-image-path Small_area.png --starmap-path StarMap.png

It takes two input:
1. One of the small images
2. The Star Map

The output is followed:
* If it finds the match:

It says 'found enough match' and returns the angle and the corner points of the small image onto Star Map. 

![](https://github.com/afraarslan/starmap_search/blob/master/screenshots/corner-points.png)

![](https://github.com/afraarslan/starmap_search/blob/master/screenshots/matched.png)

* If there is no match:

It says 'Not enough match found'

![](https://github.com/afraarslan/starmap_search/blob/master/screenshots/not-found.png)

![](https://github.com/afraarslan/starmap_search/blob/master/screenshots/not-matched.png)

