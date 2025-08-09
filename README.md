# ZeroGrapics
A small graphics engine, that allows for text based models to generate videos.

---

#Summary!

This graphics engine uses SQLite DB to store Vector styled data, and a coordinate system based graphics render.

---

#architecture:

Init.py

Spawns the custom SQLite DB with a custom struckture.

Struckture:

TABLE1:
ROW1: COLUM1: {x;y;z};COLUM2: {x};COLUM3:{x;y;z;...};COLUM4: {[x;y;z];[a;b;c];...}
ROW2:...
ROW...

TABLE2:
ROW1: COLUM1: {x;y};COLUM2: {x;y;z;...};COLUM3: {x};COLUM4: {x;y;z};COLUM5: {x};COLUM6: {x;y;z;...};COLUM7: {[x;y([x;y];[x;y;z])];...}
ROW...

TABLE3:
ROW1:COLUM1:{x};COLUM2:{x;y};COLUM3:{x;y;z};COLUM4:{x};COLUM5:{(x;y;[{x;y;z};{x};{x};{x;y;z}]);...};


In table 1, each row represents a point, and all the colums are used to describe the point. 
Colum1 has a vector of 3 numbers and represents the coordinates of point.
Colum2 has a UUID for the point.
Colum3 has a vector of all the UUIDs of the points to wich a line shall be drawn.
Colum4 has the movements of the node encrypted in it via a list of vectors, each having 3 numbers and representing one movement.


TABLE2 --> FACES IS NOT YET READY! IGNORE FOR NOW!

In table 2, each row represents a face, and all the colums are used to describe the shape.
Colum1 has a vector with 2 numbers in it, and they describe in wich direction the shape is facing.
Colum2 has a vector 
Colum3 has a binary in it, that decribes whether the face is a figure with points as edges of it, or custom fluidly tunable figure. 0 means a fill out, meaning that colums 5 and 6 are ignored.



Table 3, lines.

Each line is asigned a UUID, and it is saved in colum 1. 
Colum 2 has a vector of 2 numbers, wich are UUIDs of points that the line is drawn from and to.
Colum 3 has a vector describing the coordinates of the "pull point" of the line.
Colum 4 describes how hard the point pulls.
Colum 5 describes the changes. It does so in this way: {[time before the change;the duration of change;the new values for changeable parameters];[...]...}
Colum 6 has the thickness of line as one number
Colum 7 has the color of line in a vector of 3 numbers.



---

Config.py

Has all the values in it.

Values:
CAMERA_POSITION
SHADERS


---

Interpriter.py

Exposes the funktions used to comunicate with the DB system.

Exposes funktions:

"read.points"

"read.lines"

"read.shapes"

"read.movements.lines"

"read.movements.points"

"read.movements.shapes"

---

Engine.py

Uses Interpriter.py to read the DBs and renders the pictures.