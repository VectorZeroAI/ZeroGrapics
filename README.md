# ZeroGrapics
A graphics engine, that allows for text based models to generate videos.

---

#Summary!

This graphics engine uses SQLite DB to store Vector styled data, and a coordinate system based graphics render.

---

#architecture:

ZeroInit.py

Spawns the custom SQLite DB with a custom struckture.

Struckture:

In table 1, each row represents a point, and all the colums are used to describe the point.
Colum1 has a vector of 3 numbers and represents the coordinates of point.
Colum2 has a UUID for the point.
Colum3 has a vector of all the UUIDs of the points to wich a line shall be drawn to.
Colum4 has the movements of the point in a json format.

The movements are saved this way:
{(x;y;[a;b;c]);...}
whereby:
x is the wait time in ms.
y is the duration of the change
[a;b;c] are the new coordinates of the point


TABLE2 --> LINES:

Each row describes a new line.

Colum 1: the UUID of the line
Colum 2: a vector of 2 uuids of the points wich are the lines endpoints. 
Colum 3: a vector of 3 numbers that are the coordinates of the pull point.
Colum 4: has a number that tells power of the pull point.
Colum 5: has a json representing changes to the lines. It is encoded that way:
{(x;y;[a;b;c;d]);...}
Whereby:
x is the waittime
y is the duration of the change.
[a;b;c] are the new coordinates of the pullpoint
d is the new pullpoint power


Table 3 --> Shapes

Colum 1: Is the vector of all the uuids of all the points in the shape.
Colum 2:Is the vector of all the uuids of all the lines in the shape.
Colum 3: Is the vector of 3 numbers representing the fill color.
Colum 4: is a json where the changes are saved.
They are saved this way:
{(x;y;[a;b;c]);...}
Whereby:
x is wait time
y is the duration of the change
[a;b;c are the new color representing numbers]


---

Config.py

Has all the values in it.

Values:
DB_NAME
CAMERA_POSITION
SHADERS
LINE_THIKNESS
LINE_COLOR


---

ZeroInterpriter.py

Exposes the funktions used to comunicate with the DB system.

Exposes funktions:

"construct_the_model"

"read.points"

"read.lines"

"read.shapes"

"read.movements.lines"

"read.movements.points"

"read.movements.shapes"

"read.full(tick_in_ms)"

read.full(tick_in_ms) returns the state that should be rendered in that exact ms. 

---

ZeroEngine.py

Uses Interpriter.py to read the DBs and renders the pictures.

Internaly constructs a Coordinate System where all the points are inputed, to compute the curved via pull point lines before giving them to rendering library.

The pull is calculated via this formula: 
B(t) = (1-t)^2 P_0 + 2(1-t) t P_c + t^2 P_1, \quad t \in [0,1]

and the amount of points is the distanse between the 3 points that form the curve.
---

Zerofiller.py

Gets a .txt or .json data from an LLM and fills the numbers into the SQLite DB.

Has a parser and a format defined.

FORMAT:
TABLE1:
ROW1:COLUM1:{x;y;z};COLUM2:{x};COLUM3:{x;y;...};COLUM4:{(x;y;[x;y;z]);...}
ROW2...
...

TABLE2:
ROW1:COLUM1:{x};COLUM2:{x;y};COLUM3:{x;y;z};COLUM4:{x};COLUM5:{(x;y;[a;b;c;d]);...}
ROW2...
...

TABLE3:
ROW1:COLUM1:{x;y;z;...};COLUM2:{x;y;z;...};COLUM3:{x;y;z};COLUM4:{(x;y;[a;b;c]);...}