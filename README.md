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



---

Config.py

Has all the values in it.

Values:



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

Renders everything, i guess? (I kinda want to change that but also dont know how too 🥲)