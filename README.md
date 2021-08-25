# workshop-project

Goal: Given a reference sprite, animate it.

Our goal is acheieved by teaching the computer to do it. Specifically, we use DCGAN (might have to fall back to GAN).

INPUT:
1. Reference sprite of the animated character. Sprite should be 64x64 px.
2. Animation metadata:
  <br>*Type (Idle/Walk/Jump/Run)
  <br>*Frame count (up to 16)

OUTPUT:
<br>A series of images corresponding to the animation frames

Project Usage:
<br>In order to generate a sprite, run generate.py:
<p>python src/generate.py [path-to-reference] [animation-type] [animation-length] [output-folder]</p>

View Dataset:
<p>python src/database_handler.py</p>
Allows viewing the dataset, verifying its' contents, testing and generating label file.

Training:
<br>To begin training, simply run
<p>python src/gan_model.py</p>
(A label file is required)
