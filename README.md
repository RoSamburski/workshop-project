# workshop-project

Goal: Given a reference sprite, animate it.

Our goal is acheieved by teaching the computer to do it. Specifically, we use DCGAN (might have to fall back to GAN).

INPUT:
1. Reference sprite of the animated character. Sprite should be 64x64 px.
2. Animation metadata:
  *Type (Idle/Walk/Jump)
  *Direction (Left/Right)
  *Frame count (up to 16)

OUTPUT:
A series of images corresponding to the animation frames

Project Usage (currently):
1. Run train.py
2. Run generate.py on train.py's output and the desired sprite.

NOTES:
Training requires about 5.7-6 GB RAM to run
