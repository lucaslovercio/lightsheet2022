# tissue-segmentation-net

Code to segment DAPI tissue images using unet.

Development on windows might create issues with LF vs CRLF

## Ensure when using compute canada that:

1) The multiprocessing settings are True (or remove evaluation?)

2) Switch in main.py to using the compute canada paths

3) Don't push from that account

4) Script must activate python env and import modules before running program

## Dependencies

1) tensorflow

2) keras

3) imgaug

4) opencv-python

5) numpy
