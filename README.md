# tissue-segmentation-net

Code to segment DAPI tissue images using unet.

Development on windows might create issues with LF vs CRLF

Ensure when using compute canada that:
       - the IMG_SIZE is changed to 1024 in finetuning.py
       - the multiprocessing settings are True (or remove evaluation?)
       - switch in main.py to using the compute canada paths
       - don't push from that account
       - script must activate python env and import modules before running program
       