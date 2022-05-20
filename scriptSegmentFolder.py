from predict import segment_folder

model = "/media/lucas/DATA/POSDOC_Calgary/Lightsheet/Image-to-Image/Annotations/history_only_good/unet20220516/unet4levels_num_82_normtype_divide/unet4levels_num_82_normtype_divide.h5"
frame_path =    "/media/lucas/DATA/POSDOC_Calgary/Lightsheet/Image-to-Image/Annotations/Training/Original_192/Good_192"
output_folder = "/media/lucas/DATA/POSDOC_Calgary/Lightsheet/Image-to-Image/Annotations/history_only_good/Training/Good_192"
segment_folder(model, frame_path, output_folder, norm_type='divide')
