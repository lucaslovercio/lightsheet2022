from predict import segment_folder

model = "/media/lucas/DATA/POSDOC_Calgary/Lightsheet/lightsheet2021/Training_Validation_Full/LoVercio_unetMini_num_0_normtype_divide_E95_E105.h5"
frame_path =    "/media/lucas/DATOS3/Lightsheet/Volumes/2021/Feb3_E11_4_v2/VolumeDAPI_Tiles_ForTissue_Brighter_256"
output_folder = "/media/lucas/DATOS3/Lightsheet/Volumes/2021/Feb3_E11_4_v2/VolumeDAPI_Tiles_ForTissue_Brighter_256_processed"
segment_folder(model, frame_path, output_folder, norm_type='divide')