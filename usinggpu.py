import tensorflow as tf
print('-------------------------------------------------')
print('built with cuda')
print(tf.test.is_built_with_cuda())
print('-------------------------------------------------')
print('gpu available')
print(tf.test.is_gpu_available())
print('-------------------------------------------------')
print('list devices')
print(tf.config.list_physical_devices('GPU'))
