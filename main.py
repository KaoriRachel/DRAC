import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, VGG19, ResNet50, ResNet152, EfficientNetV2B0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# 配置参数"
model_name = "my_model.keras"
train_image_folder_path = 'Training Set'  # 训练集图片文件夹路径
train_csv_path = 'DRAC2022_ Image Quality Assessment_Training Labels.csv'  # 训练集标签CSV文件路径
test_image_folder_path = 'Testing Set'  # 测试集图片文件夹路径
img_width, img_height = 512, 512  # EfficientNetV2的默认输入图片尺寸
batch_size = 32  # 批量大小
epochs = 10  # 训练轮数

# 读取训练集标签CSV文件
train_labels = pd.read_csv(train_csv_path, dtype=str)


# train_labels['image name'] = train_labels['image name'].apply(lambda x1: x1) #  train_image_folder_path + '/' +

def color_jitter(image):
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image


# 训练集数据生成器
train_datagen = ImageDataGenerator(
    preprocessing_function=color_jitter,
    rescale=1. / 255,
    rotation_range=30,
    vertical_flip=True,
    horizontal_flip=True,
    fill_mode='constant'
)

# train_images = pd.DataFrame({'image name': os.listdir(train_image_folder_path)})
# train_images['image name'] = train_images['image name'].apply(lambda x2: os.path.join(train_image_folder_path, x2))

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_labels,
    directory=train_image_folder_path,
    x_col='image name',  # CSV中存储图片文件名的列名
    y_col='image quality level',  # CSV中存储质量等级的列名
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# 测试集数据生成器
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_images = pd.DataFrame({'image name': os.listdir(test_image_folder_path)})
# test_images['image name'] = test_images['image name'].apply(lambda x2: os.path.join(test_image_folder_path, x2))

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_images,
    directory=test_image_folder_path,
    x_col='image name',
    y_col=None,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)

# 创建MobileNetV2预训练模型
base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# 冻结预训练模型的权重
for layer in base_model.layers:
    layer.trainable = False

# 添加自定义层
x = base_model.output
x = Flatten()(x)
predictions = Dense(3, activation='softmax')(x)  # 3个质量等级

# 构建完整模型
if os.path.isfile('./EfficientModel512.h5'):
    model = load_model('./EfficientModel512.h5')
else:
    model = Model(inputs=base_model.input, outputs=predictions)
    # 编译模型
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(
    train_generator,
    epochs=epochs,
    verbose=1
)

model.save('./EfficientModel512.h5')

# 使用模型进行预测
predictions = model.predict(test_generator, verbose=1)

# 计算最终预测类别
final_classes = np.argmax(predictions, axis=1)

# 创建最终结果DataFrame
final_results = pd.DataFrame({
    'case': [os.path.basename(x) for x in test_generator.filenames],
    'class': final_classes,
    'P0': predictions[:, 0],
    'P1': predictions[:, 1],
    'P2': predictions[:, 2]
})

# 确保概率之和为1
# assert np.allclose(final_results[['P0', 'P1', 'P2']].sum(axis=1), 1), "The probabilities do not sum up to 1."

# 将最终结果保存到CSV文件
final_results.to_csv('final_test_quality_predictions.csv', index=False)