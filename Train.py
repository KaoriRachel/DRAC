import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from keras.src.applications import InceptionV3
# from keras.src.applications.mobilenet_v3 import MobileNetV3
from sklearn.utils import class_weight
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from sklearn.model_selection import train_test_split
# from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50, InceptionV3, Xception
# from sklearn.metrics import cohen_kappa_score


# 加载模型
model = load_model('best_model.h5')

# 配置参数
train_image_folder_path = 'Training Set'  # 训练集图片文件夹路径
train_csv_path = 'DRAC2022_ Image Quality Assessment_Training Labels.csv'  # 训练集标签CSV文件路径
test_image_folder_path = 'Testing Set'  # 测试集图片文件夹路径
img_width, img_height = 224, 224
batch_size0 = 5
batch_size1 = 10
batch_size2 = 52
batch_size = batch_size0 + batch_size1 + batch_size2  # 批量大小
epochs = 20  # 训练轮数

# 读取训练集标签CSV文件
train_labels = pd.read_csv(train_csv_path, dtype=str)
# train_labels['image name'] = train_labels['image name'].apply(lambda x1: x1) #  train_image_folder_path + '/' +

# train_labels 是所有训练数据的DataFrame
train_labels, valid_labels = train_test_split(train_labels, test_size=0.2,
                                              stratify=train_labels['image quality level'], random_state=42)
print(train_labels.shape, valid_labels.shape)
print(train_labels.iloc[0, 0])

# 指定所有可能的类别标签
classes = ['0', '1', '2']

# 类别为2的图像数据生成器
datagen_2_augmentation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='wrap'
)

# 类别为1的图像数据生成器
datagen_1_augmentation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='wrap'
)

# 类别为0的图像数据生成器
datagen_0_augmentation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='wrap'
)

mult11, mult01 = 5, 10
# 创建分开的生成器
train_generator_2_aug = datagen_2_augmentation.flow_from_dataframe(
    dataframe=train_labels[train_labels['image quality level'] == '2'],
    directory=train_image_folder_path,
    x_col='image name',
    y_col='image quality level',
    target_size=(img_width, img_height),
    batch_size=batch_size2,
    class_mode='categorical',
    classes=classes,  # 指定所有类别
    shuffle=True
)

train_generator_1_augs = []
for _ in range(mult11):
    train_generator_1_aug = datagen_1_augmentation.flow_from_dataframe(
        dataframe=train_labels[train_labels['image quality level'] == '1'],
        directory=train_image_folder_path,
        x_col='image name',
        y_col='image quality level',
        target_size=(img_width, img_height),
        batch_size=batch_size1,
        class_mode='categorical',
        classes=classes,  # 指定所有类别
        shuffle=True
    )
    train_generator_1_augs.append(train_generator_1_aug)

train_generator_0_augs = []
for _ in range(mult01):
    train_generator_0_aug = datagen_0_augmentation.flow_from_dataframe(
        dataframe=train_labels[train_labels['image quality level'] == '0'],
        directory=train_image_folder_path,
        x_col='image name',
        y_col='image quality level',
        target_size=(img_width, img_height),
        batch_size=batch_size0,
        class_mode='categorical',
        classes=classes,  # 指定所有类别
        shuffle=True
    )
    train_generator_0_augs.append(train_generator_0_aug)


def mixed_generator(gen2, gen1, gen0, mult2=1, mult1=5, mult0=10):
    while True:
        # x1, y1 = gen2.next()
        # x1_list, y1_list = [x1], [y1]  # 类别2不进行额外增强

        # 类别2增强
        x2, y2 = gen2.next()
        x1_list, y1_list = [x2], [y2]
        for i in range(mult2 - 1):  # 已有1次, 需要额外增强
            x_temp, y_temp = gen2[i].next()
            x1_list.append(x_temp)
            y1_list.append(y_temp)

        # 类别1增强2倍
        for i in range(mult1 - 1):
            x_temp, y_temp = gen1[i].next()
            x1_list.append(x_temp)
            y1_list.append(y_temp)

        # 类别0增强5倍
        for i in range(mult0 - 1):
            x_temp, y_temp = gen0[i].next()
            x1_list.append(x_temp)
            y1_list.append(y_temp)

        x_combined = np.concatenate(x1_list, axis=0)
        y_combined = np.concatenate(y1_list, axis=0)
        indices = np.arange(len(x_combined))
        np.random.shuffle(indices)
        yield x_combined[indices], y_combined[indices]


# 合并后的训练生成器
train_generator = mixed_generator(train_generator_2_aug, train_generator_1_augs, train_generator_0_augs)

# 验证集
valid_datagen = ImageDataGenerator(rescale=1./255)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_labels,
    directory=train_image_folder_path,
    x_col='image name',
    y_col='image quality level',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=classes,
    shuffle=False
)

# 测试集数据生成器
test_datagen = ImageDataGenerator(rescale=1./255)

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
'''
# 创建预训练模型
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# 冻结预训练模型的权重
for layer in base_model.layers:
    layer.trainable = False

# 添加自定义层
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)  # 3个质量等级

# 构建完整模型
model = Model(inputs=base_model.input, outputs=predictions)
'''
# 编译模型
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])

# 总训练数据量
total_train_data = len(train_labels[train_labels['image quality level'].isin(['0', '1', '2'])])

# 计算每个 epoch 的步骤数
steps_per_epoch = total_train_data // batch_size

# 计算验证数据量
total_valid_data = len(valid_labels)

# 计算验证集每个 epoch 的步骤数
validation_steps = total_valid_data // batch_size

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_labels['image quality level']),
    y=train_labels['image quality level'].values
)
class_weights_dict = dict(enumerate(class_weights))

# 训练模型
history = model.fit_generator(
    train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch+1,
    validation_data=valid_generator,
    validation_steps=validation_steps,
    verbose=1,
    class_weight=class_weights_dict
)
print(history.history)
print(history.epoch)


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


# 将最终结果保存到CSV文件
final_results.to_csv('final_test_quality_predictions.csv', index=False)

# 绘制训练历史曲线
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# 保存模型
#model.save('best_model.h5')  # 保存为 HDF5 文件
