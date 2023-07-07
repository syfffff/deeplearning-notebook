

# 改变工作目录

在vscode里的工作目录和网页版linux虚拟机不一致所以需要改变工作路径

```python
import os
os.chdir("/shangyufei" )
print(os.getcwd())
```



# csv处理

## 1.读取csv

```python

import pandas as pd
train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test.csv')
train_csv.head()
```

train_csv.head()是pandas中的一个函数，用于显示DataFrame的前几行，默认显示前5行。这个函数通常用于数据集的快速预览，以便了解数据的大致结构和内容。

|      |        image |      label       |
| ---: | -----------: | :--------------: |
|    0 | images/0.jpg | maclura_pomifera |
|    1 | images/1.jpg | maclura_pomifera |
|    2 | images/2.jpg | maclura_pomifera |
|    3 | images/3.jpg | maclura_pomifera |
|    4 | images/4.jpg | maclura_pomifera |

图片与label一一对应时，eg 图片1-dog  图片2-cat

```python
def read_csv_labels(fname):
    """读取fname来给标签字典返回一个文件名"""
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# 训练样本 :', len(labels))
print('# 类别 :', len(set(labels.values())))
```

1. 定义函数：定义一个名为read_csv_labels的函数，该函数接受一个文件名作为参数。
2. 打开文件：使用Python内置函数open()打开指定的CSV文件，并以只读模式打开。
3. 跳过文件头行：使用readlines()函数读取文件中的所有行，并将第二行至最后一行存储在变量lines中。由于第一行为文件头行，包含列名，因此需要使用Python的列表切片语法(lines[1:])跳过该行。
4. 解析数据：使用列表解析语法(tokens = [l.rstrip().split(',') for l in lines])将每一行数据解析为一个元组，其中第一个元素为文件名，第二个元素为标签。
5. 存储数据：使用Python的字典解析语法(dict(((name, label) for name, label in tokens)))将解析后的数据存储在一个字典中，其中文件名为键，标签为值。
6. 返回结果：将存储标签信息的字典作为函数的返回值返回。
7. 输出结果：调用read_csv_labels函数读取CSV文件中的标签信息，并使用Python的内置函数len()和set()分别计算训练样本数和类别数，并将结果输出到控制台。
8. `labels.values()` 返回字典中所有的值，即所有的标签信息，这些标签信息被组成一个列表。
9. `set(labels.values())` 将这个列表转换成一个集合，由于集合是无序且不重复的，因此它可以去除列表中的重复元素，只保留不同的标签。比如，如果标签列表是 ['cat', 'dog', 'cat', 'bird', 'dog']，那么转换成集合后就是 {'cat', 'dog', 'bird'}。

{'1': 'frog', '2': 'truck', '3': 'truck', '4': 'deer', '5': 'automobile', '6': 'automobile', '7': 'bird', '8': 'horse', '9': 'ship', '10': 'cat', '11': 'deer', '12': 'horse', '13': 'horse', '14': 'bird', '15': 'truck', '16': 'truck', '17': 'truck', '18': 'cat',

实现了将图片与标签对应的字典

## 2.分割csv

```
train_csv.loc[:, 'label']

```

train_csv.loc[:, 'label']的意思是选择train_csv表中的所有行（:表示所有行），并且选择label列。这是pandas中的一种用法，可以通过loc方法来选择数据集的特定行和列。这里的“:”表示选择所有的行，而“label”表示指定选择的列名。

## 3.查看有多少种类别并且给予标号



```python
train_csv.loc[:, 'label'].unique()#等价于train_csv.label.unique()
```

选择train_csv表中的所有行，然后选择label列，并使用unique()方法获取该列的唯一值。这个方法会返回一个包含该列所有唯一值的numpy数组。在这个例子中，它返回的是训练集中所有不同的标签值（labels）

```python
class_to_num = dict(zip(list(train_csv.loc[:, 'label'].unique()), range(len(train_csv.label.unique())) ))
print(class_to_num)
print(len(class_to_num))
```

len(train_csv.label.unique())为176 也就是要分类176类

list(train_csv.loc[:, 'label'].unique())将包含唯一标签的numpy数组转换为列表，这个列表中包含训练集中所有不同的标签值（labels）。range(len(train_csv.label.unique()))生成一个从0到标签数量-1的数字序列，这里的标签数量是通过train_csv.label.unique()得到的。

使用zip函数将标签字符串和数字序列一一对应，得到一个字典。这个字典中，标签字符串是键，数字序列是值，可以将标签字符串映射为数字，方便后续的数据处理和模型训练。

````
a = [1, 2, 3]
b = ['a', 'b', 'c']
zipped = zip(a, b)
```
此时，zipped是一个zip对象，可以使用list()函数将其转换为一个包含元组的列表：

```
[(1, 'a'), (2, 'b'), (3, 'c')]
````

```
`{'maclura_pomifera': 0, 'ulmus_rubra': 1, 'broussonettia_papyrifera': 2, 'prunus_virginiana': 3, 'acer_rubrum': 4,`
```

# 图片读取

## 4.图片路径

```python
import os
#path='\images'

image1_path = os.path.join( train_csv.loc[0, 'image'])
print(image1_path)
```

​        images/0.jpg

这段代码的作用是构造训练集中第一张图像的完整文件路径。具体来说，代码中首先定义了一个变量path，表示数据集所在的目录路径。然后，使用os.path.join()函数将path和训练集中第一张图像的文件名拼接起来，得到图像文件的完整路径。

在代码中，train.loc[0, 'image']表示训练集中第一行数据的'image'列，即第一张图像的文件名。os.path.join(path, train.loc[0, 'image'])将path和文件名拼接起来，得到形如'/kaggle/input/classify-leaves/train/100001.jpg'的完整文件路径，保存在image1_path变量中。

os.path.join()是Python内置的用于路径拼接的函数，可以将多个路径组合成一个完整的路径。

当path为空时候，默认为本身文件夹路径

## 5显示图片

```python
import matplotlib.pyplot as plt
from PIL import Image

image1 = Image.open(image1_path)
plt.imshow(image1)
plt.title(train.loc[0, 'label'])
plt.axis('off')
plt.show()
```

这段代码的作用是使用matplotlib库和PIL库显示训练集中第一张图像。具体来说，代码中首先使用PIL库的Image.open()函数读取指定路径中的图像文件，得到一个PIL图像对象image1。

然后，用matplotlib库的plt.imshow()函数显示这个图像。plt.imshow()函数将图像数据显示为图像，并可以添加标题、轴标签等元素。在这个例子中，使用plt.imshow()函数显示image1图像，然后使用plt.title()函数添加图像标题，使用plt.axis()函数设置坐标轴的显示方式，最后使用plt.show()函数显示图像。

这个代码可以用于检查图像文件是否能够正常读取，以及查看图像的内容和标签信息。在实际的项目中，通常需要使用类似的代码对数据集中的所有图像进行遍历和处理。

## 6.图片转tensor

```python
import torchvision.transforms as transform
img2tensor = transform.ToTensor()
print(img2tensor(image1).shape)

torch.Size([3, 224, 224])
```

这段代码的作用是将PIL图像对象转换为PyTorch张量（Tensor）对象，并检查张量的维度。

## 7.对字典 key 和value 颠倒

```python
num_to_class = { a : b for b, a in class_to_num.items()}
print(num_to_class)
```

```
0: 'maclura_pomifera', 1: 'ulmus_rubra', 2: 'broussonettia_papyrifera', 3: 'prunus_virginiana', 4: 'acer_rubrum', 5: 'cryptomeria_japonica', 6: 'staphylea_trifolia',
```

class_to_num.items()是一个字典方法，用于返回字典中的所有键值对。该方法返回一个类似于列表的可迭代对象，其中每个元素都是一个键值对，表示字典中的一个条目。

例如，如果有一个字典class_to_num，其值为{'A': 0, 'B': 1, 'C': 2}，则可以使用class_to_num.items()方法返回这个字典的所有键值对：

```
for key, value in class_to_num.items():
    print(key, value)
```

这个代码会依次输出字典中的每个键值对，即：

```
A 0
B 1
C 2
```

# 数据集创建

## 8创建dataset

```python
class LeavesDataset(Data.Dataset):
    """
    args:
    csv_path: indicates the path of csv,
    file_path: indicates where the images loads,
    mode: choose between 'train', 'valid' and 'test'. default set 'train'.
    valid_ratio: indicates the length of valid dataset. default 0.2.
    resize_height: indicates how to resize the images. default 256.
    resize_width: indicates how to resize the images. default 256.
    """
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.file_path = file_path
        self.mode = mode
        self.data_info = pd.read_csv(csv_path)
        self.data_len = len(self.data_info.index)
        self.train_len = int(self.data_len * (1 - valid_ratio))
        
        if mode == 'train':
            self.train_image = np.asarray(self.data_info.loc[:self.train_len, 'image'])
            self.train_label = np.asarray(self.data_info.loc[:self.train_len, 'label'])
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.loc[self.train_len:, 'image'])
            self.valid_label = np.asarray(self.data_info.loc[self.train_len:, 'label'])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.loc[:, 'image'])
            self.image_arr = self.test_image
            
        self.real_len = len(self.image_arr)
        
        print('Finished reading %s dataset. %d number samples found.' % (mode, self.real_len))
        
        
    def __getitem__(self, index):
        image_path = self.image_arr[index]
        image = Image.open(os.path.join(self.file_path, image_path))
        
        if self.mode == 'train':
            transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            transform = transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            
        image = transform(image)
        
        if self.mode == 'test':
            return image
        
        label = self.label_arr[index]
        label_num = class_to_num[label]
        return image, label_num
    
    
    def __len__(self):
        return self.real_len
```

创建自己的数据集，这是一个用于加载叶子图像数据集的 PyTorch 数据集类，用于将数据集处理成 PyTorch 中的数据集类型。该类的初始化函数接收以下参数：

- csv_path：表示 CSV 文件的路径，其中包含数据集中每个图像的信息及其对应的标签。
- file_path：表示包含图像文件的文件夹路径。
- mode：表示数据集的模式，可以是 'train'、'valid' 或 'test'，默认为 'train'。
- valid_ratio：表示用于验证集的数据的比例，默认为 0.2。
- resize_height：表示将每个图像调整为的高度，默认为 256。
- resize_width：表示将每个图像调整为的宽度，默认为 256。

该类中包含一个 __getitem__ 函数，用于从数据集中获取单个图像及其标签。在训练模式下，该函数会对图像进行一系列的数据增强操作，如随机水平翻转、随机

## 9.dataset实例化

```python
train_path = 'train.csv'
test_path = 'test.csv'
image_path = ''

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

train_dataset = LeavesDataset(train_path, image_path, mode='train', valid_ratio=0.2)
valid_dataset = LeavesDataset(train_path, image_path, mode='valid', valid_ratio=0.2)
test_dataset = LeavesDataset(test_path, image_path, mode='test')
```

创建数据集

```
Finished reading train dataset. 14683 number samples found.
Finished reading valid dataset. 3671 number samples found.
Finished reading test dataset. 8800 number samples found.
```

## 附：torchvision.datasets.ImageFolder创建数据集

数据集必然得按照API的要求去组织， torchvision.datasets.ImageFolder 要求数据集按照如下方式组织：

A generic data loader where the images are arranged in this way:

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
    
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png


```python
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))

test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
```



```python
train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs),batch_size=batch_size, shuffle=True)
```

### torchvision.datasets.ImageFolder 实例

先看数据集组织结构：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191203110354109.png#pic_center)

即根目录为 “./data/train/”，根目录下有三个类别文件夹，即Snowdrop、LilyValley、Daffodil，每个类别文件夹下有80个训练样本。

```python
import torchvision

dataset = torchvision.datasets.ImageFolder('./data/train/') # 不做transform
print(dataset.classes)
print(dataset.class_to_idx)
print(dataset.imgs)
```



## 10.创建dataloader迭代器

```python
train_loader = Data.DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=4)

valid_loader = Data.DataLoader(valid_dataset, batch_size=5, shuffle=True, num_workers=4)

test_loader = Data.DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=4)
```

`__getitem__()` 是在 `DataLoader` 中被调用的。在创建 `DataLoader` 对象时，会将要加载的数据集传递给 `DataLoader`，然后在调用 `DataLoader` 的 `__iter__()` 方法时，会调用数据集的 `__getitem__()` 方法，从而获取单个样本。在 `__iter__()` 方法内部，会将数据集中的样本按照指定的 `batch_size` 分成若干个批次，然后对每个批次的样本调用 `collate_fn()` 函数进行处理，最终返回一个批次的数据，即一个包含 `batch_size` 个样本的张量。

所以，在 `DataLoader` 中的调用顺序是：`__iter__()` -> `__getitem__()` -> `collate

## 11.查看dateloader迭代器

```python
def show_loader(loader):
    for (x, y) in loader:
        print(x)
        print(y)
        break
show_loader(valid_loader)
```

# 模型相关

## 12.加载pth文件（预训练模型）

```python
import torch

import torchvision.models as models



# 指定模型种类

model = models.resnet34()



# 加载.pth文件中的模型参数

model.load_state_dict(torch.load('shangyufei/resnet34-333f7ec4.pth'))



# 打印模型结构

print(model)
```

```python
#加载预训练模型
model = torchvision.models.resnet34(pretrained=True, progress=True)
print(model
```

调整模型的全连接层，移动到gpu

```python
device = "cuda:0" if torch.cuda.is_available() else "cpu"
input_channel = model.fc.in_features
model.fc = nn.Linear(input_channel, 176)
model.to(device)
```

我们选择预训练的ResNet-34模型，我们只需重复使用此模型的输出层（即提取的特征）的输入。 然后，我们可以用一个可以训练的小型自定义输出网络替换原始输出层，例如堆叠两个完全连接的图层。

```python
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # 定义一个新的输出网络，共有120个输出类别
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # 将模型参数分配给用于计算的CPU或GPU
    finetune_net = finetune_net.to(devices[0])
    # 冻结参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

## 13.训练函数

```python
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 设定设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]#两个gpu

# 定义训练函数
def train(model, train_loader, valid_loader, epochs, lr, weight_decay,param_group=True):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    if param_group:
        params_1x = [param for name, param in model.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        optimizer = torch.optim.SGD([{'params': params_1x},
                                     {'params': model.fc.parameters(),
                                      'lr': lr * 10}],
                                    lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    
    # 将模型放到多个GPU上进行并行计算
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    
    # 记录训练过程中的loss和accuracy
    train_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    



    for epoch in range(epochs):
        train_loss = 0
        train_total = 0
        train_correct = 0

        # 训练模型
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 验证模型
        model.eval()
        valid_total = 0
        valid_correct = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        # # 测试模型
        # test_total = 0
        # test_correct = 0
        # with torch.no_grad():
        #     for inputs, labels in test_loader:
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)

        #         outputs = model(inputs)
        #         _, predicted = torch.max(outputs.data, 1)
        #         test_total += labels.size(0)
        #         test_correct += (predicted == labels).sum().item()

        # 打印loss和精度
        train_loss_list.append(train_loss/len(train_loader))
        train_acc_list.append(train_correct/train_total)
        valid_acc_list.append(valid_correct/valid_total)
        #test_acc_list.append(test_correct/test_total)
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_correct/train_total:.4f}, Valid Acc: {valid_correct/valid_total:.4f}')

    # 返回训练过程中记录的loss和accuracy
    return train_loss_list, train_acc_list, valid_acc_list
```

train(model,train_loader,valid_loader,50,0.0001,1e-4)

devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4

### 学习率衰退

```python
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)#经过lr_period epochs后 lr*=lr_decay



scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
scheduler.step()
```



## 14.保存检查点与加载ckpt

```
torch.save(model.state_dict(), 'model.ckpt')
model.load_state_dict(torch.load('model.ckpt'))
```

## 15.模型预测

```python
#test_dataset = LeavesDataset(test, image_path)
test_loader = DataLoader(dataset=test_dataset, batch_size=32)
saveFileName = 'submission.csv'
model.to(device)
#model.load_state_dict(torch.load(model_path))
model.eval()
predictions = []
for x in test_loader:
  with torch.no_grad():

​    y_pred = model(x.to(device))

​    predictions.extend(y_pred.argmax(dim=-1).cpu().numpy().tolist())

​    

test_data = pd.read_csv(test_path)

test_data['label'] = pd.Series(predictions)

submission = pd.concat([test_data['image'], test_data['label']], axis=1)

submission.to_csv(saveFileName, index=False)
```

首先，使用 `DataLoader` 类从测试数据集 `test_dataset` 中加载一个批次大小为 32 的数据集合。接着定义了结果输出文件的名称 `saveFileName`。

然后，模型被切换到设备上，如 GPU，以便在其上进行计算。 接着调用 `model.eval()` 方法将模型设置为评估模式，该模式禁用了一些模型层外的行为，例如 dropout 和 batch normalization，以确保结果的稳定性。

`predictions` 是空列表，用于存储预测结果。接下来针对每个批次，调用 `model(x.to(device))` 对输入样本进行预测，并将预测结果的类别（即概率最大的标签）添加到 `predictions` 列表中。

最终，将 `predictions` 转换为 numpy 数组，并使用 `tolist()` 方法将其转换为 Python list。

## 将预测到的数字类别映射为字典中的文字类别

```python
values_list = []
#使用循环遍历keys_list，并使用字典对象的get()方法获取每个key所对应的value
#values_list = [my_dict.get(key) for key in keys_list]
for key in predictions:

  value = num_to_class.get(key)

  values_list.append(value)

print (values_list)

#保存文件
test_data = pd.read_csv(test_path)

test_data['label'] = pd.Series(values_list)
submission = pd.concat([test_data['image'], test_data['label']], axis=1)
submission.to_csv(saveFileName, index=False)
```







# linux命令相关

## 跑ipynb

 *pip in*stall runipy

runipy <YourNotebookName>.ipynb

## 打印工作目录

pwd

# python编程函数

应用于三维张量时，`enumerate()` 会遍历张量的每个二维平面（也可以看作是每个二维子张量），并返回每个平面的索引和对应的值

```python
for i, row in enumerate(jaccard):   
# 遍历二维张量的每一行，i 是行索引，row 是对应的行数据   
# 可以在此处进行处理或使用行数据   
pass
```

## 函数参数dim=1,即对行  dim=0对列

## 匿名函数lambda

lambda x: x[1]  x为可迭代对象，取出索引为1的元素
