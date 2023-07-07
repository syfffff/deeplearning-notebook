# 数据类型转换

Python3 的六个标准数据类型中：

- **不可变数据（3 个）：**Number（数字）、String（字符串）、Tuple（元组）；
- **可变数据（3 个）：**List（列表）、Dictionary（字典）、Set（集合）。



在显式类型转换中，用户将对象的数据类型转换为所需的数据类型。 我们使用 int()、float()、str() 等预定义函数来执行显式类型转换。

```python
num_int=1234
num_str="4567"
print(num_str+str(num_int),'\n',num_int+int(num_str))


45671234 
 5801
```

## list()函数

```python
tuple1=('ab','ac','acd',1,True)
list1=list(tuple1)
print(list1)

str_syf='shang yu fei and tian ye tong'
list2=list(str_syf)
print(list2)


['ab', 'ac', 'acd', 1, True]
['s', 'h', 'a', 'n', 'g', ' ', 'y', 'u', ' ', 'f', 'e', 'i', ' ', 'a', 'n', 'd', ' ', 't', 'i', 'a', 'n', ' ', 'y', 'e', ' ', 't', 'o', 'n', 'g']
```

## **dict()** 

函数用于创建一个字典

```python
keyword_dict=dict(x=5,y=9,end='dict')
print(keyword_dict)
print(keyword_dict.items())
#使用关键字参数创建字典


{'x': 5, 'y': 9, 'end': 'dict'}
dict_items([('x', 5), ('y', 9), ('end', 'dict')])
items()返回可迭代键值对列表


empty=dict()
empty={}
print(empty)

{}#两者都是创建空字典


numbers2 = dict([('x', 5), ('y', -5)], z=8)
print('numbers2 =',numbers2)


numbers2 = {'x': 5, 'y': -5, 'z': 8}
#列表中元素为tuple，构造为字典


```

## **zip()** 函数

用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表（必须使用list函数）。之后使用dict()函数构造字典

```python
a=[1,'syf',3]
b=[2,'tyt',4]
c=zip(a,b)
print(zip(a,b),'\n',type(zip(a,b)))
print(list(zip(a,b)))

<zip object at 0x000002176B740740> 
 <class 'zip'>
[(1, 2), ('syf', 'tyt'), (3, 4)]    
 

a1,a2=list(zip(*c))# 与 zip 相反，zip(*) 可理解为解压
print(a1,'\n',a2)


(1, 'syf', 3) 
 (2, 'tyt', 4)
    
    
mydict=dict(list(zip(a,b)))
print(mydict)

{1: 2, 'syf': 'tyt', 3: 4}
```

## 同时遍历两个或更多的序列，可以使用 zip() 组合



```python
num_list=[1,2,3,4]

a_num_list=['one','two','three','four']

dict_list={}

for a,b in zip(num_list,a_num_list):

  print(a,b)

  dict_list[a]=b

print(dict_list)

1 one
2 two
3 three
4 four
{1: 'one', 2: 'two', 3: 'three', 4: 'four'}
```



# 字符串

## **f-string** 

格式化字符串以 **f** 开头，后面跟着字符串，字符串中的表达式用大括号 {} 包起来，它会将变量或表达式计算后的值替换进去

```python
Fdict_test={1:1,2:3,'adad':'dacdc'}
f'{Fdict_test[1]},{Fdict_test[2]},{Fdict_test["adad"]}'

'1,3,dacdc'
#字符串
```

## split()

split() 通过指定分隔符对字符串进行切片，如果第二个参数 num 有指定值，则分割为 num+1 个子字符串。

```python
str = "this is string example....wow!!!"
print (str.split( ))       # 以空格为分隔符
print (str.split('i',1))   # 以 i 为分隔符
print (str.split('w'))     # 以 w 为分隔符




['this', 'is', 'string', 'example....wow!!!']
['th', 's is string example....wow!!!']
['this is string example....', 'o', '!!!']
```

##  lower() 

```
str.lower()
```

方法转换字符串中所有大写字符为小写

## join()

Python join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。

````python
s1 = "-"
s2 = ""
seq = ("r", "u", "n", "o", "o", "b") # 字符串序列(tuple list string is ok)
print (s1.join( seq ))
print (s2.join( seq ))
````

## replace()

text.replace('\u202f', ' ')

## 字符串截取（反转）

str[::2]  表示的是从头到尾，步长为2。第一个冒号两侧的数字是指截取字符串的范围,第二个冒号后面是指截取的步长。

str[::-1]为反转

```
str[start:end:span]
```

遍历 [start,end)，间隔为 span，当 span>0 时顺序遍历, 当 span<0 时，逆着遍历。

start 不输入则默认为 0，end 不输入默认为长度。

## 字符串函数总览

![image-20230703145632331](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230703145632331.png)

![image-20230703145732803](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230703145732803.png)

![image-20230703145805775](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230703145805775.png)

![image-20230703145823689](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230703145823689.png)

## enumerate()

用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中\

```python
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
 list(enumerate(seasons, start=1))       # 下标从 1 开始
    
    
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]


seq = ['one', 'two', 'three']
for i, element in enumerate(seq):
    print(i, element)
```

```python
for i, v in enumerate(['tic', 'tac', 'toe']):
...   print(i, v)
...
0 tic
1 tac
2 toe
```



# 列表list

## append()

 用于在列表末尾添加新的对象。list.append(obj)

## 列表相加拼接

```python
list2=[1,2,3,'565']
list3=['s','y','f']
list2.append(list3)
print(list2)
add_list=list2+list3
print(add_list)


[1, 2, 3, '565', ['s', 'y', 'f']]
[1, 2, 3, '565', ['s', 'y', 'f'], 's', 'y', 'f']
```

## remove

```python
list1 = ['Google', 'Runoob', 'Taobao', 'Baidu']
list1.remove('Taobao')
print ("列表现在为 : ", list1)
list1.remove('Baidu')
print ("列表现在为 : ", list1)


列表现在为 :  ['Google', 'Runoob', 'Baidu']
列表现在为 :  ['Google', 'Runoob']
```

## insert

```
list.insert(index, obj)
```

index -- 对象obj需要插入的索引位置。

obj -- 要插入列表中的对象。

```python
list1 = ['Google', 'Runoob', 'Taobao']
list1.insert(1, 'Baidu')
print ('列表插入元素后为 : ', list1)


列表插入元素后为 :  ['Google', 'Baidu', 'Runoob', 'Taobao']
```

## 列表函数总览![image-20230627195329423](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230627195329423.png)

## del

用 del 语句可以从一个列表中根据索引来删除一个元素，而不是值，来删除元素。这与使用 pop() 返回一个值

```python
a = [-1, 1, 66.25, 333, 333, 1234.5]
 del a[0]
a
[1, 66.25, 333, 333, 1234.5]
```



## range()

range() 函数返回的是一个可迭代对象（类型是对象），而不是列表类型， 所以打印的时候不会打印列表。多用于for循环

```
range(stop)
range(start, stop[, step])
```

参数说明：

- start: 计数从 start 开始。默认是从 0 开始。例如 range(5) 等价于 **range(0， 5)**
- stop: 计数到 stop 结束，但不包括 stop。例如：range(0， 5) 是 [0, 1, 2, 3, 4] 没有 5
- step：步长，默认为 **1**。例如：range(0， 5) 等价于 range(0, 5, 1)

如果只提供一个参数，它将生成一个从 0 开始的整数序列，参数为结束值，步长默认为 1：

```python
**for** number **in** range(6):
  **print**(number)
```

# tuple元组

Python 的元组与列表类似，不同之处在于元组的元素不能修改。

## 元组使用小括号 ( )

列表使用方括号 **[ ]**。

元组创建很简单，只需要在括号中添加元素，并使用逗号隔开即可。

```
tup3 = "a", "b", "c", "d"   #  不需要括号也可以
type(tup3)
<class 'tuple'>
```

元组中只包含一个元素时，需要在元素后面添加逗号 **,** ，否则括号会被当作运算符使用：

```python
tup3=(50)
print(type(tup3))
tup4=(50,)
print(isinstance(tup4,tuple))


<class 'int'>
True
```

元组中的元素值是不允许修改的，但我们可以对元组通加法进行连接组合

```python
print(tup1+tup2)

(1, 2, 3, '87dsad', 'ada', 1, 2, 3, 5, '456efs', '46465')
```

元组中的元素值是不允许删除的，但我们可以使用del语句来删除整个元组

``` 
del tup
```

与字符串一样，元组之间可以使用 **+**、**+=**和 ***** 号进行运算。这就意味着他们可以组合和复制，运算后会生成一个新的元组。

```python
syf=('hello 田烨彤',)*4
print(syf)

('hello 田烨彤', 'hello 田烨彤', 'hello 田烨彤', 'hello 田烨彤')
```

# dict字典

字典的每个键值 **key=>value** 对用冒号 **:** 分割，每个对之间用逗号(**,**)分割，整个字典包括在花括号 **{}** 中 

```python
d = {key1 : value1, key2 : value2, key3 : value3 }
```

*键必须是唯一的，但值则不必。*

## 值可以取任何数据类型，但键必须是不可变的，如字符串，数字

```python
dict1={1:5246,2:'test','third':3}
print(len(dict1))
print(dict1)
print(dict1.items())

3
{1: 5246, 2: 'test', 'third': 3}
dict_items([(1, 5246), (2, 'test'), ('third', 3)])
```

空字典：{}   or   dict()

```python
print(dict1['third'])
dict1['third']=[1,2,'syf']
dict1['school']='HFUT'
print(dict1)


[1, 2, 'syf']
{1: 5246, 2: 'test', 'third': [1, 2, 'syf'], 'school': 'HFUT'}
```

清空字典

````python
dict1.clear()
print(dict1)

{}
````



## 键必须不可变，所以可以用数字，字符串或元组充当，而用列表就不行

## fromkey()

创建一个新字典，以序列seq中元素做字典的键，val为字典所有键对应的初始值

```python
seq = ('name', 'age', 'sex')
val=['tyt',23,'female']
tyt_dict=dict.fromkeys(seq)
key=list(tyt_dict.keys())
type(key)
for i in range(len(seq)):
    tyt_dict[key[i]]=val[i]
print('田烨彤的信息：' f'{tyt_dict}')

田烨彤的信息：{'name': 'tyt', 'age': 23, 'sex': 'female'}



tinydict = dict.fromkeys(seq, 10)
print ("新的字典为 : %s" %  str(tinydict))
```

## get()

字典 **get()** 函数返回指定键的值。

```python

tinydict = {'RUNOOB' : {'url' : 'www.runoob.com'}}

res = tinydict.get('RUNOOB', {}).get('url')
# 输出结果
print("RUNOOB url 为 : ", str(res))
```

## key in dict

![image-20230628160852119](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230628160852119.png)

## dict.items()

tems() 方法以列表返回视图对象，**是一个可遍历的key/value 对。**

## [dict.keys()](https://www.runoob.com/python3/python3-att-dictionary-keys.html)、[dict.values()](https://www.runoob.com/python3/python3-att-dictionary-values.html)   返回都是可以迭代的



和 dict.items() 返回的都是视图对象（ view objects），提供了字典实体的动态视图，这就意味着字典改变，视图也会跟着变化。

视图对象不是列表，不支持索引，可以使用 list() 来转换为列表。

````python
将字典的 key 和 value 组成一个新的列表：

d={1:"a",2:"b",3:"c"}
result=[]
for k,v in d.items():
    result.append(k)
    result.append(v)

print(result)
````

## 字典函数总览

![image-20230628161847508](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230628161847508.png)

## dict()

如果关键字只是简单的字符串，使用关键字参数指定键值对有时候更方便：

```
dict(sape=4139, guido=4127, jack=4098)
{'sape': 4139, 'jack': 4098, 'guido': 4127}
```

构造函数 dict() 直接从键值对元组列表中构建字典。如果有固定的模式，列表推导式指定特定的键值对：

```
dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])
{'sape': 4139, 'jack': 4098, 'guido': 4127}
```



# set集合

集合（set）是一个无序的不重复元素序列。

可以使用大括号 **{ }** 或者 **set()** 函数创建集合，注意：创建一个空集合必须用 **set()** 而不是 **{ }**，因为 **{ }** 是用来创建一个空字典。

```
parame = {value01,value02,...}
或者
set(value)
```

```python
basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket)                      # 这里演示的是去重功能
{'orange', 'banana', 'pear', 'apple'}
 'orange' in basket                 # 快速判断元素是否在集合内
True
>>> 'crabgrass' in basket
False
```

# Python3 条件控制

```python
if condition_1:
    statement_block_1
elif condition_2:
    statement_block_2
else:
    statement_block_3
```


1、每个条件后面要使用冒号 **:**，表示接下来是满足条件后要执行的语句块。

2、使用缩进来划分语句块，相同缩进数的语句在一起组成一个语句块。

# 循环语句(注意：)

## while 循环

在 Python 中没有 do..while 循环。

```python
n = 100
 
sum = 0
counter = 1
while counter <= n:
    sum = sum + counter
    counter += 1
print(f"0到{n}的和为{sum}")

0到100的和为5050
```

## while   else 语句

如果 while 后面的条件语句为 false 时，则执行 else 的语句块。 

```
while <expr>:
    <statement(s)>
else:
    <additional_statement(s)>
```

## for 循环

可以遍历任何可迭代对象，如一个列表或者一个字符串。

for循环的一般格式如下：

```python
for <variable> in <sequence>:
    <statements>
else:
    <statements>
```

\#  1 到 5 的所有数字：

 for number in range(1, 6):  

​	  print(number)

## for...else

```python
for item in iterable:
    # 循环主体
else:
    # 循环结束后执行的代码
```

当循环执行完毕（即遍历完 iterable 中的所有元素）后，会执行 else 子句中的代码，如果在循环过程中遇到了 break 语句，则会中断循环，此时不会执行 else 子句。

##  range() 函数用法

 range() 函数返回的是一个可迭代对象（类型是对象），而不是列表类型， 所以打印的时候不会打印列表。

 list() 函数是对象迭代器，可以把 range() 返回的可迭代对象转为一个列表，返回的变量类型为列表。

```python
range(stop)#计数到 stop 结束，但不包括 stop。例如：range(0， 5) 是 [0, 1, 2, 3, 4] 没有 5
range(start, stop[, step])
```

## **break**  continue

**break** 语句可以跳出 for 和 while 的循环体。如果你从 for 或 while 循环中终止，任何对应的循环 else 块将不执行。

**continue** 语句被用来告诉 Python 跳过当前循环块中的剩余语句，然后继续进行下一轮循环。

# 推导式

Python 支持各种数据结构的推导式：

- 列表(list)推导式
- 字典(dict)推导式
- 集合(set)推导式
- 元组(tuple)推导式

## 列表推导式

```python
[表达式 for 变量 in 列表] 
[out_exp_res for out_exp in input_list]

或者 

[表达式 for 变量 in 列表 if 条件]
[out_exp_res for out_exp in input_list if condition]
```

必须加方括号

```python
tyt_list=['t','y','t','不能说冷漠词','不能赌气','20230703']
new_tyt_list=[tyt for tyt in tyt_list if tyt.isalpha()]
new_tyt_list.append('syf')
print(new_tyt_list)


['t', 'y', 't', '不能说冷漠词', '不能赌气', 'syf']
```

循环里套循环

```python
vec1 = [2, 4, 6]
vec2 = [4, 3, -9]
print([x*y for x in vec1 for y in vec2],'\n',[x+y for x in vec1 for y in vec2])
```



## 字典推导式

```
{ key_expr: value_expr for value in collection }

或

{ key_expr: value_expr for value in collection if condition 必须花括号
```

```python
syf_dict={'123':1,'tyt':2,'ttytt':3}
print(list(syf_dict.items()))
new_syf_dict={ key:value for key,value in list(syf_dict.items()) if  value>1}
print(new_syf_dict)


[('123', 1), ('tyt', 2), ('ttytt', 3)]
{'tyt': 2, 'ttytt': 3}
```

## 集合推导式

```python
{ expression for item in Sequence }
或
{ expression for item in Sequence if conditional }
```

```python
seta={x for x in 'adhiuxzccdia' if x not in 'adcads'}
print(seta)

{'x', 'z', 'i', 'h', 'u'}
```

## 元组推导式（生成器表达式）

元组推导式返回的结果是一个生成器对象

```python
(expression for item in Sequence )
或
(expression for item in Sequence if conditional )
```

```
>>> a = (x for x in range(1,10))
>>> a
<generator object <genexpr> at 0x7faf6ee20a50>  # 返回的是生成器对象

>>> tuple(a)       # 使用 tuple() 函数，可以直接将生成器对象转换成元组
(1, 2, 3, 4, 5, 6, 7, 8, 9)
```

# 迭代器

迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不会后退。

迭代器有两个基本的方法：**iter()** 和 **next()**。

字符串，列表或元组对象都可用于创建迭代器

```python
l=[1,2,'adsvc',456]
it=iter(l)
next(it)


1
```

````python
it=iter(l)
for i in it:
    print(i)
    
1
2
adsvc
456
````

把一个类作为一个迭代器使用需要在类中实现两个方法 __iter__() 与 __next__() 

__iter__() 方法返回一个特殊的迭代器对象， 这个迭代器对象实现了 __next__() 方法并通过 StopIteration 异常标识迭代的完成。

__next__() 方法（Python 2 里是 next()）会返回下一个迭代器对象。

StopIteration 异常用于标识迭代的完成，防止出现无限循环的情况，在 __next__() 方法中我们可以设置在完成指定循环次数后触发 StopIteration 异常来结束迭代。

```python
class iter_class:
    def __iter__(self):
        self.key=['s','y','f','l','o','v','e','t','y','t']
        self.i=0
        return self
    def __next__(self):
        
        if(self.i<len(self.key)) :   
            x=self.key[self.i]
            self.i+=1
            return x
        else:
            raise StopIteration
    
myclass=iter_class()
for x in myclass:
  print(x,end=" ")



s y f l o v e t y t 
```

# 生成器

在 Python 中，使用了 **yield** 的函数被称为生成器（generator）。

**yield** 是一个关键字，用于定义生成器函数，生成器函数是一种特殊的函数，可以在迭代过程中逐步产生值，而不是一次性返回所有结果。

跟普通函数不同的是，生成器是一个返回迭代器的函数，只能用于迭代操作，更简单点理解生成器就是一个迭代器。

当在生成器函数中使用 **yield** 语句时，函数的执行将会暂停，并将 **yield** 后面的表达式作为当前迭代的值返回。

然后，每次调用生成器的 **next()** 方法或使用 **for** 循环进行迭代时，函数会从上次暂停的地方继续执行，直到再次遇到 **yield** 语句。这样，生成器函数可以逐步产生值，而不需要一次性计算并返回所有结果。

调用一个生成器函数，返回的是一个迭代器对象。

1. **打个比方的话，yield有点像断点。   加了yield的函数，每次执行到有yield的时候，会返回yield后面的值 并且函数会暂停，直到下次调用或迭代终止；**
2. **yield后面可以加多个数值（可以是任意类型），但返回的值是元组类型的。**

**什么情况下需要使用 yield？**

一个函数 f，f 返回一个 list，这个 list 是动态计算出来的（不管是数学上的计算还是逻辑上的读取格式化），并且这个 list 会很大（无论是固定很大还是随着输入参数的增大而增大），这个时候，我们希望每次调用这个函数并使用迭代器进行循环的时候一个一个的得到每个 list 元素而不是直接得到一个完整的 list 来节省内存，这个时候 yield 就很有用。

```python
def fab(max): 
    n, a, b = 0, 0, 1 
    while n < max: 
        yield b 
        # print b 
        a, b = b, a + b 
        n = n + 1

for x in fab(1000):
    print(x)
```

# 函数的不定长参数

## *args

你可能需要一个函数能处理比当初声明时更多的参数。这些参数叫做不定长参数，和上述 2 种参数不同，声明时不会命名。基本语法如下：

```
def functionname([formal_args,] *var_args_tuple ):
   "函数_文档字符串"
   function_suite
   return [expression]
```

加了星号 ***** 的参数会以元组(tuple)的形式导入，存放所有未命名的变量参数。

```python
def test_for_star(x1,*var_x2):
    print(x1)
    print(var_x2)

test_for_star(12345,123,'syf','564')

12345
(123, 'syf', '564')
```

## **kwargs

还有一种就是参数带两个星号 ***\***基本语法如下：

```
def functionname([formal_args,] **var_args_dict ):
   "函数_文档字符串"
   function_suite
   return [expression]
```

加了两个星号 ***\*** 的参数会以**字典**的形式导入。

```python
def printinfo( arg1, **vardict ):
   "打印任何传入的参数"
   print ("输出: ")
   print (arg1)
   print (vardict)
 
# 调用printinfo 函数
printinfo(1, a=2,b=3)

输出: 
1
{'a': 2, 'b': 3}
```

## 匿名函数**lambda**

所谓匿名，意即不再使用 **def** 语句这样标准的形式定义一个函数。

```python
lambda [arg1 [,arg2,.....argn]]:expression
```

```python
x = lambda a : a + 10
print(x(5))
```

```python
sum = lambda arg1, arg2: arg1 + arg2
 
# 调用sum函数
print ("相加后的值为 : ", sum( 10, 20 ))
print ("相加后的值为 : ", sum( 20, 20 ))
```

# Python3 模块

在前面的几个章节中我们基本上是用 python 解释器来编程，如果你从 Python 解释器退出再进入，那么你定义的所有的方法和变量就都消失了。

为此 Python 提供了一个办法，把这些定义存放在文件中，为一些脚本或者交互式的解释器实例使用，这个文件被称为模块。

模块是一个包含所有你定义的函数和变量的文件，其后缀名是.py。模块可以被别的程序引入，以使用该模块中的函数等功能。这也是使用 python 标准库的方法。

**模块是一个包含所有你定义的函数和变量的文件，其后缀名是.py**

## 搜索路径

当解释器遇到 import 语句，如果模块在当前的搜索路径就会被导入。

当我们使用 import 语句的时候，Python 解释器是怎样找到对应的文件的呢？

这就涉及到 Python 的搜索路径，搜索路径是由一系列目录名组成的，Python 解释器就依次从这些目录中去寻找所引入的模块。

这看起来很像环境变量，事实上，也可以通过定义环境变量的方式来确定搜索路径。

搜索路径是在 Python 编译或安装的时候确定的，安装新的库应该也会修改。搜索路径被存储在 sys 模块中的 path 变量，做一个简单的实验，在交互式解释器中，输入以下代码：

```python
import sys
sys.path


['g:\\d2l\\d2l_notebook_learning',
 'e:\\miniconda_installlocation\\envs\\PyTorch_date3.8\\python39.zip',
 'e:\\miniconda_installlocation\\envs\\PyTorch_date3.8\\DLLs',
 'e:\\miniconda_installlocation\\envs\\PyTorch_date3.8\\lib',
 'e:\\miniconda_installlocation\\envs\\PyTorch_date3.8',
 '',
 'e:\\miniconda_installlocation\\envs\\PyTorch_date3.8\\lib\\site-packages',
 'e:\\miniconda_installlocation\\envs\\PyTorch_date3.8\\lib\\site-packages\\win32',
 'e:\\miniconda_installlocation\\envs\\PyTorch_date3.8\\lib\\site-packages\\win32\\lib',
 'e:\\miniconda_installlocation\\envs\\PyTorch_date3.8\\lib\\site-packages\\Pythonwin']
```

因此若像我一样在当前目录下存在与要引入模块同名的文件，就会把要引入的模块屏蔽掉。

了解了搜索路径的概念，就可以在脚本中修改sys.path来引入一些不在搜索路径中的模块

现在，在解释器的当前目录或者 sys.path 中的一个目录里面来创建一个fibo.py的文件，代码如下：

```python
#filename fibo.py
def fib(n):    # 定义到 n 的斐波那契数列
    a, b = 0, 1
    while b < n:
        print(b, end=' ')
        a, b = b, a+b
    print()
 
def fib2(n): # 返回到 n 的斐波那契数列
    result = []
    a, b = 0, 1
    while b < n:
        result.append(b)
        a, b = b, a+b
    return result
```

在另一文件中 import fibo

```python
fibo.fib(100)
fibo.fib2(100)

1 1 2 3 5 8 13 21 34 55 89 
[1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
```

如果你打算经常使用一个函数，你可以把它赋给一个本地的名称：

```
>>> fib = fibo.fib
```

## from … import 语句

Python 的 from 语句让你从模块中导入一个指定的部分到当前命名空间中，语法如下：

```
from modname import name1[, name2[, ... nameN]]
```

```python
>>> from fibo import fib, fib2
>>> fib(500)#不需要加.访问了
```

from … import * 语句

把一个模块的所有内容全都导入到当前的命名空间也是可行的，只需使用如下声明：

```
from modname import *
```

## __name__属性

一个模块被另一个程序第一次引入时，其主程序将运行。如果我们想在模块被引入时，模块中的某一程序块不执行，我们可以用__name__属性来使该程序块仅在该模块自身运行时执行。

```python
if __name__ == '__main__':
   print('程序自身在运行')
else:
   print('我来自另一模块')
```

**说明：** 每个模块都有一个__name__属性，当其值是'__main__'时，表明该模块自身在运行，否则是被引入。

说明：**__name__** 与 **__main__** 底下是双下划线， **_ _** 是这样去掉中间的那个空格。

## 包结构

包是一种管理 Python 模块命名空间的形式，采用"点模块名称"。

比如一个模块的名称是 A.B， 那么他表示一个包 A中的子模块 B 。

就好像使用模块的时候，你不用担心不同模块之间的全局变量相互影响一样，采用点模块名称这种形式也不用担心不同库之间的模块重名的情况。

这里给出了一种可能的包结构（在分层的文件系统中）:

```
sound/                          顶层包
      __init__.py               初始化 sound 包
      formats/                  文件格式转换子包
              __init__.py
              wavread.py
              wavwrite.py
              aiffread.py
              aiffwrite.py
              auread.py
              auwrite.py
              ...
      effects/                  声音效果子包
              __init__.py
              echo.py
              surround.py
              reverse.py
              ...
      filters/                  filters 子包
              __init__.py
              equalizer.py
              vocoder.py
              karaoke.py
              ...
```

在导入一个包的时候，Python 会根据 sys.path 中的目录来寻找这个包中包含的子目录。

目录只有包含一个叫做 __init__.py 的文件才会被认作是一个包，主要是为了避免一些滥俗的名字（比如叫做 string）不小心的影响搜索路径中的有效模块。

最简单的情况，放一个空的 :file:__init__.py就可以了

用户可以每次只导入一个包里面的特定模块，比如:

```
import sound.effects.echo
```

这将会导入子模块:sound.effects.echo。 他必须使用全名去访问:

```
sound.effects.echo.echofilter(input, output, delay=0.7, atten=4)
```

还有一种导入子模块的方法是:

```
from sound.effects import echo
```

这同样会导入子模块: echo，并且他不需要那些冗长的前缀，所以他可以这样使用:

```
echo.echofilter(input, output, delay=0.7, atten=4)
```

还有一种变化就是直接导入一个函数或者变量:

```
from sound.effects.echo import echofilter
```

同样的，这种方法会导入子模块: echo，并且可以直接使用他的 echofilter() 函数:

```
echofilter(input, output, delay=0.7, atten=4)
```

注意当使用 **from package import item** 这种形式的时候，对应的 item 既可以是包里面的子模块（子包），或者包里面定义的其他名称，比如函数，类或者变量。

反之，如果使用形如 **import item.subitem.subsubitem** 这种导入形式，除了最后一项，都必须是包，而最后一项则可以是模块或者是包，但是不可以是类，函数或者变量的名字。

## 导入自己的模块

关于导入模块，自己写的程序，自己也可以把它保存下来，以后需要的时候导入使用，例如下面所示。

我有个代码名称为 test1.py，它的所在路径为 **D:\test** 下面。那我只需要完成以下步骤就可以把它作为模块 **import** 到其他代码中了。

-  1.**import sys**
-  2.**sys.path.append("D:\\test")**

在 test2.py 中我们就可以直接 **import test1.py** 了。成功导入后，test1中 的方法也就可以在 test2 中进行使用。

```
import test1
```

# Python3 输入和输出

## 输出格式美化

### format_字符串

format作为Python的的格式字符串函数，主要通过字符串中的花括号{}，来识别替换字段，从而完成字符串的格式化。

```python
print('{} love {}'.format('tyt','syf'))
a,b=1,'abc'
print('{} like {}'.format(a,b))
c=[1,2]
print('{0} like {2}'.format(a,b,c))

tyt love syf
1 like abc
1 like [1, 2]
```

````python
table = {'Google': 1, 'Runoob': 2, 'Taobao': 3}
for name, number in table.items():
    print('{0:8} ==> {1:10}'.format(name, number))
    
Google   ==>          1
Runoob   ==>          2
Taobao   ==>          3


table = {'Google': 1, 'Runoob': 2, 'Taobao': 3}
print('Runoob: {0[Google]}; Google: {0[Taobao]}; Taobao: {0[Runoob]}'.format(table))


Runoob: 1; Google: 3; Taobao: 2
````

在 **:** 后传入一个整数, 可以保证该域至少有这么多的宽度。 用于美化表格时很有用

## 读取键盘输入

Python 提供了 [input() 内置函数](https://www.runoob.com/python3/python3-func-input.html)从标准输入读入一行文本，默认的标准输入是键盘

```python
str = input("请输入：");
print ("你输入的内容是: ", str)
```

## 读和写文件open

open() 将会返回一个 file 对象，基本语法格式如下:

```
open(filename, mode)
```

- filename：包含了你要访问的文件名称的字符串值。
- mode：决定了打开文件的模式：只读，写入，追加等。所有可取值见如下的完全列表。这个参数是非强制的，默认文件访问模式为只读(r)。

![image-20230707174445893](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230707174445893.png)

```
f=open('mycode.txt','w')
f.write('python第一步雀氏纸尿裤')
f.close()
```

open函数必须搭配.close()方法使用，先用open打开文件，然后进行读写操作，最后用.close()释放文件。

## open绝对路径记得加r

```
f1=open(r'G:\d2l\d2l_notebook_learning\mycode.txt','w')
f1.write('python第一步雀氏纸尿裤\n第二行')
f1.close()
```

