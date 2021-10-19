# ML HW1
> 0716081 葉晨

## Project preview
1. Data Input
2. Data Visualization
3. Data Preprocessing
4. Model Construction
5. Train-Test-Split
5. Results
6. Compare and Conclusion
7. Question

## Data Input
### Mushroom
```python
name_list = ['edible','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']
feature_list = name_list
feature_list.remove('edible')
attribute_list = [['b','c','x','f','k','s'],['f','g','y','s'],['n','b','c','g','r','p','u','e','w','y'],['t','f'],['a','l','c','y','f','m','n','p','s'],['a','d','f','n'],['c','w','d'],['b','n'],['k','n','b','h','g','r','o','p','u','e','w','y'],['e','t'],['b','c','u','e','z','r'],['f','y','k','s'],['f','y','k','s'],['n','b','c','g','o','p','e','w','y'],['n','b','c','g','o','p','e','w','y'],['p','u'],['n','o','w','y'],['n','o','t'],['c','e','f','l','n','p','s','z'],['k','n','b','h','r','o','u','w','y'],['a','c','n','s','v','y'],['g','l','m','p','u','w','d']]
cate_num_list = [6,4,10,2,9,4,3,2,12,2,4,4,9,9,2,4,3,8,9,6,7]
data = pd.read_csv('agaricus-lepiota.data', sep=",", names=name_list)
```

### Iris
```python
name_list = ['sepal-length','sepal-width','petal-length','petal-width','class']
feature_list = name_list
feature_list.remove('class')
label_list = ['Iris-setosa','Iris-versicolor','Iris-virginica']
data = pd.read_csv('iris.data', sep=",", names=name_list)
data_group = data.groupby('class')

```
#### Variables
- **name_list**: 為 .data 檔案加上每個 column 的說明
- **feature_list**: 每個 feature 的名稱
- **attribute_list**: 每個 feature 的所有可能的值 (Mushroom)
- **cate_num_list**: 每個 feature 的所有可能的值的**數量** (Mushroom)
- **label_list**: label 的所有可能的值 (Iris)

## Data Visualization (未完成！！！)
### Mushroom
![2021-10-19 下午 10-26-25](https://raw.githubusercontent.com/denny3388/ML_HW1/master/pictures/2021-10-19%20%E4%B8%8B%E5%8D%88%2010-26-25.png)

![20211019223439](https://raw.githubusercontent.com/denny3388/ML_HW1/master/pictures/20211019223439.png)

## Model Construction
### Mushroom

我使用了`CategoricalNB()`當作model，並且將**Model Construction**、**Validation**寫在函式`holdout()`及`kfold()`裡面

```python
model = CategoricalNB(alpha=alpha_, min_categories=cate_num_list)
```

### Iris

我使用了`GaussianNB()`當作model，並且將**Model Construction**、**Validation**寫在函式`holdout()`及`kfold()`裡面

```python
model = GaussianNB()
```

## Train-Test-Split
### Mushroom
#### Select data for modeling
```python
X=data[feature_list]
y=data['edible'].values

enc = OrdinalEncoder()
X = enc.fit_transform(X)
```
- **X**: features
- **y**: label (edible)

要用 OrdinalEncoder() 是因為要把 feature 的 char 轉成 int 型態

#### Holdout
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
clf = model.fit(X_train, y_train)
```
利用 `train_test_split()` 將 X, y 分成 **Train:Test = 7:3**

之後將 training data 利用 `fit()` 餵入建立好的 model

#### K-fold
```python
K = 3
kf = KFold(n_splits=K)

for train_index, test_index in kf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = model.fit(X_train, y_train)
```
利用 `Kfold()` 將 X, y 分成指定區間

之後利用 for loop 加上 `Kfold.split()` 逐次將每個 fold 的 training data 餵入 model

### Iris

#### Select data for modeling
```python
X=data[feature_list]
y=data['class'].values
```
- **X**: features
- **y**: label (class)

#### Holdout
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
clf = model.fit(X_train, y_train)
```
利用 `train_test_split()` 將 X, y 分成 **Train:Test = 7:3**

之後將 training data 利用 `fit()` 餵入建立好的 model

#### K-fold
```python
K = 3
kf = KFold(n_splits=K)

for train_index, test_index in kf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = model.fit(X_train, y_train)
```
利用 `Kfold()` 將 X, y 分成指定區間

之後利用 for loop 加上 `Kfold.split()` 逐次將每個 fold 的 training data 餵入 model

## Results
### Mushroom
### Iris