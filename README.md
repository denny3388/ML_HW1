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

## Data Preprocessing
### Mushroom
```python
# Drop feature w/ missing value 
data = data.drop(columns='stalk-root')
feature_list.remove('stalk-root')

# shuffle
data = data.sample(frac=1)
```
- Drop features with missing value
- Shuffle the data using `sameple()`

### Iris
```python
# Divide data into group according to their class
data_group = data.groupby('class')

# shuffle
data = data.sample(frac=1)
```
- 將 data 以他們的 class 分組，之後在做資料分析時較方便
- Shuffle the data using `sameple()`

## Data Visualization (未完成！！！)
### Mushroom

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

此處以 **'e' 為 positive**，並計算 Confusion matrix, Accuracy, Sensitivity(Recall), Precision 等數值
```python
TP, FP, FN, TN = 0, 0, 0, 0
pred = clf.predict(X_test)
for i in range(len(X_test)):
    if pred[i] == 'e' and y_test[i] == 'e':
        TP += 1
    elif pred[i] == 'e' and y_test[i] == 'p':
        FP += 1
    elif pred[i] == 'p' and y_test[i] == 'e':
        FN += 1
    elif pred[i] == 'p' and y_test[i] == 'p':
        TN += 1
confusion_mat = pd.DataFrame([[TP, FP], [FN, TN]])
acc = (TP+TN)/len(X_test)
recall = TP/(TP+FN)
precision = TP/(TP+FP)
```

#### Confusion matrix
- Without Laplace

|Validation:|Holdout|   | K-fold |    |
|  ----  | ----  | ---- | ---- | ---- |
|**Predict \ Actual**|**e** |**p**|**e**|**p**|
|**e**| 1256 | 2         |1395.67|1    |
|**p**| 9    |1171       |7    |1304.33|
> K-fold 的 confusion matrix 為三個 fold 計算結果相加之後再取平均

- With Laplace

|Validation:|Holdout|   | K-fold |    |
|  ----  | ----  | ---- | ---- | ---- |
|**Predict \ Actual**|**e** |**p**|**e**|**p**|
|**e**| 1249 | 104       |1395|104.33    |
|**p**| 8    | 1077      |7.67 |1201|

#### Accuracy, Sensitivity(Recall), Precision

|Validation:| \|-----| Holdout  | -----\||\|-----|  K-fold  |-----\||
|  ----  | ----  | ---- | ---- | ---- | ---- | ---- |
|**Smoothing \ Performance(%)**|**Acc** |**Sens**|**Prec**|**Acc** |**Sens**|**Prec**|
|**Without Laplace**|99.63|99.36|99.92|99.7 |99.51|99.93|
|**With Laplace**   |95.41|99.36|92.31|95.86|99.46|93.04|


### Iris

此處 **分別以三個不同 class 為 positive**，並計算 Confusion matrix, Accuracy, Sensitivity(Recall), Precision 等數值
```python
for p in range(3): # for each class being positive
    TP, FP, FN, TN = 0, 0, 0, 0
    pred = clf.predict(X_test)
    positive = label_list[p]
    for i in range(len(X_test)):
        if pred[i] == positive and y_test[i] == positive:
            TP += 1
        elif pred[i] == positive and y_test[i] != positive:
            FP += 1
        elif pred[i] != positive and y_test[i] == positive:
            FN += 1
        elif pred[i] != positive and y_test[i] != positive:
            TN += 1
    mat = pd.DataFrame([[TP, FP], [FN, TN]])
    confusion_mat.append(mat)
    acc.append((TP+TN)/len(X_test))
    recall.append(TP/(TP+FN))
    precision.append(TP/(TP+FP))
```

#### Confusion matrix
- Positive = **Iris-setosa**

|Validation:|Holdout|   | K-fold |    |
|  ----  | ----  | ---- | ---- | ---- |
|**Predict \ Actual**|**Iris-setosa** |**Else**|**Iris-setosa**|**Else**|
|**Iris-setosa**| 19 | 0         |16.67|0    |
|**Else**| 0    |26       |0    |33.33|

- Positive = **Iris-versicolor**

|Validation:|Holdout|   | K-fold |    |
|  ----  | ----  | ---- | ---- | ---- |
|**Predict \ Actual**|**Iris-versicolor** |**Else**|**Iris-versicolor**|**Else**|
|**Iris-versicolor**| 13 | 7         |15.67|1.33    |
|**Else**| 0    |31       |1    |32|

- Positive = **Iris-virginica**

|Validation:|Holdout|   | K-fold |    |
|  ----  | ----  | ---- | ---- | ---- |
|**Predict \ Actual**|**Iris-virginica** |**Else**|**Iris-virginica**|**Else**|
|**Iris-virginica**|12|0	|15.33|	1|
|**Else**|1	|32|1.33|	32.33|

#### Accuracy, Sensitivity(Recall), Precision

|Validation:| \|-----| Holdout  | -----\||\|-----|  K-fold  |-----\||
|  ----  | ----  | ---- | ---- | ---- | ---- | ---- |
|**Positive \ Performance(%)**|**Acc** |**Sens**|**Prec**|**Acc** |**Sens**|**Prec**|
|**Iris-setosa**|100|100|100|100|100|100|
|**Iris-versicolor**|97.78|100|92.86|95.33|93.94|91.91|
|**Iris-virginica**|97.78|92.31|100|95.33|92.59|93.86|
|**\*Mean**|98.52|97.44|97.62|96.89|95.51|95.26|
> The row **\*Mean** is the mean of three label being positive

## Comparison & Conclusion

## Question
### Mushoroom
```python
cate_list = ['n','b','c','g','o','p','e','w','y'] # All category that 'stalk-color-below-ring' have
num_Xi_y = [0,0,0,0,0,0,0,0,0]

filter_e = (data['edible']=='e')
num_y = len(data[filter_e])
cnt = data[filter_e]['stalk-color-below-ring'].value_counts()
num_Xi_y[7] = cnt[0]
num_Xi_y[3] = cnt[1]
num_Xi_y[5] = cnt[2]
num_Xi_y[4] = cnt[3]
num_Xi_y[6] = cnt[4]
num_Xi_y[0] = cnt[5]

# Without Laplace
prob = [x / num_y for x in num_Xi_y]

# With Laplace
alpha = 0.1
prob_l = [(x+alpha) / num_y+(alpha*len(cate_list)) for x in num_Xi_y]
```

- Without Laplace

- With Laplace (alpha = 0.1)