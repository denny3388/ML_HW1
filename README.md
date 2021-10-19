# ML HW1
> 0716081 葉晨

## Project preview
1. Data Input
2. Data Visualization
3. Data Preprocessing
4. Model Construction
5. Train-Test-Split
6. Results
7. Compare and Conclusion
8. Question

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

## Data Visualization
