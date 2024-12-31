from function import *
class DecisionTree():
    def __init__(self,data,label_index,atributes_index):
        self.root = self.ID3(data,label_index,atributes_index)
        print(f"root edge {self.root.edge_value}")
    def ID3(self,data,label_index,atributes_index):
        unique_values,count = np.unique(data[:,label_index],return_counts=True)
        root = Node('empty')
        if(len(count)==1):
            root.field = unique_values[0]
            
        elif(atributes_index.size==0): 
            max_count = max(count)
            root.field = unique_values[list(count).index(max_count)]
        else :
            info_gain_lists = [info_gain(data,idx,label_index) for idx in atributes_index]
            chosen_atribute_val = max(info_gain_lists)
            print(f"info gain list is {info_gain_lists}")
            print(f"chosen_atribute is {chosen_atribute_val}")
            root.field = atributes_index[info_gain_lists.index(chosen_atribute_val)]
            print(f"root field is {root.field}")
            edges = np.unique(data[:,root.field])
            root.edge_value = edges
            new_atributes_index = atributes_index.copy()
            new_atributes_index=np.delete(new_atributes_index,np.where(new_atributes_index==root.field))
            for value in root.edge_value:
                subtables =  data[data[:, root.field] == value]       
                root.child.append(self.ID3(subtables,label_index,new_atributes_index))
        
        return root    
    def classify(self,data):
        curr = self.root
        while True :    # su dung while true khong rang buoc co dinh vi co cac cay co chieu dai khac nhau
            curr = curr.child[list(curr.edge_value).index(data[curr.field])]
            if(not curr.child):
                return curr.field
class Node():
    def __init__(self,field):
        self.field=field
        # Lưu ref đến các cây con 
        self.child = []
        # Lưu giá trị các cạnh để khi predict biết chọn nhánh nào
        self.edge_value=[] 
    



"""" some test case"""
#tc1
data = np.array([
    [1, 'a'],
    [2, 'b'],
    [2, 'a'],
    [3, 'a'],
    [4, 'c'],
])

a_index = np.array([])
print(a_index.size)
tree = DecisionTree(data,label_index=1,atributes_index=a_index)
print(tree.root.field)

#tc2
data = np.array([
    [1, 'b'],
    [2, 'b'],
    [2, 'b'],
    [3, 'b'],
    [4, 'b'],
])

a_index = np.array([0])
print(a_index.size)
tree = DecisionTree(data,label_index=1,atributes_index=a_index)
print(tree.root.field)

#tc3
data = np.array([
    [1,'a',0],
    [2, 'b',1],
    [2, 'a',0],
    [3, 'a',0],
    [4, 'c',0],
])

a_index = np.array([0,2])
print(a_index.size)
tree = DecisionTree(data,label_index=1,atributes_index=a_index)
print(tree.root.field)
print(tree.root.edge_value)
print(tree.root.child[1].field)
print(tree.root.child[1].edge_value)
print(tree.root.child[2].field) # expect a


#tc 4
data = np.array([
    [1,'a',0],
    [2, 'b',1],
    [2, 'a',0],
    [3, 'a',0],
    [4, 'c',0],
])

a_index = np.array([0,2])
tree = DecisionTree(data,label_index=1,atributes_index=a_index)
unknow_data = np.array([2,'not know',0])
print(tree.classify(unknow_data))
