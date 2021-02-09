import pprint


class Tree_node:
    def __init__(self, name, count, fa_point):
        self.name = name
        self.count = count
        self.fa_point = fa_point
        self.node_link = None
        self.child = {}

    def show(self):
        '''
        打印整课树
        :return:
        '''
        pprint.pprint(self.name + " " + str(self.count))
        for item in self.child.values():
            item.show()
        print("==================")


def data_set(D):
    '''
    将交易集转换成 key-value 形式 方便处理
    :param D: 交易集
    :return:  key-value形式交易集
    '''
    dataSet = {}
    for t in D:
        key = frozenset(t)
        if key in dataSet.keys():
            dataSet[key] += 1
        else:
            dataSet[key] = 1
    return dataSet


def createHeaderTable(dataSet, min_sup):
    '''
    根据交易集和最小支持度，建立头表
    :param dataSet: 交易集
    :param min_sup: 最小支持度
    :return:   头表
    '''
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item,0) + dataSet[trans]
            #  get(key, default)
    finalHeaderTable = {}
    for key, value in headerTable.items():
        if headerTable[key] < min_sup:
            # del(headerTable[key])
            continue
        else:
            finalHeaderTable[key] = value
    del headerTable
    for key in finalHeaderTable.keys():
        finalHeaderTable[key] = [finalHeaderTable[key], None]    # [频度， node_link]
    return finalHeaderTable


def create(dataSet, min_sup):
    '''
    建立一个FP-Tree
    :param dataSet:  交易集
    :param min_sup:  最小支持度
    :return:  FP树, 头表
    '''
    headerTable = createHeaderTable(dataSet, min_sup) # 建立头表
    print("******************")
    pprint.pprint(headerTable)
    print("******************")
    root = Tree_node("root", 0, None)
    frequent_item = set(headerTable.keys())
    if len(frequent_item) == 0: return None,None
    for trans, count in dataSet.items():
        temp_D = {}
        for item in trans:
            if item in frequent_item:
                temp_D[item] = headerTable[item][0]
        # pprint.pprint(temp_D)
        if temp_D:
            sortItem = [v[0] for v in sorted(temp_D.items(), key=lambda p:p[1], reverse=True)]
            # pprint.pprint(sortItem)
            print(count, "\t\t\t", sortItem)
            # 按照频繁项大小排序
            buildTree(root, headerTable, sortItem, count)
    return root, headerTable


def buildTree(root, headerTable, sortItem, count):
    '''
    用项集元素来更新或建立树的节点
    :param root: 根节点（父节点）
    :param headerTable: 头表
    :param sortItem: 排序后的项集
    :param count: 项集的次数
    :return:
    '''
    key  = sortItem[0]
    if key in root.child.keys():
        root.child[key].count += count                  #增加计数
    else:
        root.child[key] = Tree_node(key, count, root)   #新节点
        if headerTable[key][1] == None:                 # 头表未连接
            headerTable[key][1] = root.child[key]       # 将头表和树节点连接起来
        else:                                           #用node_link 将相同的name的树节点连接起来
            cur_node = headerTable[key][1]              # headerTable[key][1] 中存储着树节点地址
            while (cur_node.node_link != None):
                cur_node = cur_node.node_link            # 指向下一节点
            cur_node.node_link = root.child[key]         # 连接新节点
    if len(sortItem) > 1:
        buildTree(root.child[key], headerTable, sortItem[1:], count) # 递归建树


def findPath(table_item):
    '''
    根据表项寻找路径，并将路径组合成新的数据集
    :param table_item:
    :return:
    '''
    data_set = {}
    tree_node = table_item
    while tree_node != None:
        path = []
        father = tree_node.fa_point
        while father.fa_point != None:
            path.append(father.name)
            father = father.fa_point
        if path:
            pprint.pprint(path)
            data_set[frozenset(path)] = tree_node.count
        tree_node = tree_node.node_link
    return data_set


def mining(root, headerTable, min_sup, fre_set, fre_list):
    '''
    从FP-Tree中挖掘出所有的频繁集
    :param root:  FP树
    :param headerTable:  头表
    :param min_sup: 最小支持度
    :param fre_set:  频繁集
    :param fre_list:  频繁集列表
    :return:
    '''
    items = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[1][0])]  #头表降序排序
    # pprint.pprint(items)
    for item in items:
        newFreqSet = fre_set.copy()                 # 深拷贝
        newFreqSet.add(item)                        #
        fre_list.append((newFreqSet, headerTable[item][0]))     # 表头元素的支持度 为当前 频繁集 的支持度
        # pprint.pprint(item)
        data_set = findPath(headerTable[item][1])               # 从表项得到一个新的数据集
        # pprint.pprint(data_set)
        tree, header = create(data_set, min_sup)                # 建立一个新的FP-Tree, 头表
        if header != None:
            mining(tree, header, min_sup, newFreqSet, fre_list)     # 递归 挖掘


if __name__ == '__main__':
    # 交易集
    D = [["bread", "cream", "milk", "tea"],
            ["bread", "milk", "cream"],
            ["cake", "milk"],
            ["tea", "milk"],
            ["bread", "cake", "milk"],
            ["bread", "tea"],
            ["bread", "tea"],
            ["beer", "milk", "tea"],
            ["bread", "cream", "milk", "tea"],
            ["bread", "milk", "tea"],
        ]
    # 最小支持度
    min_sup = 3
    dataSet = data_set(D)
    pprint.pprint(dataSet)
    root, headerTable = create(dataSet, min_sup)
    root.show()
    frequent_list = []
    mining(root, headerTable, min_sup, set([]), frequent_list)
    # pprint.pprint(frequent_list)
    pprint.pprint(sorted(frequent_list, key=lambda p: len(p[0])))

