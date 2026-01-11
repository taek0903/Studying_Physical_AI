# dfs.py
MyGraph={
    "A":['B','C',"D"], 
    'B':['A','E'],
    'C':['A','F','G'],
    'D':['A','H'], 
    'E':['B','I'], 
    'F':['C','J'],
    'G':['C'],
    'H':['D'],
    'I':['E'],
    'J':['F']
}

def my_dfs(graph, start_node):
    # 1
    stack=list()
    
    # 2
    visited = list()
    
    # 3
    stack.append(start_node)
    
    # 4
    while stack:
        # 4
        node = stack.pop()

        # 5 
        if node not in visited:
            stack.extend(reversed(graph[node]))
            visited.append(node)
    
    print(f'dfs - ',visited)
    return visited

my_dfs(MyGraph,'A')

