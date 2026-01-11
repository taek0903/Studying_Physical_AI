# bfs.py

def my_bfs(graph, start_node):
    # 1
    queue = list()      # 큐(빈 리스트) 생성

    # 2
    visited = list()    # 방문 확인용 리스트 생성

    # 3
    queue.append(start_node)    # 시작 노드 큐에 삽입

    while queue:
        # 4
        node = queue.pop(0)     # 큐에서 노드 꺼냄  
        # 5
        if node not in visited:         # 방문 하지 않았다면
            queue.extend(graph[node])   # 인접 노드를 큐에 삽입
            visited.append(node)        # 방문 처리
    
    print(f'bfs - {visited}')
    return visited      # 방문 순서 확인

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

my_bfs(MyGraph,'A')
