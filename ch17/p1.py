# p1.py

graph={
    1 : [2,3,4],
    2 : [1,5],
    3 : [1,6],
    4 : [1,6],
    5 : [2,6],
    6 : [3,5,6]
}

def my_dfs(graph, start_node):
    stack = list()
    visited = list()
    stack.append(start_node)

    while stack:                        # stack에 값이 없을 때 까지 반복
        node = stack.pop()              # stack의 마지막 값을 추출하여 node 변수에 저장
        if node not in visited:         # node 변수 값이 visited 리스트에 없는 값이면
            visited.append(node)        # visitied 리스트에 값을 추가
            stack.extend(reversed(graph[node])) # 딕셔너리에서 키가 node인 값의 value값을 거꾸로 집어넣기 ex [1,2,3] 일 경우 [3,2,1]로 집어 넣기
    print(f'dfs - ',visited)
    return visited                      # 방문한 노드를 반환

my_dfs(graph,1)

def my_bfs(graph, start_node):
    stack = list()
    visited = list()
    stack.append(start_node)

    while stack:
        node = stack.pop(0)         # stack의 첫 부분을 추출 하여 node 변수에 저장
        if node not in visited:
            visited.append(node)
            stack.extend(graph[node])
    print(f'bfs - ',visited)
    return visited

my_bfs(graph,1)