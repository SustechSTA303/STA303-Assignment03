from typing import List
from plot_underground_path import plot_path
from build_data import Station, build_data
import argparse
import time


#the cost function

#曼哈顿距离 L-1范数
def L1(a : str, b : str): # a and b are two different stations name
    x1,y1,x2,y2 = stations[a].position[0],stations[a].position[1],stations[b].position[0],stations[b].position[1]
    return abs(x1-x2) + abs(y1-y2)
# 欧几里得距离 L-2范数
def L2(a : str, b : str): 
    x1,y1,x2,y2 = stations[a].position[0],stations[a].position[1],stations[b].position[0],stations[b].position[1]
    return ((x1-x2)**2 + (y1-y2)**2)**.5
#对角距离  L-无穷范数
def Linf(a : str, b : str): 
    x1,y1,x2,y2 = stations[a].position[0],stations[a].position[1],stations[b].position[0],stations[b].position[1]   
    return max(abs(x1-x2),abs(y1-y2))
#Dijstra算法用的函数 返回恒定为0
def L_0(a : str, b : str):
    return 0

def astar(start,end,h,b2):      #start and end are str, the name of stations , h is the cost function u use
    """
    cost f,history cost g,future cost h,father node parent_idx,is_in_openlist,is_in_closedlist
    """
    class Node:
        def __init__(self,name,f,g,h,parent_name,is_in_openlist,is_in_closedlist):
            self.name = name
            self.f = f
            self.g = g
            self.h = h
            self.parent_name = parent_name
            self.is_in_openlist = is_in_openlist
            self.is_in_closedlist = is_in_closedlist

    
    openlist = []
    closedlist= []
    nodes = {}
    
    #将所有station写入到节点中  
    for station in stations:
        node = Node(station,0,0,0,None,0,0)
        nodes[station] = node
    
    #完善初始节点的相关属性
    nodes[start].h = h(start,end)*b2
    openlist.append(nodes[start])
    nodes[start].is_in_openlist = 1
    
    while openlist!={}:
        A=[]
        #选择一个代价f最小的节点
        current_node = min(openlist, key=lambda x: x.f)
        current_station = stations[current_node.name]
        #拓展current_node，将其所有子节点放入临时集合A中
        for child_station in current_station.links:
            child_node = nodes[child_station.name]
            A.append(child_node)
        #开始访问每一个子节点并进行处理
        reach = False
        for child_node in A:
            if(child_node.name == end):
                reach = True
            if(child_node.is_in_closedlist==1):
                continue
            if(child_node.is_in_openlist==0):
                child_node.parent_name = current_node.name
                #子节点的历史代价=子节点-父节点的L2距离+父节点.g
                child_node.g = L2(child_node.name,child_node.parent_name)+nodes[child_node.parent_name].g
                child_node.h = h(child_node.name,end)*b2
                child_node.f = child_node.g + child_node.h
                openlist.append(child_node)
                child_node.is_in_openlist = 1

            #如果有子节点之前已经遇到过，那么需要选择历史代价更小的作为父节点（因为未来代价一致）
            else:
                oldg = child_node.g
                newg = current_node.g + L2(child_node.name,child_node.parent_name)
                if(oldg > newg):
                    child_node.parent_name = current_node.name
                    child_node.g = newg
                    child_node.f = child_node.g + child_node.h

        #将current_node从openlist 移入 closedlist
        openlist.remove(current_node)
        current_node.is_in_openlist = 0  
        current_node.is_in_closedlist = 1       
        if(reach):
            break
            
    B = [nodes[end]]
    while (B[-1].name != start):
        B.append(nodes[B[-1].parent_name])
        
    reversed_B = list(reversed(B))
    C = []
    for node in reversed_B:
        C.append(node.name)
        print(node.name)
       
    return C


def BFS(start,end):
    start_time = time.time()
    class Node:
        def __init__(self,name,f,parent_name,is_in_openlist,is_in_closedlist):
            self.name = name
            self.f = f
            self.parent_name = parent_name
            self.is_in_openlist = is_in_openlist
            self.is_in_closedlist = is_in_closedlist

    
    openlist = []
    closedlist= []
    nodes = {}
    
    #将所有station写入到节点中  
    for station in stations:
        node = Node(station,0,None,0,0)
        nodes[station] = node
    
    openlist.append(nodes[start])
    nodes[start].is_in_openlist = 1
    
    while openlist!={}:
        A=[]
        #选择一个代价f最小的节点
        current_node = min(openlist, key=lambda x: x.f)
        current_station = stations[current_node.name]
        #拓展current_node，将其所有子节点放入临时集合A中
        for child_station in current_station.links:
            child_node = nodes[child_station.name]
            A.append(child_node)
        #开始访问每一个子节点并进行处理
        reach = False
        for child_node in A:
            if(child_node.name == end):
                reach = True
            if(child_node.is_in_closedlist==1):
                continue
            if(child_node.is_in_openlist==0):
                child_node.parent_name = current_node.name
                #子节点的历史代价=子节点-父节点的L2距离+父节点.g
                child_node.f = 1 + nodes[child_node.parent_name].f
                openlist.append(child_node)
                child_node.is_in_openlist = 1

            #如果有子节点之前已经遇到过，那么需要选择历史代价更小的作为父节点（因为未来代价一致）
            else:
                oldf = child_node.f
                newf = current_node.f + 1
                if(oldf > newf):
                    child_node.parent_name = current_node.name
                    child_node.f = newf

        #将current_node从openlist 移入 closedlist
        openlist.remove(current_node)
        current_node.is_in_openlist = 0  
        current_node.is_in_closedlist = 1       
        if(reach):
            break
            
    B = [nodes[end]]
    while (B[-1].name != start):
        B.append(nodes[B[-1].parent_name])
        
    reversed_B = list(reversed(B))
    C = []
    for node in reversed_B:
        s = node.name
        C.append(s)
        
    end_time = time.time()
    
    time1 = (end_time-start_time)*1000
    
    return C
           

# Implement the following function
def get_path(start_station_name: str, end_station_name: str, map: dict[str, Station]): 
    
    start_time = time.time()
    # astar(start,end,L1,b1)
    # L1,L2,Linf,L_0 are the future cost you want to use, and b1 are the coefficient of the future cost
    #C = astar(start_station_name,end_station_name,L_0,1)
    ## you could use BFS(start,end) to find another path as well
    C = BFS(start_station_name,end_station_name)
    end_time = time.time()
    print((end_time-start_time)*1000)
    
    return C






if __name__ == '__main__':

    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('start_station_name', type=str, help='start_station_name')
    parser.add_argument('end_station_name', type=str, help='end_station_name')
    args = parser.parse_args()
    start_station_name = args.start_station_name
    end_station_name = args.end_station_name

    # The relevant descriptions of stations and underground_lines can be found in the build_data.py
    stations, underground_lines = build_data()
    path = get_path(start_station_name, end_station_name, stations)
    # visualization the path
    # Open the visualization_underground/my_path_in_London_railway.html to view the path, and your path is marked in red
    plot_path(path, 'visualization_underground/my_shortest_path_in_London_railway.html', stations, underground_lines)
