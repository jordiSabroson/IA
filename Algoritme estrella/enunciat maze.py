from pyamaze import maze, agent
from queue import PriorityQueue

ROWS = 4
COLS = 4

def distance(cell1, cell2):
    x1, y1 = cell1
    x2, y2 = cell2
    return abs(x1 - x2) + abs(y1 - y2)

def aStar(m):
    start = (m.rows, m.cols)
    pq = PriorityQueue()
    pq.put((distance(start, (1,1)), distance(start, (1,1)), start))
    forwardPath = {}
    return forwardPath

m = maze(ROWS,COLS)
m.CreateMaze()
# pre_Astar = time.time()
# # path = aStar(m)
# post_Astar = time.time()
a = agent(m, footprints=True)
# print(post_Astar - pre_Astar)
# m.tracePath({a:path},delay=5)
# m.run()