import time
from pyamaze import maze, agent
from queue import PriorityQueue

ROWS = 10
COLS = 10

# Funció que retorna la distància Manhattan
def distance(cell1, cell2):
    x1, y1 = cell1
    x2, y2 = cell2
    return abs(x1 - x2) + abs(y1 - y2)

def aStar(m):
    # Posició d'inici (cantonada inferior dreta)
    start = (m.rows, m.cols)

    # Diccionari amb els costos per arribar a la cel·la actual des de la cel·la d'inici
    g_score = {cell:float('inf') for cell in m.grid}

    # Posem tots els costos a infinit menys el de la cel·la d'inici
    g_score[start] = 0


    # Diccionari amb els costos totals per arribar a la cel·la actual
    f_score = {cell:float('inf') for cell in m.grid}

    # Posem el cost de la casella d'inici que és la distància Manhattan que hi ha fins la casella final
    f_score[start] = distance(start, (1,1))
    
    # La priority queue serveix per agafar la tupla de menys valor
    pq = PriorityQueue()
    pq.put((distance(start, (1,1)), distance(start, (1,1)), start))

    aPath = {}

    # Mentre que la Priority Queue no estigui buida o s'hagi arribat al destí
    while not pq.empty():

        # Agafem el valor actual de la cel·la
        cela_actual = pq.get()[2]

        if cela_actual == (1,1):
            break

        for d in 'EWNS':
            if m.maze_map[cela_actual][d]==True:
                if d == 'E':
                    cela_seguent = (cela_actual[0], cela_actual[1]+1)
                if d == 'W':
                    cela_seguent = (cela_actual[0], cela_actual[1]-1)
                if d == 'N':
                    cela_seguent = (cela_actual[0]-1, cela_actual[1])
                if d == 'S':
                    cela_seguent = (cela_actual[0]+1, cela_actual[1])

                aux_g_score = g_score[cela_actual]+1
                aux_f_score = aux_g_score + distance(cela_seguent, (1,1))

                if aux_f_score < f_score[cela_seguent]:
                    g_score[cela_seguent] = aux_g_score
                    f_score[cela_seguent] = aux_f_score
                    pq.put((aux_f_score, distance(cela_seguent, (1,1)), cela_seguent))
                    aPath[cela_seguent] = cela_actual


    forwardPath = {}
    cela = (1,1)

    while cela != start:
        forwardPath[aPath[cela]] = cela
        cela = aPath[cela]

    return forwardPath

m = maze(ROWS,COLS)
m.CreateMaze()
pre_Astar = time.time()
path = aStar(m)
post_Astar = time.time()
a = agent(m, footprints=True)
print(post_Astar - pre_Astar)
m.tracePath({a:path},delay=10)
# m.enableWASD(a)
m.run()
# print(m.maze_map)
