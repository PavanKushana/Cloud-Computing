from operator import attrgetter
import random, sys, time, copy
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False 

class WSN(object):
    xm    = 200  # Length of the yard
    ym    = 200  # Width of the yard
    n     = 100  # total number of nodes
    sink  = None # Sink node
    nodes = None # All sensor nodes set
    # Energy model (all values in Joules)
    # Eelec = ETX = ERX
    ETX = 50 * (10 ** (-9))   
    ERX = 50 * (10 ** (-9))      

    Efs = 10 * (10 ** (-12))     
    Emp = 0.0013 * (10 ** (-12)) 
    EDA = 5 * (10 ** (-9))      
    f_r = 0.6                    
    CM = 32     
    DM = 4096 
    do = np.sqrt(Efs / Emp) 

    m_n = 3 
    

    n_dead           = 0 # The number of dead nodes
    flag_first_dead  = 0 # Flag tells that the first node died
    flag_all_dead    = 0 # Flag tells that all nodes died
    flag_net_stop    = 0 # Flag tells that network stop working:90% nodes died
    round_first_dead = 0 # The round when the first node died
    round_all_dead   = 0 # The round when all nodes died
    round_net_stop   = 0 # The round when the network stop working
    
    def dist(x, y):
        distance = np.sqrt(np.power((x.xm - y.xm), 2) + np.power((x.ym - y.ym), 2))
        return distance
        
    def trans_energy(data, dis):
        if dis > WSN.do:
            energy = WSN.ETX * data + WSN.Emp * data * (dis ** 4)
        else: 
            energy = WSN.ETX * data + WSN.Efs * data * (dis ** 2)
        return energy
        
    def node_state(r):
        global nodes
        nodes  = WSN.nodes
        n_dead = 0
        for node in nodes:
            if node.energy <= Node.energy_threshold:
                n_dead += 1
                if WSN.flag_first_dead == 0 and n_dead == 1:
                    WSN.flag_first_dead  = 1
                    WSN.round_first_dead = r - Leach.r_empty
        if WSN.flag_net_stop == 0 and n_dead >= (WSN.n * 0.9):
            WSN.flag_net_stop  = 1
            WSN.round_net_stop = r - Leach.r_empty
        if n_dead == WSN.n - 1:
            WSN.flag_all_dead  = 1
            WSN.round_all_dead = r - Leach.r_empty
        WSN.n_dead = n_dead
    
class Node(object):
    energy_init      = 0.5 

    energy_threshold = 0.001
    
    def __init__(self):
        self.id      = None # 节点编号
        self.xm      = np.random.random() * WSN.xm
        self.ym      = np.random.random() * WSN.ym
        self.energy  = Node.energy_init
        self.type    = "N" 
        self.G       = 0 
        self.head_id = None 
        
    def init_nodes():
        nodes = []
        for i in range(WSN.n):
            node    = Node()
            node.id = i
            nodes.append(node)
        # Initial sink node
        sink    = Node()
        sink.id = -1
        sink.xm  = 0.5 * WSN.xm # x coordination of base station
        sink.ym  = 50 + WSN.ym # y coordination of base station
        WSN.nodes = nodes
        WSN.sink  = sink
        
    def init_malicious_nodes():
        for i in range(WSN.m_n):
            node = Node()
            node.id = WSN.n + i
            WSN.nodes.append(node)
    
    def plot_wsn():
        nodes = WSN.nodes
        n = WSN.n
        m_n = WSN.m_n
        sink = WSN.sink
        plt.plot([sink.xm], [sink.ym], 'r^')
        n_flag = True
        for i in range(n):
            if n_flag:
                plt.plot([nodes[i].xm], [nodes[i].ym], 'b+')
                n_flag = False
            else:
                plt.plot([nodes[i].xm], [nodes[i].ym], 'b+')
        m_flag = True
        for i in range(m_n):
            j = n + i
            if m_flag:
                plt.plot([nodes[j].xm], [nodes[j].ym], 'kd')
                m_flag = False
            else:
                plt.plot([nodes[j].xm], [nodes[j].ym], 'kd')
        plt.legend()
        plt.xlabel('X/m')
        plt.ylabel('Y/m')
        plt.show()
        
class Leach(object):
    p       = 0.1
    period  = int(1/p) 
    heads   = None
    members = None 
    cluster = None 
    r       = 0
    rmax    = 5
    r_empty = 0 
    
    def show_cluster():
        fig = plt.figure()
        plt.xlabel('X/m')
        plt.ylabel('Y/m')
        icon = ['o', '*', '.', 'x', '+', 's']
        color = ['r', 'b', 'g', 'c', 'y', 'm']
        i = 0
        nodes = WSN.nodes
        for key, value in Leach.cluster.items():
            cluster_head = nodes[int(key)]
            for index in value:
                plt.plot([cluster_head.xm, nodes[index].xm], [cluster_head.ym, nodes[index].ym], 
                         c=color[i % 6], marker=icon[i % 5], alpha=0.4)
                if index >= WSN.n:
                    plt.plot([nodes[index].xm], [nodes[index].ym], 'dk')
            i += 1
        plt.show()
        
    def optimum_number_of_clusters():
        N = WSN.n - WSN.n_dead
        M = np.sqrt(WSN.xm * WSN.ym)
        d_toBS = np.sqrt((WSN.sink.xm - WSN.xm) ** 2 +
                         (WSN.sink.ym - WSN.ym) ** 2)
        k_opt = (np.sqrt(N) / np.sqrt(2 * np.pi) * 
                 np.sqrt(WSN.Efs / WSN.Emp) *
                 M / (d_toBS ** 2))
        p = int(k_opt) / N
        return p
    
    def cluster_head_selection():
        nodes   = WSN.nodes
        n       = WSN.n
        heads   = Leach.heads = []
        members = Leach.members = []
        p       = Leach.p
        r       = Leach.r
        period  = Leach.period
        Tn      = p / (1 - p * (r % period)) 
        print(Leach.r, Tn)
        for i in range(n):
            if nodes[i].energy > Node.energy_threshold: 
                if nodes[i].G == 0: 
                    temp_rand = np.random.random()
                    if temp_rand <= Tn:
                        nodes[i].type = "CH" 
                        nodes[i].G = 1 
                        heads.append(nodes[i])
                        max_dis = np.sqrt(WSN.xm ** 2 +  WSN.ym ** 2)
                        nodes[i].energy -= WSN.trans_energy(WSN.CM, max_dis)
                if nodes[i].type == "N": 
                    members.append(nodes[i])
        m_n = WSN.m_n
        for i in range(m_n):
            j = n + i
            members.append(nodes[j])
        if not heads:
            Leach.r_empty += 1
            print("---> 本轮未找到簇头！")
        print("The number of CHs is:", len(heads), (WSN.n - WSN.n_dead))
        return None # heads, members
                
    def cluster_formation():
        nodes   = WSN.nodes
        heads   = Leach.heads
        members = Leach.members
        cluster = Leach.cluster = {} 
        if not heads:
            return None
        for head in heads:
            cluster[str(head.id)] = [] 
        for member in members:
            min_dis = np.sqrt(WSN.xm ** 2 +  WSN.ym ** 2) 
            head_id = None
            member.energy -= WSN.ERX * WSN.CM * len(heads)
            for head in heads:
                tmp = WSN.dist(member, head)
                if tmp <= min_dis:
                    min_dis = tmp
                    head_id = head.id
            member.head_id = head_id 
            member.energy -= WSN.trans_energy(WSN.CM, min_dis)
            head = nodes[head_id]
            head.energy -= WSN.ERX * WSN.CM
            cluster[str(head_id)].append(member.id) 
        for key, values in cluster.items():
            head = nodes[int(key)]
            if not values:
                max_dis = np.sqrt(WSN.xm ** 2 +  WSN.ym ** 2)
                head.energy -= WSN.trans_energy(WSN.CM, max_dis)
                for x in values:
                    member = nodes[int(x)]
                    member.energy -= WSN.ERX * WSN.CM
        return None 
        
    def set_up_phase():
        Leach.cluster_head_selection()
        Leach.cluster_formation()
        
    def steady_state_phase():
        nodes   = WSN.nodes
        cluster = Leach.cluster
        if not cluster:
            return None
        for key, values in cluster.items():
            head     = nodes[int(key)]
            n_member = len(values) # 簇成员数量
            for x in values:
                member = nodes[int(x)]
                dis    = WSN.dist(member, head)
                member.energy -= WSN.trans_energy(WSN.DM, dis) 
                head.energy   -= WSN.ERX * WSN.DM
            d_h2s = WSN.dist(head, WSN.sink)
            if n_member == 0: 
                energy = WSN.trans_energy(WSN.DM, d_h2s)
            else:
                new_data = WSN.DM * (n_member + 1)
                E_DA     = WSN.EDA * new_data 
                if WSN.f_r == 0:
                    new_data_ = WSN.DM
                else:
                    new_data_ = new_data * WSN.f_r 
                E_Trans  = WSN.trans_energy(new_data_, d_h2s)
                energy = E_DA + E_Trans
            head.energy -= energy
            
    def Leach():
        Leach.set_up_phase()
        Leach.steady_state_phase()
        
    def run_Leach():
        for r in range(Leach.rmax):
            Leach.r = r
            nodes   = WSN.nodes
            if (r % Leach.period) == 0:
                print("==============================")
                for node in nodes:
                    node.G = 0
            for node in nodes:
                node.type = "N"
            Leach.Leach()
            WSN.node_state(r)
            if WSN.flag_all_dead:
                print("==============================")
                break
            Leach.show_cluster()

def main():
    Node.init_nodes()
    Node.init_malicious_nodes()
    Node.plot_wsn()
    Leach.run_Leach()
    
    
if __name__ == '__main__':
    main()    



class Graph:

	def __init__(self, amount_vertices):
		self.edges = {} 
		self.vertices = set() 
		self.amount_vertices = amount_vertices 



	def addEdge(self, src, dest, cost = 0):
		if not self.existsEdge(src, dest):
			self.edges[(src, dest)] = cost
			self.vertices.add(src)
			self.vertices.add(dest)


	def existsEdge(self, src, dest):
		return (True if (src, dest) in self.edges else False)


	def showGraph(self):
		print('Showing the graph:\n')
		for edge in self.edges:
			print('%d linked in %d with cost %d' % (edge[0], edge[1], self.edges[edge]))

	def getCostPath(self, path):
		
		total_cost = 0
		for i in range(self.amount_vertices - 1):
			total_cost += self.edges[(path[i], path[i+1])]

		total_cost += self.edges[(path[self.amount_vertices - 1], path[0])]
		return total_cost


	def getRandomPaths(self, max_size):

		random_paths, list_vertices = [], list(self.vertices)

		initial_vertice = random.choice(list_vertices)
		if initial_vertice not in list_vertices:
			print('Error: initial vertice %d not exists!' % initial_vertice)
			sys.exit(1)

		list_vertices.remove(initial_vertice)
		list_vertices.insert(0, initial_vertice)

		for i in range(max_size):
			list_temp = list_vertices[1:]
			random.shuffle(list_temp)
			list_temp.insert(0, initial_vertice)

			if list_temp not in random_paths:
				random_paths.append(list_temp)

		return random_paths


class CompleteGraph(Graph):

	def generates(self):
		for i in range(self.amount_vertices):
			for j in range(self.amount_vertices):
				if i != j:
					weight = random.randint(1, 10)
					self.addEdge(i, j, weight)


class Particle:

	def __init__(self, solution, cost):

		# current solution
		self.solution = solution

		# best solution (fitness) it has achieved so far
		self.pbest = solution

		# set costs
		self.cost_current_solution = cost
		self.cost_pbest_solution = cost


		self.velocity = []

	def setPBest(self, new_pbest):
		self.pbest = new_pbest

	# returns the pbest
	def getPBest(self):
		return self.pbest

	def setVelocity(self, new_velocity):
		self.velocity = new_velocity

	def getVelocity(self):
		return self.velocity

	def setCurrentSolution(self, solution):
		self.solution = solution

	# gets solution
	def getCurrentSolution(self):
		return self.solution

	# set cost pbest solution
	def setCostPBest(self, cost):
		self.cost_pbest_solution = cost

	# gets cost pbest solution
	def getCostPBest(self):
		return self.cost_pbest_solution

	# set cost current solution
	def setCostCurrentSolution(self, cost):
		self.cost_current_solution = cost

	# gets cost current solution
	def getCostCurrentSolution(self):
		return self.cost_current_solution

	# removes all elements of the list velocity
	def clearVelocity(self):
		del self.velocity[:]


class PSO:

	def __init__(self, graph, iterations, size_population, beta=1, alfa=1):
		self.graph = graph # the graph
		self.iterations = iterations # max of iterations
		self.size_population = size_population # size population
		self.particles = [] # list of particles
		self.beta = beta # the probability that all swap operators in swap sequence (gbest - x(t-1))
		self.alfa = alfa # the probability that all swap operators in swap sequence (pbest - x(t-1))

		solutions = self.graph.getRandomPaths(self.size_population)

		if not solutions:
			print('Initial population empty! Try run the algorithm again...')
			sys.exit(1)

		for solution in solutions:
			particle = Particle(solution=solution, cost=graph.getCostPath(solution))
			self.particles.append(particle)

		self.size_population = len(self.particles)


	def setGBest(self, new_gbest):
		self.gbest = new_gbest

	def getGBest(self):
		return self.gbest


	def showsParticles(self):

		print('Showing particles...\n')
		for particle in self.particles:
			print('pbest: %s\t|\tcost pbest: %d\t|\tcurrent solution: %s\t|\tcost current solution: %d' \
				% (str(particle.getPBest()), particle.getCostPBest(), str(particle.getCurrentSolution()),
							particle.getCostCurrentSolution()))
		print('')


	def run(self):

		for t in range(self.iterations):

			self.gbest = min(self.particles, key=attrgetter('cost_pbest_solution'))

			for particle in self.particles:

				particle.clearVelocity() # cleans the speed of the particle
				temp_velocity = []
				solution_gbest = copy.copy(self.gbest.getPBest()) # gets solution of the gbest
				solution_pbest = particle.getPBest()[:] # copy of the pbest solution
				solution_particle = particle.getCurrentSolution()[:] # gets copy of the current solution of the particle

				for i in range(self.graph.amount_vertices):
					if solution_particle[i] != solution_pbest[i]:
						swap_operator = (i, solution_pbest.index(solution_particle[i]), self.alfa)

						temp_velocity.append(swap_operator)

						aux = solution_pbest[swap_operator[0]]
						solution_pbest[swap_operator[0]] = solution_pbest[swap_operator[1]]
						solution_pbest[swap_operator[1]] = aux

				for i in range(self.graph.amount_vertices):
					if solution_particle[i] != solution_gbest[i]:
						swap_operator = (i, solution_gbest.index(solution_particle[i]), self.beta)

						temp_velocity.append(swap_operator)


						aux = solution_gbest[swap_operator[0]]
						solution_gbest[swap_operator[0]] = solution_gbest[swap_operator[1]]
						solution_gbest[swap_operator[1]] = aux

				

				particle.setVelocity(temp_velocity)


				for swap_operator in temp_velocity:
					if random.random() <= swap_operator[2]:

						aux = solution_particle[swap_operator[0]]
						solution_particle[swap_operator[0]] = solution_particle[swap_operator[1]]
						solution_particle[swap_operator[1]] = aux
				

				particle.setCurrentSolution(solution_particle)

				cost_current_solution = self.graph.getCostPath(solution_particle)

				particle.setCostCurrentSolution(cost_current_solution)


				if cost_current_solution < particle.getCostPBest():
					particle.setPBest(solution_particle)
					particle.setCostPBest(cost_current_solution)
		

if __name__ == "__main__":
	

	graph = Graph(amount_vertices=5)


	graph.addEdge(0, 1, 1)
	graph.addEdge(1, 0, 1)
	graph.addEdge(0, 2, 3)
	graph.addEdge(2, 0, 3)
	graph.addEdge(0, 3, 4)
	graph.addEdge(3, 0, 4)
	graph.addEdge(0, 4, 5)
	graph.addEdge(4, 0, 5)
	graph.addEdge(1, 2, 1)
	graph.addEdge(2, 1, 1)
	graph.addEdge(1, 3, 4)
	graph.addEdge(3, 1, 4)
	graph.addEdge(1, 4, 8)
	graph.addEdge(4, 1, 8)
	graph.addEdge(2, 3, 5)
	graph.addEdge(3, 2, 5)
	graph.addEdge(2, 4, 1)
	graph.addEdge(4, 2, 1)
	graph.addEdge(3, 4, 2)
	graph.addEdge(4, 3, 2)

	# creates a PSO instance
	pso = PSO(graph, iterations=100, size_population=10, beta=1, alfa=0.9)
	pso.run()
	pso.showsParticles() 


	print('gbest: %s | cost: %d\n' % (pso.getGBest().getPBest(), pso.getGBest().getCostPBest()))


shalu1 = []
for i in range(1, 103):
    shalu=nodes[i]
    shalu=shalu.xm
    shalu1.append(shalu)        

shalu=shalu1.sort(reverse=True) 
import pandas as pd
df = pd.DataFrame(shalu1)
df.columns =['Node'] 
df.plot();
plt.ylabel('Number of Cluster ')
plt.xlabel('Transmission Range')
plt.title('Number of Node 200  ')
plt.show() 
df.to_csv('out.csv')
