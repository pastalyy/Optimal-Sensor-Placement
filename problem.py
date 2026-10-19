from individual import Individual
import topology
import random
import numpy as np
import time


def matrix_and(a, b):
    n = a.shape[0]
    c = np.zeros((n, n), dtype=int)
    for i in range(n):
        c[i] = a[i] & b[i]
    return c


class Problem:

    def __init__(self, objectives, node_num,upstream_arr,upstream_set,priority,conn_dict={},sensor_upbound=100,coverage_constrain=0,graph=None,sol={},allowed_types=(1,2),type_upbounds=None):
        self.num_of_objective=len(objectives)
        self.objectives=objectives
        self.node_num=node_num
        self.upstream_arr=upstream_arr
        self.upstream_set=upstream_set
        self.sensor_upbound=sensor_upbound
        # 限制sensor个数的上界
        self.graph=graph
        self.priority=priority
        self.conn_dict=conn_dict
        self.coverage_constrain=coverage_constrain
        self.sol=sol
        self.allowed_types=list(allowed_types)  # which sensor types (1=pulse,2=level) may be used
        # type_upbounds: dict mapping sensor type -> max number of that type
        # e.g. {1: 30, 2: 10} means "at most 30 pulse sensors, at most 10 level
        # sensors" - independent budgets, matching a real utility's actual
        # sensor inventory rather than one combined budget.
        # Defaults to sensor_upbound for every allowed type (old behaviour)
        # if not explicitly given.
        if type_upbounds is None:
            self.type_upbounds = {t: sensor_upbound for t in self.allowed_types}
        else:
            self.type_upbounds = type_upbounds
        # 限制新生成的individual的coverage

        # self.conn_matrix=conn_matrix


    # @profile
    def generate_individual(self):
        individual=Individual(self.node_num)
        all_nodes = list(range(self.node_num))
        random.shuffle(all_nodes)

        # For each allowed sensor type, independently choose how many of
        # that type to place (up to its own budget), then take that many
        # nodes from a shared shuffled pool (so pulse and level sensors
        # never claim the same node).
        cursor = 0
        positive_nodes = []
        for sensor_type in self.allowed_types:
            max_for_type = self.type_upbounds.get(sensor_type, 1)
            max_for_type = max(1, min(max_for_type, self.node_num - cursor))
            count_for_type = random.randint(1, max_for_type) if max_for_type >= 1 else 0
            chosen = all_nodes[cursor:cursor+count_for_type]
            cursor += count_for_type
            for i in chosen:
                individual.chromosome[i] = sensor_type
                positive_nodes.append(i)

        individual.positive_nodes=positive_nodes
        return individual

    def generate_individual_with(self,individual1):
        '''
        create a copy of an Individual

        Parameters
        ----------
        individual1 : TYPE Individual
            DESCRIPTION. An individual to create a copy of.

        Returns
        -------
        individual : TYPE Individual
            DESCRIPTION. An independent copy of individual1

        '''
        individual=Individual(self.node_num)
        individual.chromosome=individual1.chromosome.copy()
        # individual.Gsub=individual1.Gsub
        individual.positive_nodes=individual1.positive_nodes.copy()
        return individual


    def generate_individual_with_positive_nodes(self, positive_nodes):
        individual = Individual(self.node_num)
        individual.positive_nodes = positive_nodes
        for i in positive_nodes:
            individual.chromosome[i] = 1
        self.calculate_objectives(individual)

        return individual

    # @profile
    def calculate_objectives(self, individual):
        # using the Gsub
        individual.Gsub=topology.extract_connectivity_fastdict(self.graph, individual.chromosome,self.conn_dict)

        downstream_node = [i[0] for i in individual.Gsub.out_degree() if i[1] == 0]  # 每个individual中outdegree为0的node
        coverage_set = set()
        for i in downstream_node:
            coverage_set = coverage_set | self.upstream_set[i]  # 按位或运算符


        individual.coverage_set=coverage_set
        individual.objectives=[f(self,individual) for f in self.objectives]
        individual.positive_nodes = np.array(np.where(individual.chromosome == 1))[0].tolist()