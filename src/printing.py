
import numpy as np
import networkx as nx

class SCMMappingPrinter():
    def __init__(self,A):
        self.A = A
        
    def print_M0_cardinalites(self):
        for n in self.A.M0.nodes():
            print('M0: cardinality of {0}: {1}'.format(n,self.A.M0.get_cardinality(n)))
            
    def print_M1_cardinalites(self):
        for n in self.A.M1.nodes():
            print('M1: cardinality of {0}: {1}'.format(n,self.A.M1.get_cardinality(n)))
            
    def list_DAG_nodes(self):
        print("M0 - Nodes: {0}".format(self.A.M0.nodes))
        print("M1 - Nodes: {0}".format(self.A.M1.nodes))
        
    def list_DAG_edges(self):
        print("M0 - Edges: {0}".format(self.A.M0.edges))
        print("M1 - Edges: {0}".format(self.A.M1.edges))
              
        a_edges = []
        for k in self.A.a.keys():
            print
            a_edges.append((k, self.A.a[k]))                      
        print("a  - Edges: {0}".format(a_edges))
        
    def plot_DAG_M0(self):
        nx.draw(nx.DiGraph(self.A.M0.edges()),with_labels='True')
        
    def plot_DAG_M1(self):
        nx.draw(nx.DiGraph(self.A.M0.edges()),with_labels='True')
        
        
class AbstractionPrinter(SCMMappingPrinter):

    def __init__(self,A):
        super().__init__(A)
    
    def print_R_cardinalites(self):
        for n in self.A.R:
            print('R: cardinality of {0}: {1}'.format(n,self.A.M0.get_cardinality(n)))
    
    def print_relevant_vars(self):
        print(self.A.R)
    
    def print_mapping_a(self):
        print('** The mapping a is indexed by R/M0 **')
        print(self.A.a)
        
    def print_mappings_alphas(self):
        print('** The mappings alpha are indexed by M1 **')
        for k in self.A.alphas.keys():
            domain = self.A.invert_a(k)
            print('Mapping alpha_{0}: {1} -> {2}'.format(k, domain ,k))
    
    def print_mappings_alphas_cardinalities(self):
        print('** The mappings alpha are indexed by M1 **')
        for k in self.A.alphas.keys():
            card_domain,card_codomain = self.A.get_cardinalities_alpha(k)
            print('Mapping alpha_{0}: {1} -> {2}'.format(k, card_domain, card_codomain))
            
    def list_DAG_nodes(self):
        print("M0 - Nodes: {0}".format(self.A.M0.nodes))
        print("M1 - Nodes: {0}".format(self.A.M1.nodes))
        print("R  - Nodes: {0}".format(self.A.R))
                
    def list_FinStoch_objects_M0(self):
        print("Objects (sets) in FinStoch picked by M0:")
        for n in self.A.M0.nodes():
            print("{0}: {1}".format(n, np.arange(self.A.M0.get_cardinality(n))))
        print("** Some sets may be repeated. FinStoch contains also all products. **")
        
    def list_FinStoch_objects_M1(self):
        print("Objects (sets) in FinStoch picked by M1:")
        for n in self.A.M1.nodes():
            print("{0}: {1}".format(n, np.arange(self.A.M1.get_cardinality(n))))
        print("** Some sets may be repeated. FinStoch contains also all products. **")
        
    def list_FinStoch_objects_R(self):
        print("Objects (sets) in FinStoch picked by R:")
        for n in self.A.R:
            print("{0}: {1}".format(n, np.arange(self.A.M0.get_cardinality(n))))
        print("** Some sets may be repeated. FinStoch contains also all products. **")
        
    def list_FinStoch_morphisms_M0(self):
        print("Morphisms (stochastic matrices) in FinStoch picked by M0:")        
        for n in self.A.M0.nodes():
            print("phi_{0}: {1}  ->  {2}".format(n, self.A.M0.get_cpds(n).get_values().shape[1], self.A.M0.get_cpds(n).get_values().shape[0]))
        
    def list_FinStoch_morphisms_M1(self):
        print("Morphisms (stochastic matrices) in FinStoch picked by M1:")
        for n in self.A.M1.nodes():
            print("phi_{0}: {1}  ->  {2}".format(n, self.A.M1.get_cpds(n).get_values().shape[1], self.A.M1.get_cpds(n).get_values().shape[0]))
        
    def list_FinStoch_morphisms_alphas(self):
        print("Morphisms (stochastic matrices) in FinStoch picked by alphas:")
        for k in self.A.alphas.keys():
            print("alpha_{0}: {1}  ->  {2}".format(k, self.A.alphas[k].shape[1], self.A.alphas[k].shape[0]))
    
    def plot_variable_level_mapping(self):
        G = self.A.M0.copy()
        relabel_map = {}
        for n in G.nodes():
            relabel_map[n] = 'M0_'+str(n)
        G0 = nx.relabel.relabel_nodes(G,relabel_map)

        G = self.A.M1.copy()
        relabel_map = {}
        for n in G.nodes():
            relabel_map[n] = 'M1_'+str(n)
        G1 = nx.relabel.relabel_nodes(G,relabel_map)

        U = nx.union(G0,G1)

        edge_list = [('M0_'+str(k), 'M1_'+str(self.A.a[k])) for k in self.A.a.keys()]
        U.add_edges_from(edge_list)

        pos = nx.shell_layout(U)

        for k in pos.keys():
            if 'M1' in k:
                pos[k] = pos[k] + [10,0]
                
        R_list = np.array(['M0_'+n for n in self.A.R])
        nR = list(set(self.A.M0.nodes()) - set(self.A.R))
        nR_list = np.array(['M0_'+n for n in nR])

        nx.draw_networkx_nodes(U,pos,nodelist=R_list,node_color='b',alpha=.5)
        nx.draw_networkx_nodes(U,pos,nodelist=nR_list,node_color='b',alpha=.2)
        nx.draw_networkx_labels(U,pos)
        nx.draw_networkx_edges(U,pos,edgelist=G0.edges(),edge_color='k')

        nx.draw_networkx_nodes(U,pos,nodelist=G1.nodes(),node_color='g',alpha=.5)
        nx.draw_networkx_labels(U,pos)
        nx.draw_networkx_edges(U,pos,edgelist=G1.edges(),edge_color='k')

        nx.draw_networkx_edges(U,pos,edgelist=edge_list,edge_color='r',style='dashed')