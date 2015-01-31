import numpy
import networkx
import matplotlib.pyplot as pyplot 
import matplotlib
import random
matplotlib.rc('text', usetex=True)


# Default parameters
_DEFAULT_PARAMS = {
    'rho': lambda: 0.1*numpy.random.randn() + 0.6,
    'gamma': 0.21,
    'alpha': lambda: 0.3,  # Constant for all users
    'beta': 0.8,
    'tau': 17.0,  # Cuttoff on time between calls
    'xmin': 1.0,
    'network_type': 'sim',
    'fname': None,  # Used to load an empirical network
    'nusers': 100,
    'nlocations': 100,
    'ncontacts': 30,
    'nsteps': 100,
}


class GeoSim():
    '''
    An implementation of the GeoSim mobility model described by Toole et al.


    The GeoSim model is a model that reproduces patterns of mobility within cities
    taking into account individual and social preferences.  The model takes 
    a social network, either generated or empirical, as input and simulates
    the mobility of agents. Designed to model movements within cities, agents
    may visit a number of discrete locations where the distance between
    locations is assumed to have little affect on choices of where to go.
    Instead, choices are based on preferential return to previously visited
    locations as well as the locations of social contacts.  This model is 
    an extension of Individual Mobility Model described by Song et al.
    doi:10.1038/nphys1760

    The model works as follows:
        - A number of users are distributed randomly across N locations and 
        connected as part of a social network.
        - Agents are assigned a waiting time at a given location after which
        they will make a jump to another location.
        - Agents choose their next destination based on either individual
        preferences or social behavior.
        - With probability rho*S^(-gamma) an agent explores and visits a
        location they have not previously visited, where rho and gamma are
        parameters and S is the number of unique locations they have visited
        thus far.
        - With probability (1-rho*S^(-gamma)) an agent returns to a location
        they have visited previously.
        - Given exploration, with probability alpha an agent choses a new
        location to visit based on preferential visit to a contacts location.
        The probability a contact is chosen is proportional to the cosine
        similarity of the visit frequences between the agent and each contact.
        - With probability (1-alpha) an agent chooses a new location to visit
        uniformly at random.
        - Given the return to a previously visited location, with probability
        alpha, the agent again chooses a contact proportional to the cosine
        similarities.  Given this choice, they will preferentially visit
        a location visited by this contact and that the agent has already
        visited.
        - With probability (1-alpha), the agent preferentially returns to a
        location based only on their own past visit frequencies.

    The cosine similiarty is measured between the visitation frequencies of
    two agents.

    Choices made "preferentially" refer to cases where the choice of a location
    to visit is directly proportial to the frequency that location has been
    visited in the past.  For example, if an agent preferentially chooses a
    location based on the visits of their social contact, the probility of 
    choosing location k is proportional to the fraction of all visits that
    contact is observed at k.  

    When alpha=0, we recover the Individual mobility model of Song et al.

    A number of parameters can be used for the GeoSim model. In general, these
    parameters are global to each agent in the simulation, but some must be
    provided as distributions.  Specifically, rho and alpha may be different
    for different agents, representing heterogeneity in the social preferences
    of different individuals.  These parameters must be input as generating
    functions that return a value for each agent when the model is initialized.

    For example, the assign each agent the same value of alpha, a lambda
    function can be passed: lambda: 0.3

    To assign a parameter randomly with a normal distribution, a lamda function
    of the following form can be used: lambda: numpy.random.randn()
    '''
    

    def __init__(self, params=None):
        '''Creates a new GeoSim model object.

        Args:
            params - dict, a dictionary of parameter values. If this is not
            provided, defaults are set.
        '''
        self.graph = None
        self.params = _DEFAULT_PARAMS;

        if params:
            for param, value in params.iteritems():
                self.params[param] = value

        self.initialize_sim()


    def initialize_sim(self, type='lognorm'):
        '''Initializes the simulation.

        Three types of network can be specified, a real network can be provided
        with a filename to a pickled networkx object, a small world network
        created using the small world generated in networkx, or a random graph
        with a lognormal degree distribution.
        '''

        # Intialize the network.
        if self.params['network_type'] == 'real':
            self.create_network_from_file(self.params['fname'])
        elif self.params['network_type'] == 'smallworld':
            self.create_smallworld_network()
        else:
            self.create_lognormal_network()

        # Intialize the attributes of nodes and edges.
        self.set_initial_attributes()
        # Initialize the locations of users.
        self.set_initial_locations()
        # Compute initial similarity.
        self.calculate_similarity()
        # The mode tracks the average growth of the number of unique locations,
        # S, over the length of the similation.  We compute and store these
        # values as the time goes on.  We measure S(t) at various points along
        # the simulation.
        max_time_pow = numpy.ceil(numpy.log10(self.params['nsteps']))
        self.time = numpy.logspace(1, max_time_pow, 20)
        self.St = numpy.zeros(len(self.time))


    def set_initial_attributes(self):
        '''Initialize the attributes of nodes in the social network.'''

        for i in self.graph.nodes_iter():
            # A location vector for a user tracks the number of visits made
            # by a user to each location.  It is initially all zeros as the 
            # user has yet to travel anywhere.
            self.graph.node[i]['lvec'] = numpy.zeros(self.params['nlocations'],
                                                     dtype=int)
            self.graph.node[i]['S'] = 0
            # Set alpha and rho
            self.graph.node[i]['alpha'] = self.params['alpha']()
            self.graph.node[i]['rho'] = self.params['rho']()

    
    def set_initial_locations(self):
        '''Sets the initial location for each user.'''

        # Each user is assigned to a location uniformly at random.
        #a = numpy.arange(self.params['nlocations'], dtype=float)
        #p = a / numpy.sum(a)
        #p = numpy.cumsum(p)
        for i in self.graph.nodes_iter():
            #r = numpy.random.rand(self.graph.node[i]['S'])
            #l = numpy.digitize(r, p)
            l = numpy.random.randint(self.params['nlocations'])
            self.graph.node[i]['lvec'][l] = 1
            while self.graph.node[i]['S'] < 1:
                l = numpy.random.randint(self.params['nlocations'])
                if self.graph.node[i]['lvec'][l] == 0:
                    self.graph.node[i]['S'] += 1
                self.graph.node[i]['lvec'][l] += 1


    def create_network_from_file(self, filename):
        '''Loads a network from a pickled networkx file.'''
        self.graph = networkx.read_gpickle(filename)
        self.params['nusers'] = self.graph.number_of_nodes()


    def create_smallworld_network(self, prob=0.9):
        '''Creates a small world network.'''
        self.graph = networkx.newman_watts_strogatz_graph(
            self.params['nusers'], self.params['ncontacts'], prob)


    def create_lognormal_network(self):
        '''Creates a network with a lognormal degree distribution.'''
        self.graph = networkx.Graph()
        nodes = numpy.arange(self.params['nusers'])
        degs = numpy.random.lognormal(numpy.log(self.params['ncontacts']),
                                      0.3, self.params['nusers'])
        for i in nodes:
            self.graph.add_node(i)

        # connect the network
        for i in nodes:
            stubs_left = degs[i] - self.graph.degree(i)
            if stubs_left > 0:
                nbrs = []
                while len(nbrs) < stubs_left:
                    tries = 0
                    j = nodes[numpy.random.randint(self.params['nusers'])]
                    while ((degs[j] - self.graph.degree(j) <= 0 or
                            i == j) and (tries < 1000)):
                        j = nodes[numpy.random.randint(self.params['nusers'])]
                        tries += 1
                    nbrs.append(j)
                edges = [(i, j, {'sim': None}) for j in nbrs]
                self.graph.add_edges_from(edges)
            if i % (self.params['nusers'] / 10) == 0:
                print i, self.params['nusers']

    def run(self):
        '''Runs the GeoSim Model.'''
        print 'Running Model...'

        t = 10 # time starts at 10 hours
        tidx = numpy.digitize([t], self.time)[0]
        # Vector to track the time that the users will make their next move.
        nexttime = numpy.zeros(self.params['nusers']) + t
        # For each hour of the simulation, do...
        while t < self.params['nsteps']:
            # Report evey 30 days
            if t%(30*24) == 0:
                print 'Day %d' % int(t/24)

            # Every ten days of similuation, recompute the similarity values.
            if t%(24*10) == 0:
                self.calculate_similarity()
            # At certain points, compute the S(t) and store for analysis.
            if t > self.time[tidx]:
                self.calculate_st(tidx)
                tidx+=1
            # For each user...
            for u in xrange(self.params['nusers']):
                # If it is not time for the user to move, do not do anything.
                if t < nexttime[u]:
                    continue

                # If the user is moving, make choices based on the dynamics
                # outline in the GeoSim paper.
                r = numpy.random.rand()
                tries = 0
                l = None
                explore_probability = (self.graph.node[u]['rho'] *
                                       self.graph.node[u]['S'] **
                                       (-self.params['gamma']))
                # Explore...
                if r < explore_probability:
                    r = numpy.random.rand()
                    # Make an individual choice...
                    if r > self.graph.node[u]['alpha']:
                        # Choose a random location until one is found that
                        # has not been visited before.
                        while True:
                            l = self.get_random_location()
                            tries += 1
                            if (self.graph.node[u]['lvec'][l] == 0 or
                               tries > 100):
                                break
                    # Make a social choice...
                    else:         
                        # Choose a location based on social contacts until one
                        # is found that has not been visited before.       
                        while True:
                            l = self.get_social_location(u)
                            tries += 1
                            if (self.graph.node[u]['lvec'][l] == 0 or
                               tries > 100):
                                break
                    self.graph.node[u]['S'] += 1
                # Return.
                else:
                    r = numpy.random.rand()
                    # Make an individual choice...
                    if r > self.graph.node[u]['alpha']:
                        l = self.get_return_location(u)
                    # Make a social choice.
                    else:
                        # Choose locations based on the visits of social
                        # contacts until one is found that has been visited
                        # by the agent before.
                        while True:
                            l = self.get_social_location(u)
                            tries += 1
                            if (self.graph.node[u]['lvec'][l] != 0 or
                               tries > 100):
                                break
                            
                    
                # If a location has not been chosen yet, assign a random one.
                # This should not happen very often.
                if not l:
                    l = self.get_random_location()
                    if self.graph.node[u]['lvec'][l] == 0:
                        self.graph.node[u]['S'] += 1
                #nextlocs[u] = l
                self.graph.node[u]['lvec'][l] += 1
                nexttime[u] += self.get_next_time() 
            t += 1
        # After the simulation has finished, compute the final cosine similarity
        # and predictability of users.
        self.calculate_similarity()
        self.calculate_predictability()


    def get_return_location(self, node_id):
        ''' 
        Choose a location by preferential return. 
        
        The probability an individual returns to a location is proportional
        to the frequency they visit those locations.

        Args:
            node_id - the id of the node making the choice.
        Returns:
            location - the index of a location to visit.
        '''
        lvec = self.graph.node[node_id]['lvec']
        p = numpy.cumsum(lvec)/float(numpy.sum(lvec))
        r = numpy.random.rand()
        return numpy.digitize([r], p)[0]


    def get_social_location(self, node_id):
        '''
        Choose a social contact and then choose one of their locations to visit

        A contact is chosen with probability proportional to the cosine
        similarity between the visitation patterns of the two nodes.

        Args:
            node_id - the id of the node making the choice.
        Returns:
            location - the index of a location to visit.
        ''' 
        p = numpy.array([v['sim'] for v in self.graph[node_id].values()])
        p = numpy.cumsum(p)/float(numpy.sum(p))
        r = numpy.random.rand()
        f = numpy.digitize([r], p)[0]
        f = self.graph.neighbors(node_id)[f]
        return self.get_return_location(f)


    def get_random_location(self):
        '''
        Returns a location uniformly at random.

        Returns:
            location - the index of a location to visit.
        '''
        return numpy.random.randint(self.params['nlocations'])


    def get_next_time(self):
        '''
        Generates a random waiting time for a user to remain at a location.

        The distribution is based off of the empirical measurements of Song
        et al. and follows an power law distribution with an exponential cutoff
        at tau hours.

        Return:
            waiting_time - The number of hours a user will remain at a location.
        '''
        return randomht(1, self.params['xmin'], 1+self.params['beta'],
                        1./self.params['tau'])[0]


    def calculate_similarity(self):
        '''
        Calculates the mobility similarity between every two connected nodes.

        The mobility similarity is definited as the cosine similarity of the
        location vectors (visit frequencies) of two users.

        The values are assigned to networkx graph edge attributes.
        '''
        print 'Calculating similarity.'
        for i in self.graph.nodes_iter():
            l1 = self.graph.node[i]['lvec']
            for j in self.graph.neighbors(i):
                l2 = self.graph.node[j]['lvec']
                self.graph.edge[i][j]['sim'] = cosine_similarity(l1, l2)


    def calculate_predictability(self):
        '''
        Calculates the predictability of each agent based on their contacts.
         
        The predictability is computed by calculating the ratio between the
        magnitude of their location vector and a predicted location vector
        constructed as a linear combination of the location vectors of their
        contacts.

        '''
        print 'Calculating predictability.'
        for u in self.graph.nodes_iter():
            uvec = self.graph.node[u]['lvec'].astype(float)
            neighbors = self.graph.neighbors(u)
            fvecs = numpy.array([self.graph.node[f]['lvec'] for f in neighbors])
            fvecs = fvecs.astype(float)
            q, r = numpy.linalg.qr(fvecs.T)
            p = []
            for f in xrange(numpy.shape(q)[1]):
                p.append(numpy.dot(q[:, f], uvec)*q[:, f])
            p = numpy.sum(numpy.array(p), 0)
            p_norm = numpy.linalg.norm(p) / numpy.linalg.norm(uvec)
            self.graph.node[u]['pred'] = p_norm


    def calculate_st(self, tidx):
        '''
        Compute S(t) at a given time.

        Args:
            tidx - the index of the time ranges to compute S(t)
        '''
        print 'Calculating S(t).'
        s = numpy.sum(networkx.get_node_attributes(self.graph, 'S').values())
        s = s / float(self.graph.number_of_nodes())
        self.St[tidx] = s


def randomht(n, xmin, alpha, tau):
    '''Returns an list of numbers from a power law distribution with an
    exponential cutoff.

    Adapted from: http://tuvalu.santafe.edu/~aaronc/powerlaws/

    Args:
        n - the number of values to return
        xmin - the minimum value to return
        alpha - the exponent of the power law part of the distribution
        tau - the exponent of the cutoff
    Returns:
        x - a list of random numbers generated from the distribution
    '''
    x = []
    y=[]
    for i in range(10*n):
        y.append(xmin - (1./tau)*numpy.log(1.-numpy.random.rand()))
    while True:
        ytemp=[]
        for i in range(10*n):
            if numpy.random.rand()<pow(y[i]/float(xmin),-alpha):
                ytemp.append(y[i])
        y = ytemp
        x = x+y
        q = len(x)-n
        if q==0: 
            break;
        if (q>0):
            r = range(len(x))
            random.shuffle(r)
            xtemp = []
            for j in range(len(x)):
                if j not in r[0:q]:
                    xtemp.append(x[j])
            x=xtemp
            break;
        if (q<0):
            y=[]
            for j in range(10*n):
                y.append(xmin - (1./tau)*numpy.log(1.-numpy.random.rand()))
    return x


def cosine_similarity(u, v):
    '''Computes the cosine similarity between two vectors

    Args:
        u - a numpy vector
        v - a numpy vector
    Returns:
        cosine_similarity - float, the cosine similarity between u and v
    '''
    u = u.astype(float)
    v = v.astype(float)
    return numpy.dot(u, v)/(numpy.linalg.norm(u)*numpy.linalg.norm(v))


def plot_similarity(graph, legend=False, label='', color='k', ax=None,
                    axis='semilogy', xlim=[0,1], ylim=None, marker='o'):
    '''Plots the distribution of mobility similarity values for a graph.


    The user can control the style of the plot with optional arguments and can
    add this distribution to a previously created axis by passing the handle
    as a keyword argument.

    Args:
        graph - a networkx graph object where edges have a sim attributes

    Returns:
        ax - an axis object with the distribution added.
    '''
    # Create an axes if one is not provided.
    if not ax:
        figure = pyplot.figure()
        ax = figure.add_subplot(1, 1, 1)

    # Compute the distribution
    data = numpy.array(networkx.get_edge_attributes(graph, 'sim').values())
    xbins = numpy.linspace(0, 1, num=30)
    x = xbins[1:] - (xbins[1:]-xbins[:-1])/2.
    y, edges = numpy.histogram(data, xbins, density=True)

    # Plot the distribution on the axis.
    if axis == 'semilogy':
        ax.semilogy(x, y, '-', color=color, mew=0, label=label, marker=marker)
    elif axis == 'loglog':
        ax.loglog(x, y, '-', color=color, mew=0, label=label, marker=marker)
    else:
        ax.plot(x, y, '-', color=color, mew=0, label=label, marker=marker)

    if legend:
        ax.legend(frameon=False, loc='best')
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_ylabel(r'$p(cos\theta)$')
    ax.set_xlabel(r'$cos\theta$')
    return ax


def plot_predictability(graph, legend=False, label='', color='k', ax=None,
                    axis='semilogy', xlim=[0, 1], marker='o'):
    '''Plots the distribution of predicatability values for nodes in a graph.


    The user can control the style of the plot with optional arguments and can
    add this distribution to a previously created axis by passing the handle
    as a keyword argument.

    Args:
        graph - a networkx graph object where nodes have a pred attributes

    Returns:
        ax - an axis object with the distribution added.
    '''
    # Create an axes if one is not provided.
    if not ax:
        figure = pyplot.figure()
        ax = figure.add_subplot(1, 1, 1)

    # Compute the distribution.
    data = numpy.array(networkx.get_node_attributes(graph, 'pred').values())
    xbins = numpy.linspace(0, 1, num=30)
    x = xbins[1:] - (xbins[1:]-xbins[:-1])/2.
    y, edges = numpy.histogram(data, xbins, density=True)

    # Plot the distribition on the axis.
    if axis == 'semilogy':
        ax.semilogy(x, y, '-', color=color, mew=0, label=label, marker=marker)
    elif axis == 'loglog':
        ax.loglog(x, y, '-', color=color, mew=0, label=label, marker=marker)
    else:
        ax.plot(x, y, '-', color=color, mew=0, label=label, marker=marker)

    if legend:
        ax.legend(frameon=False, loc='best')
    ax.set_xlim(xlim)
    ax.set_xlabel(r"$\frac{|\hat{\mathbf{v}}|}{|\mathbf{v}|}$")
    ax.set_ylabel(r"$p(\frac{|\hat{\mathbf{v}}|}{|\mathbf{v}|})$")

    return ax


def plot_fk(graph, legend=False, label='', color='k', ax=None,
            axis='loglog', xlim=None, ylim=None, marker='o'):
    '''Plots the location visit frequencies, f_k, of nodes.


    The user can control the style of the plot with optional arguments and can
    add this distribution to a previously created axis by passing the handle
    as a keyword argument.

    Args:
        graph - a networkx graph object where nodes have a lvec attributes

    Returns:
        ax - an axis object with the distribution added.
    '''
    # Create an axes if one is not provided.
    if not ax:
        figure = pyplot.figure()
        ax = figure.add_subplot(1, 1, 1)

    # Compute the visit frequencies.
    data = numpy.array(networkx.get_node_attributes(graph, 'lvec').values())
    data = data.astype(float) / numpy.sum(data, 1)[:,None]
    data.sort(axis=1)
    data = numpy.ma.masked_equal(data, 0)
    freq = numpy.ma.mean(data,0)

    # Plot
    if axis == 'loglog':
        ax.loglog(numpy.arange(1, len(freq)+1), 
                  freq[::-1], '-', color=color, mew=0, label=label,
                  marker=marker)
    else:
        ax.plot(numpy.arange(1, len(freq)+1), 
                  freq[::-1], '-', color=color, mew=0, label=label,
                  marker=marker)

    if legend:
        ax.legend(frameon=False, loc='best')
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$f_k$')
    return ax

