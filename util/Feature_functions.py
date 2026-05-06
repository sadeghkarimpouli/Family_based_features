
import numpy as np

# Event rate 
def Event_rate(data, time_win):

    return data.shape[0]/time_win

# Moments rate 
def Moment_rate(data, time_win):
    Moments = 10**(1.5*(data[:,5] + 9.1))

    return np.log10(np.sum(Moments))/time_win

# Gutenberg-Richter b-value
def b_value(data, b_flag, Mc=None):
    if b_flag == '1':
        return 1, None
    
    # maximum-likelihood estimate (MLE) of b (Deemer & Votaw 1955; Aki 1965; Kagan 2002):
    elif b_flag == 'b':
        X = data[np.where(data[:,5]>Mc)[0],:]
        if X.shape[0] > 0:
            b = 1/((np.mean(X[:,5] - Mc))*np.log(10))
            std_error = b/np.sqrt(X.shape[0])
        else:
            raise ValueError("All events in the current time window have a magnitude less than 'completeness magnitude'. Use another value either for 'time window', 'minimum number of events' or 'completeness magnitude'. Also check 'time window type'.")
        return b, std_error
    
    # B-positive (van der Elst 2021)
    elif b_flag == 'bp':
        num_bootstraps = 100
        # Function to perform bootstrap estimation
        def bootstrap_estimate(data, num_bootstraps):
            estimates = []
            for _ in range(num_bootstraps):
                # Generate bootstrap sample
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                # Perform maximum likelihood estimation on bootstrap sample
                diff_mat = np.diff(bootstrap_sample)
                diff_mat = diff_mat[np.where(diff_mat>0)[0]]
                estimate = 1/((np.mean(diff_mat - np.min(diff_mat)))*np.log(10))
                estimates.append(estimate)
            return np.array(estimates)
        
        diff_mat = np.diff(data[:,5])
        diff_mat = diff_mat[np.where(diff_mat>0)[0]]
        bp = 1/((np.mean(diff_mat - np.min(diff_mat)))*np.log(10))
        # bootstrap_estimates = bootstrap_estimate(diff_mat, num_bootstraps)
        # std_error = np.std(bootstrap_estimates, axis=0)
        return bp, 0 #std_error
    
    # Tapered Gutenberg_Richter (TGR) distribution (Kagan 2002)
    elif b_flag == 'b_TGR':
        from scipy.optimize import minimize

        # The logarithm of the likelihood function for the TGR distribution (Kagan 2002)
        def log_likelihood(params, data):
            beta, Mcm = params
            n = len(data)
            Mt = np.min(data)
            l = n*beta*np.log(Mt)+1/Mcm*(n*Mt-np.sum(data))-beta*np.sum(np.log(data))+np.sum(np.log([(beta/data[i]+1/Mcm) for i in range(len(data))]))
            return -l
        X = data[np.where(data[:,5] > Mc)[0],:]
        M = 10**(1.5*X[:,5]+9.1)
        initial_guess = [0.5, np.max(M)]
        bounds = [(0.0, None), (np.max(M), None)]

        # Minimize the negative likelihood function for beta and maximum moment
        result = minimize(log_likelihood, initial_guess, args=(M,), bounds=bounds, method='L-BFGS-B',
                options={'gtol': 1e-12, 'disp': False})
        beta_opt, Mcm_opt = result.x
        eta = 1/Mcm_opt
        S = M/np.min(M)
        dldb2 = -np.sum([1/(beta_opt-eta*S[i])**2 for i in range(len(S))])
        dldbde = -np.sum([S[i]/(beta_opt-eta*S[i])**2 for i in range(len(S))])
        dlde2 = -np.sum([S[i]**2/(beta_opt-eta*S[i])**2 for i in range(len(S))])
        std_error_beta = 1/np.sqrt(dldb2*dlde2-dldbde**2)*np.sqrt(-dlde2)
        
        return beta_opt*1.5, std_error_beta*1.5

# Correlation integral
def correlation_integral(points, r_max, num_steps):
    from itertools import combinations

    num_points = len(points)
    pairs = np.array(list(combinations(range(num_points), 2)))
    pair_distances = np.sqrt((points[pairs[:,0],0] - points[pairs[:,1],0])**2 +\
                             (points[pairs[:,0],1] - points[pairs[:,1],1])**2 +\
                             (points[pairs[:,0],2] - points[pairs[:,1],2])**2)

    r_values = np.linspace(0, r_max, num_steps)
    counts = np.zeros_like(r_values)
    i = 0
    for r in r_values:
        num_pairs_in_radius = len([d for d in pair_distances if d <= r])
        counts[i] = num_pairs_in_radius
        i += 1
    
    return counts[1:]/np.shape(pairs)[0]

# Interevent time and distance
def Interevent_ts(data):
    from statistics import mode

    pair_times = np.abs(np.diff(data[:,4]))

    pair_distances = np.sqrt((np.diff(data[:,1]))**2 + (np.diff(data[:,2]))**2 + (np.diff(data[:,3]))**2)

    return np.max(pair_times), np.mean(pair_distances)

# Clustering features
# This part contains modified portions of code from eqclustering
# by Mark Williams (University of Nevada, Reno),
# licensed under Apache License 2.0.
def Clustering_analysis(data, b=1, fd=2.5, p=0.5, event_lag=250, time_lag=None, c=6.5):
    import networkx as nx
    from pandas import DataFrame

    # j is the parent of i
    idx  =data[:,0]
    x, y, z = data[:,1], data[:,2], data[:,3]
    time = (data[:,4] - data[0,4])
    mag = data[:,5]

    def dist_lab(x1, y1, z1, x2, y2, z2):
        """
        Spacial distance (in mm) between points given as [x,y,z]
        """
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1-z2)**2)/1000

    _n = len(time)

    P = np.empty(_n) * np.nan # tree
    P_fma = np.empty(_n) * np.nan # tree for for- main- and aftershock
    eta = np.empty(_n) * np.inf # eta*
    T = np.empty(_n) * np.nan # T used in eta (tij)
    TN = np.empty(_n) * np.nan # Normalized T (tij*10**(-p*b*m))
    R = np.empty(_n) * np.nan # R used in eta (rij)
    RN = np.empty(_n) * np.nan # Normalized R (rij**fd*10**(-(1-p)*b*m))
    M = np.empty(_n) * np.nan # Mag of parent event in tree

    # Function to calulate "Eta(i,j)" distance
    def eta_ij(t, r, m):
        return t * r**fd * 10**(-b*m)

    def normalmix_1D(x, mu1i, mu2i, sigma1i, sigma2i):
        """
        Estimates parameters of a 2-component 1-D Gaussian mixture model
        (uses the EM algorithm)
        
        Inputs
        ------
        x : the data sample to be analysed
        mu#i,sigma#i : initial parameters of components
        
        Outputs
        -------
        c, mu1, mu2, sigma1, sigma2
            c : the maximum likelihood boundary between the hats
            mu#, sigma# : are the estimated paraeters of components
        
        """
        A1i = 0.5
        A2i = 0.5
        epsilon = 1
        p1a = np.zeros(x.size)
        p2a = np.zeros(x.size)
        p1 = np.zeros(x.size)
        p2 = np.zeros(x.size)
        epsilon0 = 0.001

        while epsilon >  epsilon0:
            p1a = A1i * (1/(np.sqrt(2*np.pi*sigma1i**2))*np.exp(-(((x-mu1i)/sigma1i)**2)/2))
            p2a = A2i * (1/(np.sqrt(2*np.pi*sigma2i**2))*np.exp(-(((x-mu2i)/sigma2i)**2)/2))

            p1 = p1a / (p1a+p2a)
            p2 = p2a / (p1a+p2a)

            A1 = np.sum(p1)/len(x)
            A2 = np.sum(p2)/len(x)

            mu1 = np.sum(p1*x)/np.sum(p1)
            mu2 = np.sum(p2*x)/np.sum(p2)

            sigma1 = np.sqrt(np.sum(p1*(x-mu1)**2)/np.sum(p1))
            if sigma1 == 0:
                sigma1 = np.random.rand()/1
            sigma2 = np.sqrt(np.sum(p2*(x-mu2)**2)/np.sum(p2))
            if sigma2 == 0:
                sigma2 = np.random.rand()/1

            epsilon1 = np.abs(A1i-A1)
            epsilon2 = np.abs(mu1i-mu1)
            epsilon3 = np.abs(mu2i-mu2)
            epsilon4 = np.abs(sigma1i-sigma1)
            epsilon5 = np.abs(sigma2i-sigma2)
            epis = [epsilon1, epsilon2, epsilon3, epsilon4, epsilon5]
            epsilon = np.amax(epis)

            A1i = A1
            A2i = A2
            mu1i = mu1
            mu2i = mu2
            sigma1i = sigma1
            sigma2i = sigma2

        aa = -2*sigma1**2 + 2*sigma2**2
        bb = (2*mu2 * 2*sigma1**2) + (-2*mu1 * 2*sigma2**2)
        cc = (-mu2**2 * 2*sigma1**2) + (mu1**2* 2*sigma2**2) - (np.log(sigma2/sigma1) * 2*sigma1**2 * 2*sigma2**2)
        c1 = (-bb + np.sqrt(bb**2 - 4*aa*cc)) / (2*aa)
        c2 = (-bb - np.sqrt(bb**2 - 4*aa*cc)) / (2*aa)

        if mu1 < c1 < mu2:
            c = c1
        else:
            c = c2
        return c, mu1, mu2, sigma1, sigma2, A1, A2, p1, p2

    for i in np.arange(time.size):
        if time_lag:
            I = (time < time[i]) & (time >= time[i]-time_lag)
        elif event_lag:
            I = (time < time[i])
            if i > event_lag:
                I[:i-event_lag] = False
        else:
            I = (time < time[i])
        
        if np.any(I):
            r = dist_lab(x[i], y[i], z[i], x[I], y[I], z[I])
            t =  time[i] - time[I] 
            eta_c = eta_ij(t, r, mag[I])

            imin = eta_c.argmin()
            eta[i] = eta_c[imin] # minimum eta for the current event
            P_fma[i] = np.arange(len(I))[imin]
            P[i] = idx[I][imin] #np.arange(len(I))[imin] # index of the parent (j) of i
            T[i] = t[imin] # T for minmum eta
            R[i] = r[imin] # R for minimum eta
            M[i] = mag[I][imin] # Mag for minmum eta
            
            TN[i] = T[i]*10**(-p*b*M[i])
            RN[i] = R[i]**fd*10**(-(1-p)*b*M[i])

    TRp = np.log10(TN)+np.log10(RN)
    I = (np.isfinite(TRp)) & (TN > 0) & (RN > 0)
    
    while np.isnan(c):
        c, mu1, mu2, sigma1, sigma2, A1, A2, p1, p2 = normalmix_1D(TRp[I], 
                                                                   -6+np.random.rand()/10, -3+np.random.rand()/10, 
                                                                   1+np.random.rand()/100, 1+np.random.rand()/100)

    # Tree = np.vstack((np.arange(P.size), P)).T
    Tree = np.zeros((P.size,2))
    Tree[:,0] = idx
    Tree[:,1] = P
    Tree = Tree[1:,:]
    Families_index = np.where(TRp[1:,] < c)[0]
    clust_ratio = len(Families_index)/len(idx)

    g_df = DataFrame({'target' : Tree[Families_index,0], 
           'source' : Tree[Families_index,1]})
    G = nx.from_pandas_edgelist(g_df, source='source', target='target')
    All_g = list(G.subgraph(cc) for cc in nx.connected_components(G))
    
    clust_number = len(All_g)
    clust_member = []
    clust_lenght = []
    clust_size = []
    clust_radius = []
    clust_diameter = []
    # clust_eccentricity = []
    # clust_center = []
    # clust_periphery = []
    clust_density = []
    for j in range(len(All_g)):
        g_idx = np.array(list(All_g[j]))#.astype('int')
        unique_idx = [np.where(idx == g_i)[0][0] for g_i in g_idx]

        clust_member.append(len(unique_idx))
        clust_lenght.append(time[np.max(unique_idx)] - time[np.min(unique_idx)])
        clust_size.append(np.sqrt((np.min(x[unique_idx])-np.max(x[unique_idx]))**2 +\
                                  (np.min(y[unique_idx])-np.max(y[unique_idx]))**2 +\
                                  (np.min(z[unique_idx])-np.max(z[unique_idx]))**2))
        clust_radius.append(nx.radius(All_g[j]))
        clust_diameter.append(nx.diameter(All_g[j]))
        # clust_eccentricity.append(nx.eccentricity(All_g[j]))
        # clust_center.append(nx.center(All_g[j]))
        # clust_periphery.append(nx.periphery(All_g[j]))
        clust_density.append(nx.density(All_g[j]))

    def descend_all(parent, i):
        """
        Return all children from parent of index i
        """
        d = [i]
        for child in np.arange(parent.size)[parent==i]:
            d += descend_all(parent, child)
        return d

    P1 = P_fma.copy()  # maybe not needed
    K = ((np.log10(TN) + np.log10(RN)) > c)  # Mainshock condition
    P1[K] = np.nan

    # TODO: fixed in bp function, can remove...
    #P1[0] = np.nan  # first event won't pass cut, but is always a root

    Ifor = np.array([])
    Iaft = np.array([])
    Imain = np.array([])
    Pfin = np.zeros(P1.size)

    K = np.isnan(P1) # weird, K should already be this...
    Ki = np.arange(K.size)[K]

    for i in np.arange(Ki.size):
        I0 = descend_all(P1, Ki[i])
        I = np.array(I0)
        # TODO: if not mag: imax = I[0]
        imax = mag[I].argmax()
        IM = I[imax]
        IF0 = I[time[I] < time[IM]]
        IA0 = I[time[I] > time[IM]]
        #print "Subtree: event:{0}, main:{2}, cluster:{1}".format(Ki[i], I0, IM)
        
        # Keep track of fore/after shocks
        Ifor = np.append(Ifor, IF0)
        Iaft = np.append(Iaft, IA0)
        Imain = np.append(Imain, IM)
        
        # Change parent of all cluster members to main shock ID
        Pfin[IF0] = IM
        Pfin[IA0] = IM
        Pfin[IM] = IM
    
    I = (np.isfinite(np.log10(TN*RN))) & (TN > 0) & (RN > 0)
    trp = np.log10(TN[I]*RN[I])
    trr = np.log10(TN[I]/RN[I])
    Cluster_mat = np.zeros((len(TN), 3))
    Cluster_mat[Ifor.astype('int'),0] = 1
    Cluster_mat[Imain.astype('int'),1] = 1
    Cluster_mat[Iaft.astype('int'),2] = 1
    P_ForMainAft = np.sum(Cluster_mat, axis=0) / np.sum(Cluster_mat)

    return np.mean(trp), np.mean(trr), P_ForMainAft[0], P_ForMainAft[1], P_ForMainAft[2],\
           clust_number, np.mean(clust_member), np.mean(clust_lenght), np.mean(clust_size), np.mean(clust_radius),\
           np.mean(clust_diameter), np.mean(clust_density), clust_ratio*100, len(x)

# 3D convex Hull volume
def Vol_ConvexHull(data):
    from scipy.spatial import ConvexHull

    hull = ConvexHull(data[:,1:4])
    return np.log10(hull.volume)

# Strain rate 
def Strain(data, t_win, strain_model = 'Kostrov', Mu = 35*10**(9)):
    if strain_model == 'Kostrov':
        Kostrov = 10**(1.5*(data[:,5] + 9.1))
        vol = Vol_ConvexHull(data)

        return np.log10(np.sum(Kostrov)/2/Mu/vol/t_win)

# Scaled energy rate
def Scaled_energy(data, time_win):

    return np.sum(data[:,6])/time_win

class CandidateEventsTS:
    def __init__(self, data, time_win, space_win='inf', Mc=None):
        assert time_win > 0, f"Time windows is {time_win}, which should be a positive number"
        self.data = data
        self.Mc = Mc
        self.time_win = time_win
        self.space_win = space_win

    def filter_by_magnitude(self):
        indx = np.where(self.data[:, 5] >= self.Mc)[0]
        if len(indx) > 0:
            self.data = self.data[indx, :]
        else:
            self.data = []

    def filter_by_time(self):
        indx = np.where(self.data[:, 4] > (self.data[-1,4] - self.time_win))[0]
        if len(indx) > 0:
            self.data = self.data[indx, :]
        else:
            self.data = []

    def filter_by_space(self):
        dist = np.sqrt(np.sum((self.data[:, 1:4] - self.data[-1, 1:4]) ** 2, axis=1))
        indx = np.where(dist < self.space_win)[0]
        if len(indx) > 0:
            self.data = self.data[indx, :]
        else:
            self.data = []

    def filter_data(self):
        if self.Mc:
            self.filter_by_magnitude()
        if len(self.data) > 0:
            self.filter_by_time()
            if len(self.data) > 0 and self.space_win != 'inf':
                self.filter_by_space()
        return self.data
    