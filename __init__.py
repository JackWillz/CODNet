import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import operator
import statistics
import seaborn as sns
from sklearn.decomposition import PCA
import multiprocessing
from scipy import stats
from collections import Counter
import time as t

# For notebook presentation
from IPython.display import display


def min_samples(g, k):
    return int((len(g.nodes) / k) * 25)


def find_neighbours(g, node):
    neighbours = []
    for edge in list(g.out_edges(node)):
        neighbours.append(edge[1])
    for edge in list(g.in_edges(node)):
        neighbours.append(edge[0])
    return neighbours


def find_all_neighbours(g, nodes):
    all_neighbours = []
    for node in nodes:
        all_neighbours.extend(find_neighbours(g, node))
    all_neighbours = [node for node in all_neighbours if node not in nodes]
    return all_neighbours


def choose_next_node(g, nodes):
    all_neighbours = find_all_neighbours(g, nodes)
    if len(all_neighbours) > 0:
        next_node = np.random.RandomState().choice(all_neighbours)
    else:
        return False
    return next_node


# Collection of scoring functions

# Simple edge count
def sf_edge_count(g, sg):
    edge_count = len(sg.edges)
    return edge_count


# Total edges in the subgraph (internal nodes only) divided by all possible edges in the subgraph
def sf_internal_density(g, sg):
    int_density = nx.density(sg)
    return int_density


# Average degree of all the nodes, including their edges with nodes outside the subgraph
def sf_average_degree(g, sg):
    degrees = 0
    nodes = list(sg.nodes)
    g_degrees = g.degree()
    for node in nodes:
        degrees += g_degrees[node]
    avg_degrees = degrees / len(nodes)
    return avg_degrees

# Average internal degree of all the nodes, including their edges with nodes outside the subgraph
def sf_interal_average_degree(g, sg):
    degrees = 0
    nodes = list(sg.nodes)
    sg_degrees = sg.degree()
    for node in nodes:
        degrees += sg_degrees[node]
    avg_degrees = degrees / len(nodes)
    return avg_degrees


# Degree variance
def sf_degree_variance(g, sg):
    degrees = []
    nodes = list(sg.nodes)
    deg_dict = g.degree()
    for node in nodes:
        degrees.append(deg_dict[node])
    var = np.var(degrees)
    var = np.sqrt(var)
    return var

# Degree variance
def sf_internal_degree_variance(g, sg):
    degrees = []
    nodes = list(sg.nodes)
    deg_dict = sg.degree()
    for node in nodes:
        degrees.append(deg_dict[node])
    var = np.var(degrees)
    var = np.sqrt(var)
    return var


def sf_clustering_coefficient(g, sg):
    clust_coef = nx.average_clustering(sg)
    return clust_coef


def sf_internal_clustering_coefficient(g, sg):
    nodes = list(sg.nodes)
    clust_coef = nx.average_clustering(g, nodes)
    return clust_coef


def scaled_perc_of_max(in_edge, out_edge):
    max_edge = max(in_edge, out_edge)
    if (in_edge + out_edge) > 0:
        perc_edges = max_edge / (in_edge + out_edge)
        scaled_perc_edges = (perc_edges - 0.5) * 2
    else:
        scaled_perc_edges = 0
    return scaled_perc_edges


# The average inequality for inward/outward edges.
# 0 is all nodes have equal in/out edges and 1 is all nodes only have either inward or outward edges
def sf_internal_directional_imbalance(g, sg):
    nodes = list(sg.nodes)
    all_perc_edges = 0
    for node in nodes:
        in_edge = len(sg.in_edges(node))
        out_edge = len(sg.out_edges(node))
        scaled_perc_edges = scaled_perc_of_max(in_edge, out_edge)
        all_perc_edges += scaled_perc_edges
    avg_perc_edges = all_perc_edges / len(nodes)
    return avg_perc_edges


def sf_external_directional_imbalance(g, sg):
    nodes = list(sg.nodes)
    all_ext_perc_edges = 0
    for node in nodes:
        in_edge = g.in_edges(node)
        out_edge = g.out_edges(node)
        ext_in_edge = len([x[0] for x in in_edge if x[0] not in nodes])
        ext_out_edge = len([x[1] for x in out_edge if x[1] not in nodes])
        scaled_perc_edges = scaled_perc_of_max(ext_in_edge, ext_out_edge)
        all_ext_perc_edges += scaled_perc_edges
    avg_ext_perc_edges = all_ext_perc_edges / len(nodes)
    return avg_ext_perc_edges


# Internal to External Edge Ratio
def sf_ite_edge_ratio(g, sg):
    int_edges = 0
    ext_edges = 0
    nodes = list(sg.nodes)
    g_degree = g.degree()
    sg_degree = sg.degree()
    for node in nodes:
        ext_edges += g_degree[node]
        int_edges += sg_degree[node]
    ite_edge_ratio = int_edges / ext_edges
    return ite_edge_ratio


# Flow hierarchy is the % of the edges that are not within a cycle
# The complement of this is therefore the % of edges in a cycle
def sf_complement_flow_hierarchy(g, sg):
    score = 1 - nx.flow_hierarchy(sg)
    return score


# Recipetaory refers to the % of edges that are double directional
def sf_internal_recipetaory(g, sg):
    score = nx.reciprocity(sg)
    return score


# Recipetaory refers to the % of edges that are double directional
def sf_recipetaory(g, sg):
    nodes = list(sg.nodes)
    score_dict = nx.reciprocity(g, nodes)
    list_score = list(score_dict.values())
    score = np.mean(list_score)
    return score


def standardise_from_df(sg_score, func_name, df):
    g_row = df[df.index == func_name]
    g_score = g_row['Mean'].values[0]
    g_sd = g_row['S.D.'].values[0]
    if g_sd == 0:
        g_sd = 0.000001
    std_score = abs(sg_score - g_score) / g_sd
    return std_score


def rescore_sg(g, sg, df, scoring_functions, scoring_function_names):
    scores = []
    for i in range(len(scoring_function_names)):
        sg_score = scoring_functions[i](g, sg)
        score = standardise_from_df(sg_score, scoring_function_names[i], df)
        scores.append(score)
    return scores


# Average Deviation score
def sf_wad(g, sg, df, scoring_functions, scoring_function_names):
    scores = rescore_sg(g, sg, df, scoring_functions, scoring_function_names)
    avg_score = np.mean(scores)
    return avg_score


def create_feature_array(g, sg, df, scoring_functions, scoring_function_names):
    row = rescore_sg(g, sg, df, scoring_functions, scoring_function_names)
    X = np.array([row])
    return X


def sf_pca(pca, pca_component, g, sg, df, scoring_functions, scoring_function_names):
    X = create_feature_array(g, sg, df, scoring_functions, scoring_function_names)
    new_comp = pca.transform(X)
    coords = []
    means = []
    if pca_component == '':
        for i in range(len(new_comp[0])):
            x = new_comp[0][i]
            coords.append(x)
            means.append(0)
    else:
        x = new_comp[0][pca_component - 1]
        coords.append(x)
        means.append(0)
    dist = euc_dist(coords, means)
    return dist


def create_scoring_dict(scoring_functions):
    scores = {}
    scores["graph"] = []
    for scoring_function in scoring_functions:
        scores[scoring_function] = []
    return scores


def fill_scoring_dict(g, sg, scores, scoring_functions, scoring_function_names):
    scores['graph'].append(sg)
    for i in range(len(scoring_functions)):
        scores[scoring_function_names[i]].append(scoring_functions[i](g, sg))
    return scores

def fill_scoring_dict2(g, nodes, scoring_functions):
    scores = []
    scores.append(nodes)
    sg = g.subgraph(nodes)
    for i in range(len(scoring_functions)):
        scores.append(scoring_functions[i](g, sg))
    return scores

# Will trigger a warning if more than 10% of the subgraphs have been rejected due to size error (i.e. can't reach desired size)
# This is caused by disjointed graphs with multiple disconnected subgraphs smaller than the desired size 'k'
# Suggested fix is to start by looking at smaller subgraph sizes first, then once comforta
def small_sg_warning(small_sg_count, n):
    if small_sg_count > 0.1 * n:
        perc_small = small_sg_count / (n + small_sg_count)
        fmt_perc_small = str(round(perc_small * 100, 2)) + "%"
        print("Warning:", fmt_perc_small, "of subgraphs have been removed as they cannot reach desired size.")


def create_subgraph(g, nodes, k):
    initial_node = random.choice(nodes)
    nodes = [initial_node]
    for i in range(k - 1):
        next_node = choose_next_node(g, nodes)
        if next_node != False:
            nodes.extend([next_node])
        else:
            return False
    return nodes


def create_mp_args(batch_size, g, nodes, k):
    i = 0
    args = []
    while i < batch_size:
        args.append([g, nodes, k])
        i += 1
    return args

def create_mp_args2(g, all_nodes, scoring_functions):
    errors = 0
    args = []
    for nodes in all_nodes:
        if nodes != False:
            args.append([g, nodes, scoring_functions])
        else:
            errors += 1
    return args, errors


def ttest_samples(batches):
    for i in range(4):
        for j in range(4):
            if i != j:
                p_val = stats.ttest_rel(batches[i], batches[j])[1]
                if p_val < 0.10:
                    return False
                else:
                    continue
    return True


# Will perform t-test on batches from the sampled graphs.
# Will print warning to the user that there is a high degree of variance between two batches
# This would be an indicator of insufficient sampling
def ensure_stability(scores):
    stats = ['Internal Edges', 'Internal Directional Imbalance', 'Internal Clustering Coefficient',
             'Internal Percentage of Reciprocal Edges']
    number_of_samples = len(scores['Internal Edges'])
    batch_size = int(number_of_samples / 4)
    for stat in stats:
        batches = []
        for i in range(4):
            batch = scores[stat][i * batch_size:  (i + 1) * batch_size]
            batches.append(batch)
        if ttest_samples(batches) is False:
            print("WARNING: T-Test between batches has p-value <0.10, suggesting high variance in batch samples, "
                  "potentially a sign of insufficient sampling size.")
            return False
        else:
            continue
    return True



def ensure_node_coverage(all_nodes, nodes):
    remove_bools = [x for x in all_nodes if x != False]
    flat_list = [item for sublist in remove_bools for item in sublist]
    flat_list.extend(nodes)
    node_counts = Counter(flat_list)
    for key, values in node_counts.items():
        node_counts[key] = values - 1
    avg_coverage = np.mean(list(node_counts.values()))
    if avg_coverage < 5:
        print("WARNING: The average node only appears in", round(avg_coverage, 2), "samples. This is usually due to low sampling.")
        print("It is recommended that it is increased to be at least above 5 (5 * (graph size / desired subgraph size)).")


def sample_graphs(g, k, n, scoring_functions, scoring_function_names):
    scores = create_scoring_dict(scoring_function_names)
    nodes = list(g.nodes)
    args = create_mp_args(n, g, nodes, k)
    start_sample = t.time()
    processors = multiprocessing.cpu_count()
    print("Running on:", processors, "processors")
    with multiprocessing.Pool(processors) as pool:
        all_nodes = pool.starmap(create_subgraph, args)
    end_sample = t.time()
    print("Sampling time:", (end_sample - start_sample))

    start_scoring = t.time()
    args, small_sg_count = create_mp_args2(g, all_nodes, scoring_functions)
    with multiprocessing.Pool(processors) as pool:
        all_scores = pool.starmap(fill_scoring_dict2, args)
    end_scoring = t.time()
    print("Scoring time:", (end_scoring - start_scoring))

    start_building = t.time()
    for graph in all_scores:
        scores['graph'].append(g.subgraph(graph[0]))
        for i in range(len(scoring_function_names)):
            scores[scoring_function_names[i]].append(graph[i+1])
    end_building = t.time()
    print("Sample building time:", (end_building - start_building))


    start_testing = t.time()
    ensure_node_coverage(all_nodes, nodes)
    small_sg_warning(small_sg_count, len(scores['graph']))
    ensure_stability(scores)
    end_testing = t.time()
    print("Testing time:", (end_testing - start_testing))
    return scores


def standardise(scores):
    mean = np.mean(scores)
    sd = statistics.stdev(scores)
    # Sometimes there will be a subgraph that contains all 1s or all 0s
    # I.e. building a small subgraph where no edge is reciprocal is quite common (perc of rec edges = 0)
    # We replace the sd with a small value, meaning the values will be all 0 once standardised
    if sd == 0:
        sd = 0.000001
    std_scores = []
    for score in scores:
        std_score = abs(score - mean) / sd
        std_scores.append(std_score)
    return std_scores


def mean_sd(all_scores, scoring_function_names):
    avg_stds = []
    no_graphs = len(all_scores['graph'])
    for i in range(no_graphs):
        total_dev = 0
        for function in scoring_function_names:
            total_dev += all_scores[function + ' standardised'][i]
        avg_dev = total_dev / len(scoring_function_names)
        avg_stds.append(avg_dev)
    return avg_stds


def standardise_all(all_scores, scoring_function_names):
    for function in scoring_function_names:
        scores = all_scores[function]
        devs = standardise(scores)
        all_scores[function + ' standardised'] = devs
    all_scores['Average Deviation'] = mean_sd(all_scores, scoring_function_names)
    return all_scores


def sort_scores(all_scores):
    zipped_scores = zip(all_scores['PCA Distance'], all_scores['graph'])
    sorted_zipped_scores = sorted(zipped_scores, reverse=True, key=lambda x: x[0])
    scores, graphs = zip(*sorted_zipped_scores)
    return graphs


def sort_summaries(summaries):
    summaries = sorted(summaries, key=lambda x: x[2], reverse=True)
    return summaries


def print_summary(graph_num, stat_df, graph_index, all_scores, pca_dist, scoring_function_names):
    print("                   Graph", str(graph_num + 1), "Summary")
    print("----------------------------------------------------------")
    print("Euclidean distance from means in PCA:", str(round(pca_dist, 3)))
    summaries = []
    for func in scoring_function_names:
        graph_stat = round(all_scores[func][graph_index], 3)
        mean = round(stat_df[stat_df.index == func]['Mean'].values[0], 3)
        if mean != 0 or mean != 1:
            diff = abs(1 - (graph_stat / mean))
            if (graph_stat / mean) > 1:
                compare = "greater than"
            else:
                compare = "less than"
        else:
            diff = 0
            compare = "the same as"
        summaries.append([graph_stat, func, diff, compare, mean])
    sorted_summaries = sort_summaries(summaries)
    for summary in sorted_summaries:
        diff_format = str(round(summary[2] * 100, 1)) + "%"
        print("The Graph has", str(summary[0]), summary[1], "this is",
              diff_format, summary[3], "the average of", summary[4])


# Calculate key statistics from all sub graphs
def average_statistics(score):
    mean = np.mean(score)
    median = np.median(score)
    sd = statistics.stdev(score)
    min_val = min(score)
    max_val = max(score)
    perc_5 = np.percentile(score, 5)
    perc_25 = np.percentile(score, 25)
    perc_50 = np.percentile(score, 50)
    perc_75 = np.percentile(score, 75)
    perc_95 = np.percentile(score, 95)
    return [mean, median, sd, min_val, perc_5, perc_25, perc_50, perc_75, perc_95, max_val]


def create_stats_df(all_stats):
    df = pd.DataFrame(all_stats,
                      columns=['Score', 'Mean', 'Median', 'S.D.', 'Min. Value', '5% Perc.', '25% Perc.', '50% Perc.',
                               '75% Perc.', '95% Perc.', 'Max. Value'])
    df.set_index('Score', inplace=True)
    return df


def summary_statistics(scores, scoring_function_names):
    all_stats = []
    extra_sfn = ['PCA Distance'] + scoring_function_names
    for func in extra_sfn:
        stats = average_statistics(scores[func])
        stats.insert(0, func)
        all_stats.append(stats)
    stat_df = create_stats_df(all_stats)
    return stat_df


def draw_distribution(scores, scoring_function_name):
    plot = sns.distplot(scores[scoring_function_name], rug=True)
    plot.set(xlabel=scoring_function_name, ylabel='Frequency', title='Distribution of ' + scoring_function_name)
    plt.show()


def create_feature_df(scores, scoring_function_names):
    df = pd.DataFrame()
    for func in scoring_function_names:
        df[func + ' standardised'] = scores[func + ' standardised']
    return df


def euc_dist(a, b):
    total_sqrd_dist = 0
    for i in range(len(a)):
        diff = a[i] - b[i]
        sqrd_diff = diff ** 2
        total_sqrd_dist += sqrd_diff
    dist = np.sqrt(total_sqrd_dist)
    return dist


def fit_pca(scores, scoring_function_names):
    df = create_feature_df(scores, scoring_function_names)
    pca = PCA(n_components=0.9)
    comps = pca.fit_transform(df)
    trans_comps = comps.transpose()
    pca_dists = []
    for i in range(len(comps)):
        coords = []
        means = []
        for j in range(len(comps[i])):
            x = comps[i][j]
            coords.append(x)
            means.append(0)
        dist = euc_dist(coords, means)
        pca_dists.append(dist)
    for i in range(len(trans_comps)):
        scores['PCA_' + str(i + 1)] = trans_comps[i]
    scores['PCA Distance'] = pca_dists
    return scores, pca


def step_change(g, sg):
    nodes = list(sg.nodes)
    no_nodes = len(nodes)
    counter = 0
    success = False
    while success == False:
        # Given the weakly connected requirement, sometimes there are no new possible graphs
        # This will result in code getting stuck in a loop constantly failing to meet the wc req.
        # To counter this, we break if no graph has been found in 5 x nodes
        if counter > no_nodes * 5:
            new_sg = sg
            success = True
            change = False
        popped = nodes.pop(np.random.RandomState().randint(len(nodes)))
        new_sg = g.subgraph(nodes)
        if nx.is_weakly_connected(new_sg):
            new_neighbours = find_all_neighbours(g, nodes)
            next_node = np.random.RandomState().choice(new_neighbours)
            nodes.extend([next_node])
            new_sg = g.subgraph(nodes)
            success = True
            change = True
        else:
            nodes.insert(0, popped)
            counter += 1
            continue
    return new_sg, change


def evaluate_step(best_score, proposed_score, op):
    if op(proposed_score, best_score):
        return True
    else:
        return False


def stepwise_optimisation(g, sg, pca, pca_component, df=[], scoring_functions=[], scoring_function_names=[],
                          max_iter=10000, max_converge=5000):
    current_score = sf_pca(pca, pca_component, g, sg, df, scoring_functions, scoring_function_names)
    op = operator.gt
    current_sg = sg
    scores = [current_score]
    best_sg = current_sg
    best_score = current_score
    convergence = 0
    iteration = 0
    while iteration < max_iter and convergence < max_converge:
        if iteration % 100 == 0:
            print(iteration)
        proposed_sg, change = step_change(g, current_sg)
        # No change is made to the graph as there are no new possibilities to explore
        if change == False:
            return best_sg, scores
        proposed_score = sf_pca(pca, pca_component, g, proposed_sg, df, scoring_functions, scoring_function_names)
        if evaluate_step(current_score, proposed_score, op) is True:
            convergence = 0
            current_sg = proposed_sg
            current_score = proposed_score
            if op(current_score, best_score):
                best_sg = current_sg
                best_score = current_score
        else:
            convergence += 1
            pass
        scores.append(current_score)
        iteration += 1
    return best_sg, scores


def resort_scores(all_scores, best_sgs):
    zipped_scores = zip(all_scores, best_sgs)
    sorted_zipped = sorted(zipped_scores, reverse=True, key=lambda x: x[0])
    scores, graphs = zip(*sorted_zipped)
    return scores, graphs


def format_perc(val):
    format_val = str(round((val * 100), 2)) + "%"
    return format_val


def sort_components(pca_component, all_scores, pca):
    if type(pca_component) != str and pca_component <= len(pca.components_) and pca_component >= 1:
        comp_score, sorted_graphs = resort_scores(all_scores['PCA_' + str(pca_component)], all_scores['graph'])
        return comp_score, sorted_graphs
    else:
        print("Error: PCA has been fit with", len(pca.components_), "components.",
              "User has requested", pca_component, "which does not exist.")
        return "Error"


class GraphPackage:

    def __init__(self):
        self.scoring_function_names = ['Internal Edges', 'Internal Density',
                                       'Average Degree', 'Internal Average Degree',
                                       'Degree Variance', 'Internal Degree Variance',
                                       'Internal Directional Imbalance', 'Directional Imbalance',
                                       'Internal Clustering Coefficient', 'Clustering Coefficient',
                                       'Internal to External Edge Ratio', 'Internal Percentage of Edges in a Cycle',
                                       'Percentage of Reciprocal Edges', 'Internal Percentage of Reciprocal Edges']
        self.scoring_functions = [sf_edge_count, sf_internal_density,
                                  sf_average_degree, sf_interal_average_degree,
                                  sf_degree_variance, sf_internal_degree_variance,
                                  sf_internal_directional_imbalance, sf_external_directional_imbalance,
                                  sf_internal_clustering_coefficient, sf_clustering_coefficient,
                                  sf_ite_edge_ratio, sf_complement_flow_hierarchy,
                                  sf_recipetaory, sf_internal_recipetaory]

    def fit(self, g, k, n=''):
        if n == '':
            n = min_samples(g, k)
        scoring_function_names = self.scoring_function_names
        scoring_functions = self.scoring_functions
        all_scores = sample_graphs(g, k, n, scoring_functions, scoring_function_names)
        no_samples = len(all_scores['graph'])
        print("Sampling complete,", no_samples, "samples created and scored.")
        all_scores = standardise_all(all_scores, scoring_function_names)
        all_scores, pca = fit_pca(all_scores, scoring_function_names)
        sorted_graphs = sort_scores(all_scores)
        stat_df = summary_statistics(all_scores, scoring_function_names)
        self.graphs_ = sorted_graphs
        self.scores_ = all_scores
        self.stats_ = stat_df
        self.full_graph_ = g
        self.pca_ = pca
        return self

    def summary(self):
        # needs to be replaced outside of notebook
        display(self.stats_)

    def outliers(self, j=5, pca_component=''):
        sorted_graphs = self.graphs_
        pca = self.pca_
        all_scores = self.scores_
        scoring_function_names = self.scoring_function_names
        stat_df = self.stats_
        if pca_component != '':
            try:
                comp_score, sorted_graphs = sort_components(pca_component, all_scores, pca)
            except:
                return self
        graphs = []
        for i in range(j):
            graph = sorted_graphs[i]
            print(
                '-----------------------------------------------------------------------------------------------------------')
            nx.draw(graph)
            plt.show()
            graph_index = all_scores['graph'].index(graph)
            if pca_component == '':
                pca_dist = all_scores['PCA Distance'][graph_index]
            else:
                pca_dist = comp_score[i]
            print_summary(i, stat_df, graph_index, all_scores, pca_dist, scoring_function_names)
            graphs.append(graph)
        self.outlier_graphs = graphs
        return self

    def optimize_outliers(self, j=5, pca_component='', max_iter=10000, max_converge='', repetition = False):
        g = self.full_graph_
        sorted_graphs = self.graphs_
        pca = self.pca_
        stat_df = self.stats_
        scoring_functions = self.scoring_functions
        scoring_function_names = self.scoring_function_names
        all_scores = self.scores_

        if repetition == False:
            no_reps = 1
        else:
            no_reps = repetition
            print("WARNING: Repeating optimisation process will increase running time")

        if pca_component != '':
            comp_score, sorted_graphs = sort_components(pca_component, all_scores, pca)

        # Unless the maximum convergence is set, then the below will estimate how many will need to be run
        # To maximize the probability that all possibilities are used.
        # Better than just using an int as it will be dependent on k, therefore small subgraphs don't require lots
        # of un-required optimisation
        if max_converge == '':
            all_deg = stat_df[stat_df.index == 'Average Degree']['Mean'].values[0]
            int_deg = stat_df[stat_df.index == 'Internal Average Degree']['Mean'].values[0]
            ext_deg = all_deg - int_deg
            k = len(sorted_graphs[0].nodes)
            max_converge = (k * (k * ext_deg)) * 1.2

        best_sgs = []
        opt_scores = []
        for i in range(j):
            print("Optimising graph", str(i + 1))
            sg = sorted_graphs[i]
            best_score = 0
            abs_best_sgs = ''
            for l in range(no_reps):
                best_sg, scores = stepwise_optimisation(g, sg, pca, pca_component,
                                                        stat_df, scoring_functions, scoring_function_names,
                                                        max_iter, max_converge)
                if scores[-1] > best_score:
                    best_score = scores[-1]
                    abs_best_sgs = best_sg
            best_sgs.append(abs_best_sgs)
            opt_scores.append(best_score)
        opt_scores, graphs = resort_scores(all_scores, best_sgs)

        score_dict = create_scoring_dict(scoring_function_names)
        for sg in graphs:
            score_dict = fill_scoring_dict(g, sg, score_dict, scoring_functions, scoring_function_names)

        pca_dists = []
        for sg in graphs:
            pca_dists.append(sf_pca(pca, pca_component, g, sg, stat_df, scoring_functions, scoring_function_names))
        score_dict['PCA Distance'] = pca_dists

        for i in range(len(graphs)):
            pca_dist = score_dict['PCA Distance'][i]
            print_summary(i, stat_df, i, score_dict, pca_dist, scoring_function_names)
            nx.draw(graphs[i])
            plt.show()
        self.optimal_scores_ = score_dict
        self.optimal_graphs_ = graphs
        return self

    def draw(self, g, sg=''):
        color_map = []
        node_size = []
        size = max(100, 10000 / len(g.nodes))
        size = min(1000, size)
        if sg != '':
            for node in g:
                if node in list(sg.nodes):
                    color_map.append('red')
                    node_size.append(size * 5)
                else:
                    color_map.append('white')
                    node_size.append(size)
        else:
            color_map = ['red'] * len(g.nodes)
            node_size = [size] * len(g.nodes)

        plt.figure(3, figsize=(14, 10))
        nx.draw(g, node_color=color_map, node_size=node_size, arrowsize=size / 20)
        plt.show()

    def draw_pca(self):
        scores = self.scores_
        pca = self.pca_
        comps = []
        columns = []
        for i in range(len(pca.components_)):
            comps.append(scores['PCA_' + str(i + 1)])
            columns.append('Principle Component ' + str(i + 1))
        pca_df = pd.DataFrame(comps).transpose()
        pca_df.columns = columns
        max_val = max(pca_df.max())
        min_val = min(pca_df.min())
        max_axis = max_val + abs(0.1 * max_val)
        min_axis = min_val - abs(0.1 * min_val)
        g = sns.pairplot(pca_df)
        for x in range(len(g.axes)):
            for y in range(len(g.axes)):
                g.axes[x, y].set_xlim((min_axis, max_axis))
                g.axes[x, y].set_ylim((min_axis, max_axis))
        plt.show()

    def summary_pca(self):
        scoring_function_names = self.scoring_function_names
        pca = self.pca_
        comps = pca.components_
        for j in range(len(pca.explained_variance_ratio_)):
            abs_comps = [abs(x) for x in comps[j]]
            total = sum(abs_comps)
            percs = []
            for i in comps[j]:
                percs.append(abs(i) / total)
            percs, sfn = resort_scores(percs, scoring_function_names)
            evr = format_perc(pca.explained_variance_ratio_[j])
            print("PCA", j + 1, "explains", evr, "of the sub-graphs variance.")
            print("The top components are as followed:")
            for i in range(len(percs)):
                if percs[i] > 1 / len(percs):
                    fmt_perc = format_perc(percs[i])
                    print("The", sfn[i], "score is", fmt_perc, "of the component.")
            print("")
