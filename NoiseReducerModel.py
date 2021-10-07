
import numpy as np

class NoiseReducer:

    def __init__(self, n_activities, t_prior_coeff=0, t_likelihood_coeff=0, t_centroid_coeff=0):
        self.n_activities = n_activities
        self.t_prior_coeff = t_prior_coeff
        self.t_likelihood_coeff = t_likelihood_coeff
        self.t_centroid_coeff = t_centroid_coeff
        self.max_len_trace = 0
        self.t_centroid = None
        self.recorded_traces = None

    def gradientDescent_multiple_variables(self, prior_traces, ground_truth_traces, t_likelihood, t_centroid, theta, alpha, num_iters):
        m = len(prior_traces)
        for i in range(num_iters):
            J = 0
            for prior_trace, y, t_like in zip(prior_traces,  ground_truth_traces, t_likelihood):
                prior_trace = np.array(prior_trace)

                # get the posterior matix
                M_pred = theta[0] * prior_trace + theta[1] * t_like + theta[2] * t_centroid

                # update the weights theta
                theta[0] += (2 * alpha / m) * np.sum((y - M_pred) * prior_trace)
                theta[1] += (2 * alpha / m) * np.sum((y - M_pred) * t_like)
                theta[2] += (2 * alpha / m) * np.sum((y - M_pred) * t_centroid)

                # calculate the cost function after one iteration on all train data
                J += np.sum(np.square(y - M_pred))
            if i % 5 == 0:
                print(f'The loss at iteration {i} is {J / m}')

        return theta[0], theta[1], theta[2]


    def calc_t_likelihood_helper(self, traces, new_trace, traces_frequencies, distance_func, longest_trace_len, n_activities):
        distances = dict()
        for trace in traces:
            distance = distance_func(np.array(new_trace), np.array(self.trace_to_matrix(trace, n_activities, longest_trace_len)))
            distances[trace] = distance
        scores = self.calc_scores(distances)
        likelihood_mat = np.zeros((n_activities, longest_trace_len))
        denominator = 0

        for trace in traces:
            likelihood_mat += np.array(self.trace_to_matrix(trace, n_activities, longest_trace_len)) * traces_frequencies[trace]                                                                                                                 * scores[trace]
            denominator += traces_frequencies[trace] * scores[trace]
        return likelihood_mat / denominator


    def traces_frequency(self, traces):
        freq_dict = {}
        for trace in traces:
            t = tuple(trace)
            if t in freq_dict:
                freq_dict[t] += 1
            else:
                freq_dict[t] = 1

        return freq_dict


    def modify_newtrace_dims(self, trace, required_length):
        n_additional_columns = required_length - len(trace[0])
        for row in trace[:-1]:
            row += [0] * n_additional_columns
        trace[-1] += [1] * n_additional_columns     # background label
        return trace


    def modify_all_traces_dims(self, traces, required_length):
        modified_traces = []
        for trace in traces:
            modified_trace = self.modify_newtrace_dims(trace, required_length)
            modified_traces.append(modified_trace)
        return modified_traces


    def calc_scores(self, distances_dict):
        scores = dict()
        total_score = sum([np.exp(-value) for value in distances_dict.values()])

        for key in distances_dict.keys():
            scores[key] = np.exp(-distances_dict[key]) / total_score

        return scores

    def calculate_centroid(self, traces):
        longest_trace_len = max([len(trace) for trace in traces])
        centroid = np.zeros((self.n_activities, longest_trace_len))
        m = len(traces)
        for trace in traces:
            matrix_form = self.trace_to_matrix(trace, self.n_activities)
            if len(matrix_form[0]) != longest_trace_len:
                matrix_form = self.modify_newtrace_dims(matrix_form, longest_trace_len)
            centroid  = centroid + matrix_form
        return centroid / m


    def convert_stringtraces_to_matrices(self, traces, n_activities, n_cols=None):
        traces_in_matrix_form = []
        for trace in traces:
            trace_matrix = self.trace_to_matrix(trace, n_activities = n_activities, n_cols=None)
            traces_in_matrix_form.append(trace_matrix)
        return traces_in_matrix_form


    def calc_t_likelihood(self, existing_traces, new_trace, n_activities):
        ''' Recieves existing_traces as list of strings and new_trace as matrix'''
        longest_trace_len = max(max([len(trace) for trace in existing_traces]), len(new_trace[0])) # new trace is a matrix
        if len(new_trace[0]) != longest_trace_len:
            new_trace = self.modify_newtrace_dims(new_trace, longest_trace_len)
        traces_frequencies = self.traces_frequency(existing_traces)
        unique_traces = [trace for trace in set(tuple(trace) for trace in existing_traces)]
        T_likelihood = self.calc_t_likelihood_helper(unique_traces, new_trace, traces_frequencies, self.calc_Frobenius_norm, longest_trace_len, n_activities)
        return T_likelihood

    def calc_Frobenius_norm(self, mat1, mat2):
        return np.linalg.norm(mat1 - mat2)


    def t_likelihood_for_all_traces(self, noisy_traces, ground_truth_traces):
        likelihoods_lst = []
        for trace in noisy_traces:
            t_likelihood = self.calc_t_likelihood(ground_truth_traces, trace, self.n_activities)
            likelihoods_lst.append(t_likelihood)

        return likelihoods_lst


    def trace_to_matrix(self, trace, n_activities, n_cols=None):
        '''This is the original trace to matrix conventer. It get as input
            list of string and returns list of lists'''
        width = len(trace)
        trace_mat = [[0] * width for _ in range(n_activities)]

        for i in range(width):
            trace_mat[ord(trace[i])-ord('A')][i] = 1

        if n_cols:
            if n_cols < width:
                raise ValueError("n_locs must be larger than the length of the trace")
            for trace in trace_mat:
                trace += [0] * (n_cols - width)

        return trace_mat


    def normalize_probabilities(self, posterior_trace):
        '''normalizes returned posterior trace prediction'''
        n_columns = len(posterior_trace[0])
        new_mat = []
        for column in posterior_trace.T:
            column_sum = column.sum()
            column = column / column_sum
            new_mat.append(column)
        new_mat = np.stack(new_mat, axis = 1 )
        return new_mat


    def normalize_coeff(self, t_prior_coeff, t_likelihood_coeff, t_centroid_coeff):
        '''normalizing coefficients of weights so the prediciton probability will sum to 1 for each activity'''
        coeff_sum = t_prior_coeff + t_likelihood_coeff + t_centroid_coeff
        return t_prior_coeff / coeff_sum, t_likelihood_coeff / coeff_sum, t_centroid_coeff / coeff_sum


    def train(self, X_train, Y_train, lr = 0.0001, n_iter = 1000):
        self.recorded_traces = Y_train
        self.max_len_trace = max([len(trace) for trace in Y_train])
        traces_likelihood_mats = self.t_likelihood_for_all_traces(X_train, Y_train)
        self.t_centroid = self.calculate_centroid(Y_train)
        Y_train_mats = self.convert_stringtraces_to_matrices(Y_train, self.n_activities)
        Y_train_modified_dimensions = self.modify_all_traces_dims(Y_train_mats, self.max_len_trace)
        self.t_prior_coeff, self.t_likelihood_coeff, self.t_centroid_coeff = self.gradientDescent_multiple_variables(X_train, Y_train_modified_dimensions, traces_likelihood_mats, self.t_centroid, [self.t_prior_coeff, self.t_likelihood_coeff, self.t_centroid_coeff], lr, n_iter)
        # self.t_prior_coeff, self.t_likelihood_coeff, self.t_centroid_coeff = self.normalize_coeff(t_prior_coeff, t_likelihood_coeff, t_centroid_coeff)

    def predict_converged(self, t_prior, traces=None, delta=0.005, dist_func=None, alpha=0.8, beta=0.1, gamma=0.1):
        if traces is None:
            traces = self.recorded_traces
        if dist_func is None:
            dist_func = self.calc_Frobenius_norm

        # compute initial values
        t_prior = np.array(t_prior)
        t_centroid = self.calculate_centroid(traces, self.n_activities)
        t_likelihood = self.calc_t_likelihood(traces, t_prior, self.n_activities)
        t_posterior_prev = alpha * t_prior + beta * t_likelihood + gamma * t_centroid

        # compute updated values
        t_likelihood_curr = self.calc_t_likelihood(traces, t_posterior_prev, self.n_activities)
        t_posterior_curr = alpha * t_posterior_prev + beta * t_likelihood_curr + gamma * t_centroid

        # iterate until convergence
        i = 0
        while dist_func(t_posterior_prev, t_posterior_curr) > delta:
            i += 1
            print(f'iteration #{i}')
            print('curr distance between traces: ', dist_func(t_posterior_prev, t_posterior_curr))
            t_posterior_prev = t_posterior_curr
            t_likelihood_curr = self.calc_t_likelihood(traces, t_posterior_prev, self.n_activities)
            t_posterior_curr = alpha * t_posterior_prev + beta * t_likelihood_curr + gamma * t_centroid

        return t_posterior_curr


    def predict(self, prior_trace):
        prior_trace = np.array(prior_trace)
        t_likelihood = self.calc_t_likelihood(self.recorded_traces, prior_trace, self.n_activities)

        t_posterior = self.t_prior_coeff * prior_trace + self.t_likelihood_coeff * t_likelihood + self.t_centroid_coeff * self.t_centroid
        #print('prior trace was: ')
        #for row in prior_trace:
        #    print(*row)
        #print()
        #print('posterior trace unnormalized is: ')
        #for row in t_posterior:
        #    print(*row)
        #print()
        t_posterior = self.normalize_probabilities(t_posterior)
        #print('posterior trace normalized is: ')
        #for row in t_posterior:
        #    print(*row)
        #print()
        return t_posterior

