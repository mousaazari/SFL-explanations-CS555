"""
Functions for explaining classifiers that use Image data.
"""
import copy
from functools import partial

import numpy as np
import sklearn
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
from tqdm.auto import tqdm


from . import lime_base
from .wrappers.scikit_image import SegmentationAlgorithm


class ImageExplanation(object):
    def __init__(self, image, segments):
        """Init function.

        Args:
            image: 3d numpy array
            segments: 2d numpy array, with the output from skimage.segmentation
        """
        self.image = image
        self.segments = segments
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = {}
        self.score = {}

    def get_image_and_mask(self, label, positive_only=True, negative_only=False, hide_rest=False,
                           num_features=5, min_weight=0.):
        """Init function.

        Args:
            label: label to explain
            positive_only: if True, only take superpixels that positively contribute to
                the prediction of the label.
            negative_only: if True, only take superpixels that negatively contribute to
                the prediction of the label. If false, and so is positive_only, then both
                negativey and positively contributions will be taken.
                Both can't be True at the same time
            hide_rest: if True, make the non-explanation part of the return
                image gray
            num_features: number of superpixels to include in explanation
            min_weight: minimum weight of the superpixels to include in explanation

        Returns:
            (image, mask), where image is a 3d numpy array and mask is a 2d
            numpy array that can be used with
            skimage.segmentation.mark_boundaries
        """
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_only & negative_only:
            raise ValueError("Positive_only and negative_only cannot be true at the same time.")
        segments = self.segments
        image = self.image
        exp = self.local_exp[label]
        mask = np.zeros(segments.shape, segments.dtype)
        if hide_rest:
            temp = np.zeros(self.image.shape)
        else:
            temp = self.image.copy()
        if positive_only:
            fs = [x[0] for x in exp
                  if x[1] > 0 and x[1] > min_weight][:num_features]
        if negative_only:
            fs = [x[0] for x in exp
                  if x[1] < 0 and abs(x[1]) > min_weight][:num_features]
        if positive_only or negative_only:
            for f in fs:
                temp[segments == f] = image[segments == f].copy()
                mask[segments == f] = 1
            return temp, mask
        else:
            for f, w in exp[:num_features]:
                if np.abs(w) < min_weight:
                    continue
                c = 0 if w < 0 else 1
                mask[segments == f] = -1 if w < 0 else 1
                temp[segments == f] = image[segments == f].copy()
                temp[segments == f, c] = np.max(image)
            return temp, mask


class LimeImageExplainer(object):
    """Explains predictions on Image (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)

    def explain_instance(self, image, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         progress_bar=True):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            image: 3 dimension RGB image. If this is only two dimensional,
                we will assume it's a grayscale image and call gray2rgb.
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            hide_color: TODO
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: TODO
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            segmentation_fn: SegmentationAlgorithm, wrapped skimage
            segmentation function
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            An ImageExplanation object (see lime_image.py) with the corresponding
            explanations.
        """


        parser = argparse.ArgumentParser(
            description='DeepCover: Uncover Bugs in Deep Learning')

        parser.add_argument('model', action='store', nargs='+', help='The input neural network model (.h5)')

        parser.add_argument(
            '--cover', metavar='ss', action='store', help='The covering method: ss, sv, ds, dv', default='ss')

        args = parser.parse_args()

        model = load_model(args.model[0])

        if not (args.cover in ['ss', 'sv', 'ds', 'dv']):
            print('Covering method cannot be recognized: ' + args.cover)
            sys.exit(0)

        print('\n== WARNING == \n')
        print(
            'The input model:       ' + args.model[0] + '\n' +
            'The covering method:   ' + args.cover + '\n'
        )
        print('This keras compatible implementation of DeepCover testing is currently under deverlopment...\n')
        print('=============\n')

        for i in range(1, I + 2):
            M = len(act[i])  # number of neurons at layer i
            for j in range(0, M):
                #### for layer (I+1) we only need to access one neuron
                if i == I + 1 and j != K: continue
                constraint = [[], []]
                constraint[0].append("x_" + str(i) + "_" + str(j))
                constraint[1].append(-1)
                for k in range(0, len(act[i - 1])):
                    constraint[0].append("x_" + str(i - 1) + "_" + str(k))
                    if i == 1 or act[i - 1][k] > 0:
                        if not (i - 1 == I and k == J):
                            constraint[1].append(nnet.weights[i - 1][k][j])
                        else:
                            constraint[1].append(0)
                    else:
                        if not (i - 1 == I and k == J):
                            constraint[1].append(0)
                        else:
                            constraint[1].append(nnet.weights[i - 1][k][j])
                constraints.append(constraint)
                rhs.append(-nnet.biases[i][j])
                constraint_senses.append("E")
                constraint_names.append("eq:" + "x_" + str(i) + "_" + str(j))

                ###### ReLU
                if i < N - 1:
                    _constraint = [[], []]
                    _constraint[0].append("x_" + str(i) + "_" + str(j))
                    _constraint[1].append(1)
                    constraints.append(_constraint)
                    rhs.append(0)
                    if not ((i == I and j == J) or (i == I + 1 and j == K)):
                        if act[i][j] > 0:
                            constraint_senses.append("G")
                        else:
                            constraint_senses.append("L")
                        constraint_names.append("relu:" + "x_" + str(i) + "_" + str(j))
                    else:
                        if act[i][j] > 0:
                            constraint_senses.append("L")
                        else:
                            constraint_senses.append("G")
                        constraint_names.append("not relu:" + "x_" + str(i) + "_" + str(j))

        if I == N - 2:  # I+1==N-1
            #### Now, we are at the output layer
            #### x_{N-1, K}>=x_{N-1,old_label}
            label = np.argmax(act[N - 1])
            for i in range(0, len(act[N - 1])):
                if i != K: continue
                constraint = [[], []]
                constraint[0].append("x_" + str(N - 1) + "_" + str(i))
                constraint[1].append(1)
                # constraint[0].append("x_"+str(N-1)+"_"+str(label))
                # constraint[1].append(-1)
                constraints.append(constraint)
                rhs.append(0.0)
                if act[N - 1][K] > 0:
                    constraint_senses.append("L")
                else:
                    constraint_senses.append("G")
                constraint_names.append("not K")

            ###### solve
        try:
            problem = cplex.Cplex()
            problem.variables.add(obj=objective,
                                  lb=lower_bounds,
                                  ub=upper_bounds,
                                  names=var_names)
            problem.linear_constraints.add(lin_expr=constraints,
                                           senses=constraint_senses,
                                           rhs=rhs,
                                           names=constraint_names)
            problem.solve()

            ####
            d = problem.solution.get_values("d")
            new_x = []
            for i in range(0, len(X)):
                v = (problem.solution.get_values('x_0_' + str(i)))
                if v < 0 or v > 1: return False, _, _
                new_x.append(v)

            if d == 0 or d == 1:
                return False, _, _, _, _

            # print problem.variables.get_num(), problem.linear_constraints.get_num()
            return True, new_x, d, problem.variables.get_num(), problem.linear_constraints.get_num()

        except:
            return False, [], -1, -1, -1

            try:
                d = problem.solution.get_values("d")
                print
                'd is {0}'.format(d)
                new_x = []
                # for i in len(X):
                #  new_x.append(problem.solution.get_values('x_0_'+str(i)))
                # return True, new_x, d
            except:
                print
                'Exception for feasible model???'
                sys.exit(0)

        def rp_dsc(I, J, nnet, X, act):

            var_names = ['d']
            objective = [1]
            lower_bounds = [0.0]
            upper_bounds = [1.0]

            N = len(act)  # #layers
            for i in range(0, N):
                M = len(act[i])  # #neurons at layer i
                for j in range(0, M):
                    var_names.append('x_' + str(i) + '_' + str(j))
                    objective.append(0)
                    lower_bounds.append(-cplex.infinity)
                    upper_bounds.append(cplex.infinity)

            constraints = []
            rhs = []
            constraint_senses = []
            constraint_names = []

            for i in range(0, len(X)):
                # x<=x0+d
                constraints.append([[0, i + 1], [-1, 1]])
                rhs.append(X[i])
                constraint_senses.append("L")
                constraint_names.append("x<=x" + str(i) + "+d")
                # x>=x0-d
                constraints.append([[0, i + 1], [1, 1]])
                rhs.append(X[i])
                constraint_senses.append("G")
                constraint_names.append("x>=x" + str(i) + "-d")
                # x<=1
                constraints.append([[i + 1], [1]])
                rhs.append(1.0)
                constraint_senses.append("L")
                constraint_names.append("x<=1")
                # x>=0
                constraints.append([[i + 1], [1]])
                rhs.append(0.0)
                constraint_senses.append("G")
                constraint_names.append("x>=0")

            # there is nothing to constrain for layer 0
            # and we start from layer 1
            # the last layer shall be handled individually
            for i in range(1, I + 1):
                M = len(act[i])  # number of neurons at layer i
                for j in range(0, M):
                    #### for layer (I+1) we only need to access one neuron
                    if i == I and j != J: continue
                    constraint = [[], []]
                    constraint[0].append("x_" + str(i) + "_" + str(j))
                    constraint[1].append(-1)
                    for k in range(0, len(act[i - 1])):
                        constraint[0].append("x_" + str(i - 1) + "_" + str(k))
                        if i == 1 or act[i - 1][k] > 0:
                            constraint[1].append(nnet.weights[i - 1][k][j])
                        else:
                            constraint[1].append(0)
                    constraints.append(constraint)
                    rhs.append(-nnet.biases[i][j])
                    constraint_senses.append("E")
                    constraint_names.append("eq:" + "x_" + str(i) + "_" + str(j))

                    ###### ReLU
                    if i < N - 1:
                        _constraint = [[], []]
                        _constraint[0].append("x_" + str(i) + "_" + str(j))
                        _constraint[1].append(1)
                        constraints.append(_constraint)
                        if not (i == I and j == J):
                            rhs.append(0)
                            if act[i][j] > 0:
                                constraint_senses.append("G")
                            else:
                                constraint_senses.append("L")
                            constraint_names.append("relu:" + "x_" + str(i) + "_" + str(j))
                        else:  ## I+1, K
                            ## ReLU sign does not change
                            rhs.append(0)
                            if act[i][j] > 0:
                                constraint_senses.append("L")
                            else:
                                constraint_senses.append("G")
                            constraint_names.append("relu:" + "x_" + str(i) + "_" + str(j))

            if I == N - 1:  # I+1==N-1
                #### Now, we are at the output layer
                #### x_{N-1, K}>=x_{N-1,old_label}
                label = np.argmax(act[N - 1])
                for i in range(0, len(act[N - 1])):
                    if i != J: continue
                    constraint = [[], []]
                    constraint[0].append("x_" + str(N - 1) + "_" + str(i))
                    constraint[1].append(1)
                    constraints.append(constraint)

                    ##1) ReLU sign does not change
                    rhs.append(0)
                    if act[I][J] > 0:
                        constraint_senses.append("L")
                    else:
                        constraint_senses.append("G")
                    constraint_names.append("relu sign:" + "x_" + str(I) + "_" + str(J))

            ###### solve
            try:
                problem = cplex.Cplex()
                problem.variables.add(obj=objective,
                                      lb=lower_bounds,
                                      ub=upper_bounds,
                                      names=var_names)
                problem.linear_constraints.add(lin_expr=constraints,
                                               senses=constraint_senses,
                                               rhs=rhs,
                                               names=constraint_names)
                problem.solve()

                ####
                d = problem.solution.get_values("d")
                new_x = []
                for i in range(0, len(X)):
                    v = (problem.solution.get_values('x_0_' + str(i)))
                    if v < 0 or v > 1: return False, _, _
                    new_x.append(v)

                if d == 0 or d == 1:
                    return False, _, _

                return True, new_x, d

            except:
                return False, [], -1

                try:
                    d = problem.solution.get_values("d")
                    print
                    'd is {0}'.format(d)
                    new_x = []
                    # for i in len(X):
                    #  new_x.append(problem.solution.get_values('x_0_'+str(i)))
                    # return True, new_x, d
                except:
                    print
                    'Exception for feasible model???'
                    sys.exit(0)

        def rp_svc(I, J, K, nnet, X, act, sfactor):

            var_names = ['d']
            objective = [1]
            lower_bounds = [0.0]
            upper_bounds = [1.0]

            N = len(act)  # #layers
            for i in range(0, N):
                M = len(act[i])  # #neurons at layer i
                for j in range(0, M):
                    var_names.append('x_' + str(i) + '_' + str(j))
                    objective.append(0)
                    lower_bounds.append(-cplex.infinity)
                    upper_bounds.append(cplex.infinity)

            constraints = []
            rhs = []
            constraint_senses = []
            constraint_names = []

            for i in range(0, len(X)):
                # x<=x0+d
                constraints.append([[0, i + 1], [-1, 1]])
                rhs.append(X[i])
                constraint_senses.append("L")
                constraint_names.append("x<=x" + str(i) + "+d")
                # x>=x0-d
                constraints.append([[0, i + 1], [1, 1]])
                rhs.append(X[i])
                constraint_senses.append("G")
                constraint_names.append("x>=x" + str(i) + "-d")
                # x<=1
                constraints.append([[i + 1], [1]])
                rhs.append(1.0)
                constraint_senses.append("L")
                constraint_names.append("x<=1")
                # x>=0
                constraints.append([[i + 1], [1]])
                rhs.append(0.0)
                constraint_senses.append("G")
                constraint_names.append("x>=0")

            # there is nothing to constrain for layer 0
            # and we start from layer 1
            # the last layer shall be handled individually
            for i in range(1, I + 2):
                M = len(act[i])  # number of neurons at layer i
                for j in range(0, M):
                    #### for layer (I+1) we only need to access one neuron
                    if i == I + 1 and j != K: continue
                    constraint = [[], []]
                    constraint[0].append("x_" + str(i) + "_" + str(j))
                    constraint[1].append(-1)
                    for k in range(0, len(act[i - 1])):
                        constraint[0].append("x_" + str(i - 1) + "_" + str(k))
                        if i == 1 or act[i - 1][k] > 0:
                            if not (i - 1 == I and k == J):
                                constraint[1].append(nnet.weights[i - 1][k][j])
                            else:
                                constraint[1].append(0)
                        else:
                            if not (i - 1 == I and k == J):
                                constraint[1].append(0)
                            else:
                                constraint[1].append(nnet.weights[i - 1][k][j])
                    constraints.append(constraint)
                    rhs.append(-nnet.biases[i][j])
                    constraint_senses.append("E")
                    constraint_names.append("eq:" + "x_" + str(i) + "_" + str(j))

                    ###### ReLU
                    if i < N - 1:
                        _constraint = [[], []]
                        _constraint[0].append("x_" + str(i) + "_" + str(j))
                        _constraint[1].append(1)
                        constraints.append(_constraint)
                        if not ((i == I and j == J) or (i == I + 1 and j == K)):
                            rhs.append(0)
                            if act[i][j] > 0:
                                constraint_senses.append("G")
                            else:
                                constraint_senses.append("L")
                            constraint_names.append("relu:" + "x_" + str(i) + "_" + str(j))
                        elif (i == I and j == J):  # Activation change
                            rhs.append(0)
                            if act[i][j] > 0:
                                constraint_senses.append("L")
                            else:
                                constraint_senses.append("G")
                            constraint_names.append("not relu:" + "x_" + str(i) + "_" + str(j))
                        else:  ## I+1, K
                            ## ReLU sign does not change
                            rhs.append(0)
                            if act[i][j] > 0:
                                constraint_senses.append("G")
                            else:
                                constraint_senses.append("L")
                            constraint_names.append("relu:" + "x_" + str(i) + "_" + str(j))

                            ## ReLU value changed
                            _constraint = [[], []]
                            _constraint[0].append("x_" + str(i) + "_" + str(j))
                            _constraint[1].append(1)
                            constraints.append(_constraint)
                            rhs.append(sfactor * act[I + 1][K])
                            if act[i][j] > 0:
                                if sfactor > 1.0:
                                    constraint_senses.append("G")
                                else:
                                    constraint_senses.append("L")
                            else:
                                if sfactor > 1.0:
                                    constraint_senses.append("L")
                                else:
                                    constraint_senses.append("G")
                            constraint_names.append("relu value change:" + "x_" + str(i) + "_" + str(j))

            if I == N - 2:  # I+1==N-1
                #### Now, we are at the output layer
                #### x_{N-1, K}>=x_{N-1,old_label}
                label = np.argmax(act[N - 1])
                for i in range(0, len(act[N - 1])):
                    if i != K: continue
                    constraint = [[], []]
                    constraint[0].append("x_" + str(N - 1) + "_" + str(i))
                    constraint[1].append(1)
                    constraints.append(constraint)

                    ##1) ReLU sign does not change
                    rhs.append(0)
                    if act[I + 1][K] > 0:
                        constraint_senses.append("G")
                    else:
                        constraint_senses.append("L")
                    constraint_names.append("relu sign:" + "x_" + str(I + 1) + "_" + str(K))

                    ## ReLU value changed
                    _constraint = [[], []]
                    _constraint[0].append("x_" + str(I + 1) + "_" + str(K))
                    _constraint[1].append(1)
                    constraints.append(_constraint)
                    rhs.append(sfactor * act[I + 1][K])
                    if act[I + 1][K] > 0:
                        if sfactor > 1.0:
                            constraint_senses.append("G")
                        else:
                            constraint_senses.append("L")
                    else:
                        if sfactor > 1.0:
                            constraint_senses.append("L")
                        else:
                            constraint_senses.append("G")
                    constraint_names.append("relu value change:" + "x_" + str(I + 1) + "_" + str(K))

            ###### solve
            try:
                problem = cplex.Cplex()
                problem.variables.add(obj=objective,
                                      lb=lower_bounds,
                                      ub=upper_bounds,
                                      names=var_names)
                problem.linear_constraints.add(lin_expr=constraints,
                                               senses=constraint_senses,
                                               rhs=rhs,
                                               names=constraint_names)
                problem.solve()

                ####
                d = problem.solution.get_values("d")
                new_x = []
                for i in range(0, len(X)):
                    v = (problem.solution.get_values('x_0_' + str(i)))
                    if v < 0 or v > 1: return False, _, _
                    new_x.append(v)

                if d == 0 or d == 1:
                    return False, _, _

                return True, new_x, d

            except:
                return False, [], -1

                try:
                    d = problem.solution.get_values("d")
                    print
                    'd is {0}'.format(d)
                    new_x = []
                    # for i in len(X):
                    #  new_x.append(problem.solution.get_values('x_0_'+str(i)))
                    # return True, new_x, d
                except:
                    print
                    'Exception for feasible model???'
                    sys.exit(0)

        def rp_dvc(I, J, nnet, X, act, sfactor):

            var_names = ['d']
            objective = [1]
            lower_bounds = [0.0]
            upper_bounds = [1.0]

            N = len(act)  # #layers
            for i in range(0, N):
                M = len(act[i])  # #neurons at layer i
                for j in range(0, M):
                    var_names.append('x_' + str(i) + '_' + str(j))
                    objective.append(0)
                    lower_bounds.append(-cplex.infinity)
                    upper_bounds.append(cplex.infinity)

            constraints = []
            rhs = []
            constraint_senses = []
            constraint_names = []

            for i in range(0, len(X)):
                # x<=x0+d
                constraints.append([[0, i + 1], [-1, 1]])
                rhs.append(X[i])
                constraint_senses.append("L")
                constraint_names.append("x<=x" + str(i) + "+d")
                # x>=x0-d
                constraints.append([[0, i + 1], [1, 1]])
                rhs.append(X[i])
                constraint_senses.append("G")
                constraint_names.append("x>=x" + str(i) + "-d")
                # x<=1
                constraints.append([[i + 1], [1]])
                rhs.append(1.0)
                constraint_senses.append("L")
                constraint_names.append("x<=1")
                # x>=0
                constraints.append([[i + 1], [1]])
                rhs.append(0.0)
                constraint_senses.append("G")
                constraint_names.append("x>=0")

            # there is nothing to constrain for layer 0
            # and we start from layer 1
            # the last layer shall be handled individually
            for i in range(1, I + 1):
                M = len(act[i])  # number of neurons at layer i
                for j in range(0, M):
                    #### for layer (I+1) we only need to access one neuron
                    if i == I and j != J: continue
                    constraint = [[], []]
                    constraint[0].append("x_" + str(i) + "_" + str(j))
                    constraint[1].append(-1)
                    for k in range(0, len(act[i - 1])):
                        constraint[0].append("x_" + str(i - 1) + "_" + str(k))
                        if i == 1 or act[i - 1][k] > 0:
                            constraint[1].append(nnet.weights[i - 1][k][j])
                        else:
                            constraint[1].append(0)
                    constraints.append(constraint)
                    rhs.append(-nnet.biases[i][j])
                    constraint_senses.append("E")
                    constraint_names.append("eq:" + "x_" + str(i) + "_" + str(j))

                    ###### ReLU
                    if i < N - 1:
                        _constraint = [[], []]
                        _constraint[0].append("x_" + str(i) + "_" + str(j))
                        _constraint[1].append(1)
                        constraints.append(_constraint)
                        if not (i == I and j == J):
                            rhs.append(0)
                            if act[i][j] > 0:
                                constraint_senses.append("G")
                            else:
                                constraint_senses.append("L")
                            constraint_names.append("relu:" + "x_" + str(i) + "_" + str(j))
                        else:  ## I+1, K
                            ## ReLU sign does not change
                            rhs.append(0)
                            if act[i][j] > 0:
                                constraint_senses.append("G")
                            else:
                                constraint_senses.append("L")
                            constraint_names.append("relu:" + "x_" + str(i) + "_" + str(j))

                            ## ReLU value changed
                            _constraint = [[], []]
                            _constraint[0].append("x_" + str(i) + "_" + str(j))
                            _constraint[1].append(1)
                            constraints.append(_constraint)
                            rhs.append(sfactor * act[I][J])
                            if act[i][j] > 0:
                                if sfactor > 1.0:
                                    constraint_senses.append("G")
                                else:
                                    constraint_senses.append("L")
                            else:
                                if sfactor > 1.0:
                                    constraint_senses.append("L")
                                else:
                                    constraint_senses.append("G")
                            constraint_names.append("relu value change:" + "x_" + str(i) + "_" + str(j))

            if I == N - 1:  # I+1==N-1
                #### Now, we are at the output layer
                #### x_{N-1, K}>=x_{N-1,old_label}
                label = np.argmax(act[N - 1])
                for i in range(0, len(act[N - 1])):
                    if i != J: continue
                    constraint = [[], []]
                    constraint[0].append("x_" + str(N - 1) + "_" + str(i))
                    constraint[1].append(1)
                    constraints.append(constraint)

                    ##1) ReLU sign does not change
                    rhs.append(0)
                    if act[I][J] > 0:
                        constraint_senses.append("G")
                    else:
                        constraint_senses.append("L")
                    constraint_names.append("relu sign:" + "x_" + str(I) + "_" + str(J))

                    ## ReLU value changed
                    _constraint = [[], []]
                    _constraint[0].append("x_" + str(I) + "_" + str(J))
                    _constraint[1].append(1)
                    constraints.append(_constraint)
                    rhs.append(sfactor * act[I][J])
                    if act[I][J] > 0:
                        if sfactor > 1.0:
                            constraint_senses.append("G")
                        else:
                            constraint_senses.append("L")
                    else:
                        if sfactor > 1.0:
                            constraint_senses.append("L")
                        else:
                            constraint_senses.append("G")
                    constraint_names.append("relu value change:" + "x_" + str(I) + "_" + str(J))

            ###### solve
            try:
                problem = cplex.Cplex()
                problem.variables.add(obj=objective,
                                      lb=lower_bounds,
                                      ub=upper_bounds,
                                      names=var_names)
                problem.linear_constraints.add(lin_expr=constraints,
                                               senses=constraint_senses,
                                               rhs=rhs,
                                               names=constraint_names)
                problem.solve()

                ####
                d = problem.solution.get_values("d")
                new_x = []
                for i in range(0, len(X)):
                    v = (problem.solution.get_values('x_0_' + str(i)))
                    if v < 0 or v > 1: return False, _, _
                    new_x.append(v)

        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
        try:
            segments = segmentation_fn(image)
        except ValueError as e:
            raise e

        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            fudged_image[:] = hide_color

        top = labels

        data, labels = self.data_labels(image, fudged_image, segments,
                                        classifier_fn, num_samples,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)

        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            top = np.argsort(labels[0])[-top_labels:]
            ret_exp.top_labels = list(top)
            ret_exp.top_labels.reverse()
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def data_labels(self,
                    image,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10,
                    progress_bar=True):
        """Generates images and predictions in the neighborhood of this image.

        Args:
            image: 3d numpy array, the image
            fudged_image: 3d numpy array, image to replace original image when
                superpixel is turned off
            segments: segmentation of the image
            classifier_fn: function that takes a list of images and returns a
                matrix of prediction probabilities
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.
            progress_bar: if True, show tqdm progress bar.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_superpixels
                labels: prediction probabilities matrix
        """
        n_features = np.unique(segments).shape[0]
        data = self.random_state.randint(0, 2, num_samples * n_features)\
            .reshape((num_samples, n_features))
        labels = []
        data[0, :] = 1
        imgs = []
        rows = tqdm(data) if progress_bar else data
        for row in rows:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = classifier_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.array(labels)
