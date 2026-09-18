#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

from functools import wraps
from inspect   import getfullargspec, signature
from copy      import deepcopy

import time

import torch
from torch.autograd import Variable
from torch.nn       import Linear, ReLU, CrossEntropyLoss, \
                           Sequential, Conv2d, MaxPool2d,  \
                           Module, Softmax, BatchNorm2d, Dropout
from torch.optim    import Adam, SGD

from src_dir import prob_norm, resid, timer, moving_average, GMRES

from src_dir import StatusPrinter


# NN class
class CNNPredictorOnline_2D(object):

    def __init__(self, D_in, D_out, Area, dx, Model):
        '''
        Constructor of the neural network
        - D_in: dimension input
        - D_out: dimension output
        - Area: Area of the grid
        - dx: spatial step
        - Model: class of the structure of the Neural Network
        '''

        # N is batch size; D_in is input dimension;
        # D_out is output dimension.
        self.D_in  = D_in
        self.D_out = D_out

        # Domain area and finite difference stencil width
        self.Area = Area
        self.dx   = dx

        # Increase layer at every multiple of this factor
        self.Factor = 40

        # Set Pytorch Seed
        torch.manual_seed(0)

        # Construct our model by instantiating the class defined above
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #GPU
        self.model = Model(self.D_in, self.D_out).to(device)

        # Construct our loss function and an Optimizer. The call to
        # model.parameters() in the SGD constructor will contain the learnable
        # parameters of the two nn.Conv modules which are members of the model.
        self.criterion = torch.nn.MSELoss(reduction='mean') # Loss = min ||\tilde x-x^n||

        ### Set optimizer
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        # self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=1e-3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1) # Adam Optimizer

        # x will hold entire training set b data
        # y will hold entire training set solution data
        self.x = torch.empty(0, self.D_in,  self.D_in).to(device)
        self.y = torch.empty(0, self.D_out, self.D_out).to(device)

        # xNew: new b additions to training set at the current time
        # yNew: new solution (x) additions to training set at the current time
        self.xNew = torch.empty(0, self.D_in,  self.D_in)
        self.yNew = torch.empty(0, self.D_out, self.D_out)

        # Set train flag
        self._is_trained = False

        # Diagnostic data => remove in production 
        self.loss_val = list()


    @property
    def is_trained(self):
        # return the train flag
        return self._is_trained


    @is_trained.setter
    def is_trained(self, value):
        # change the train flag
        self._is_trained = value


    @property
    def counter(self):
        # Counter is based off of the data set to be added to training set
        return self.xNew.size(0)


    @timer
    def retrain_timed(self):
        '''
        Function to train the NN, return its runtime thanks to @timer
        '''
    
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #GPU

        # New set at the problem i
        self.xNew = self.xNew.to(device) 
        self.yNew = self.yNew.to(device)

        # Add it to the trainerset list
        self.x=torch.cat((self.x,self.xNew))  
        self.y=torch.cat((self.y,self.yNew))

        self.loss_val = list()  # clear loss val history
        self.loss_val.append(10.0)

        batch_size = 16 # How many sets will use to train
        numEpochs  = 1000 # Max of iters
        e1         = 1e-15 # Tolerance of the loss value
        epoch      = 0 

        while self.loss_val[-1] > e1 and epoch < numEpochs - 1:
            permutation = torch.randperm(self.x.size()[0]) # choose random permutation of all the sets
            for t in range(0, self.x.size()[0], batch_size):

                ## indicies of random batch
                indices = permutation[t:t+batch_size]

                ## dataset batches
                batch_x, batch_y = self.x[indices],self.y[indices]

                ## batch of predictions, forward in the NN
                y_pred = self.model(batch_x,self.x.size(0),self.Factor)

                ## Compute and print loss
                loss = (self.criterion(y_pred, batch_y))
                self.loss_val.append(loss.item())

                ## Print loss to console
                StatusPrinter().update_training(loss.item())

                ## Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch=epoch+1

        ## Add recent data to final batch and take one more step:

        permutation = torch.randperm(self.x.size()[0])
        indices     = permutation[0:0 + batch_size]
        batch_x, batch_y = self.x[indices], self.y[indices]

        # Adding new data to each batch
        # Note: only adding at most 3 data points to each batch
        batch_xMix = torch.cat((batch_x, self.xNew))
        batch_yMix = torch.cat((batch_y, self.yNew))

        ## Forward pass: Compute predicted y by passing x to the model
        y_pred = self.model(batch_xMix, self.x.size(0), self.Factor)

        ## Compute and print loss
        loss = (self.criterion(y_pred, batch_yMix))
        self.loss_val.append(loss.item())

        ## Print loss to console
        StatusPrinter().update_training(loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ## Clear tensors that are used to add data to training set
        self.xNew = torch.empty(0, self.D_in,  self.D_in)
        self.yNew = torch.empty(0, self.D_out, self.D_out)

        ## Print number of parameters to console
        numparams = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        StatusPrinter().update_training_summary(numparams, self.x.size(0))

        self.is_trained = True # set the flag of trained to True


    def add(self, x, y):
        '''
        Add to the set of training {x,y}
        - x : b (random matrix)
        - y: x (solution)
        '''
        # TODO: don't use `torch.cat` in this incremental mode => will scale
        # poorly instead: use batched buffers
        self.xNew = torch.cat(
                (self.xNew, torch.from_numpy(x).unsqueeze_(0).float()), 0
            )
        self.yNew = torch.cat(
                (self.yNew, torch.from_numpy(y).unsqueeze_(0).float()), 0
            )


    def add_init(self, x, y):
        '''
        Add to the set of training {x,y} using GPU
        - x : b (random matrix)
        - y: x (solution)
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # GPU
        self.x = torch.cat(
                (self.x, torch.from_numpy(x).unsqueeze_(0).float().to(device)),
                0
            )
        self.y = torch.cat(
                (self.y, torch.from_numpy(y).unsqueeze_(0).float().to(device)),
                0
            )

    
    def predict(self, x):
        '''
        Calls the forward function of the NN
        '''
        # inputs need to be [[x_1, x_2, ...]] as floats
        # outputs need to be numpy (non-grad => detach)
        # outputs need to be [y_1, y_2, ...]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # GPU
        a1 = torch.from_numpy(x).unsqueeze_(0).float().to(device)
        a2 = np.squeeze(
                self.model.forward(
                    a1, self.x.size(0), self.Factor
                ).detach().cpu().numpy()
            )
        return a2



class ArgsView(object):
    """
    A view class that gives access to input arguments of the trained function
    """

    def __init__(self, spec, args, kwargs):
        """
        Constructor
        - spec: names of the args
        - args: values of the args
        - kwargs: values of the additional key args
        """
        self.__spec = spec
        self.__args = args
        self.__kwargs = kwargs

        # set the given args (e.g. self.A = [[1,2],[2,1]])
        for i, arg_name in enumerate(self.__spec.args):
            setattr(
                self,
                arg_name,
                self.__arg_view(i, arg_name)
            )

        # set the additional args with no name in _vargs
        self._vargs = []
        for j in range(i+1, len(self.__args)):
            self._vargs.append(
                self.__varg_view(j)
            )

        # set the additional key args (e.g. debug="True")
        for arg_name in self.__spec.kwonlyargs:
            setattr(
                self,
                arg_name,
                self.__kwoarg_view(arg_name)
            )


    @property
    def _spec(self):
        return self.__spec


    def _replace(self, name, data):
        '''
        Set the arg with name "name" and value "data"
        '''
        setattr(self, name, lambda: data)


    def __arg_view(self, arg_idx, arg_name):
        # return the value of the arg with the arg_idx or the arg_name

        if arg_idx < len(self.__args): 
            return lambda: self.__args[arg_idx] # if the arg sought isnt an additional arg
        if arg_name in self.__kwargs:
            return lambda: self.__kwargs[arg_name] # if the arg sought is an additional key arg

        defaults_offset = len(self.__spec.args) - len(self.__spec.defaults) # else the arg is in defaults
        return lambda: self.__spec.defaults[arg_idx - defaults_offset]


    def __kwoarg_view(self, arg_name):
        # return the value of the additional key argument with the name arg_name

        if arg_name in self.__kwargs: # if the key arg sought is in kwargs 
            return lambda: self.__kwargs[arg_name]

        return lambda: self.__spec.kwonlydefaults[arg_name] # else the arg is in kwonlydefaults


    def __varg_view(self, idx):
        # return the value of the non-named arg with the idx
        return lambda: self.__args[idx]


    def __str__(self):
        return str(self.__dict__.keys())


    def __repr__(self):
        return repr(self.__dict__.keys())



class PreconditionerTrainer(object):
    """
    Class of the trainer
    """

    def __init__(
            self, preconditioner, linop_name="A", prob_rhs_name="b",
            prob_lhs_name="x", prob_init_name="x0", prob_tolerance_name="e",
            retrain_freq=1, debug=False, Initial_set=32, diagnostic_probe=1
        ):

        '''
        Constructor
        - preconditioner: the NN the trainer will use
        - linop_name: name of the arg of the laplace operator (or the matrix A)
        - prob_rhs_name: name of the arg of the vector b
        - prob_lhs_name: name of the arg of the unknown variable x
        - prob_init_name: name of the arg that represents the initial condition x_0 of the problem
        - prob_tolerance_name: name of the arg of the relative tolerance e
        - retrain_freq: how many new vectors (fulfilling certain conditions) you need to train again
        - debug: flag to the debug mode
        - Initial_set: number of problems it needs to train
        - diagnostic_probe: get the information each diagnostic probe
        '''

        self.preconditioner   = preconditioner
        self.retrain_freq     = retrain_freq
        self.debug            = debug
        self.Initial_set      = Initial_set
        self.diagnostic_probe = diagnostic_probe

        # Describe how we get specific arguments out of the input args
        self.linop_name          = linop_name
        self.prob_rhs_name       = prob_rhs_name
        self.prob_lhs_name       = prob_lhs_name
        self.prob_init_name      = prob_init_name
        self.prob_tolerance_name = prob_tolerance_name

        # Additional args
        self.ML_GMRES_Time_list = list() # List of the runtimes only in the GMRES algorythm part
        self.ProbCount          = 0 # Number of problems 
        self.prob_debug         = False, # 
        self.blist              = list() # list of the vectors b to use in the trainer
        self.reslist            = list() # list of the results
        self.Err_list           = list() # list of the numeric errors in the diagnostic_probe iterations
        self.reslist_flat       = list() # list of the results in one dimension
        self.IterErrList        = list() # list of the numeric errors of every iteration
        self.trainTime          = list() # list of the time in the trained section
        self.trainInfo          = list() # list of the problems trained with its runtime


    def set_args_view(self, spec, args, kwargs):
        '''
        Construct view of a function
        - spec: name of the args of a function
        - args: values of the args of the function
        - kwargs: values of the additional key args
        '''
        self.args_view = ArgsView(spec, args, kwargs)


    def get_problem_data(self):
        # Get the args view into A, b, x0, and e
        A  = getattr(self.args_view, self.linop_name)
        b  = getattr(self.args_view, self.prob_rhs_name)
        x0 = getattr(self.args_view, self.prob_init_name)
        e  = getattr(self.args_view, self.prob_tolerance_name)

        return A(), b(), x0(), e()
         
    # Cambiado **
    def emptyTrainTime(self):
        self.trainTime=[]

    @staticmethod
    def fill_args(arg_view):
        '''
        Return the args and key args of the function that uses the trainer
        '''
        args = []

        # Add every arg
        for arg in arg_view._spec.args:
            args.append(
                getattr(arg_view, arg)()
            )

        # Add every variable arg
        for argv in arg_view._vargs:
            args.append(
                argv()
            )

        # Add every key arg
        kwargs = dict()
        for kw in arg_view._spec.kwonlyargs:
            kwargs[kw] = getattr(arg_view, kw)()

        return args, kwargs


    def predict(self, A, b, x0, b_scale):
        '''
        Auxiliar function to call the prediction of the NN if it has trained 
        - A: funtion of the laplacian operator
        - b: random matrix with a dipole
        - x0: matrix of the initial condition
        - b_scale: maximum of b
        '''

        # if NN trained, predict x0 and reescalate
        if self.preconditioner.is_trained:
            pred_x0 = self.preconditioner.predict(b/b_scale) 
            pred_x0 = pred_x0 * b_scale 
            # target_test=GMRES(A, b, x0, e, 6,1, True)
            # IterErr_test = resid(A, target_test, b)
            # print('size',len(IterErr_test))
            # print(IterErr_test[5],max(self.Err_list))
            # if (IterErr_test[5]>1.75*max(self.Err_list)):
            #     print('poor prediction,using initial x0')
            # pred_x0 = x0

        # else keep x0
        else:
            pred_x0 = x0

        return pred_x0

    @timer
    def add_single(self, res, b, scale):
        '''
        Function to add a new set to the trainer set if it fulfills some conditions
        - res: solution of the GMRES (matrix dim x dim)
        - b: random matrix dim x dim with a dipole
        - scale: scale of b (non normalized maximum of b)
        '''
        # Rescale RHS so that network is trained on normalized data
        b   = b   / scale
        res = res / scale

        if self.ProbCount <= self.Initial_set:
            self.preconditioner.add_init(b, res) # if number of sets < initial_set, add directly
        if self.ProbCount == self.Initial_set: 
            timeLoop = self.preconditioner.retrain_timed() # if number of sets = initial_set, train and save runtime
            self.trainTime.append(timeLoop[-1])
            self.trainInfo.append((self.ProbCount, timeLoop[-1])) # Cambiado **

        # Compute moving averages used to filter data
        if self.ProbCount > self.Initial_set:
            IterTime_AVG = moving_average(
                    np.asarray(self.ML_GMRES_Time_list),
                    self.ProbCount
                )
            IterErr10_AVG = moving_average(
                    np.asarray(self.Err_list),
                    self.ProbCount
                )

        # Filter for data to be added to training set
        if self.ProbCount > self.Initial_set: # if enough problems in the training set
            # First filter: slower than avg and more error than avg
            if self.ML_GMRES_Time_list[-1] > IterTime_AVG \
            and self.Err_list[-1] > IterErr10_AVG: 

                # Second filter: use the 50% of the sets {b,x}
                CoinToss = np.random.rand()
                if (CoinToss < 0.20):
                    self.blist.append(b)
                    self.reslist.append(res)
                    self.reslist_flat.append(
                            np.reshape(res,(1,-1), order='C').squeeze(0)
                        )

                # Third filter: when there are 3 different sets {b,x} check if they are ortogonally enough
                if len(self.blist) == 3:
                    resMat        = np.asarray(self.reslist_flat)
                    resMat_square = resMat**2
                    row_sums      = resMat_square.sum(axis=1, keepdims=True)
                    resMat        = resMat/np.sqrt(row_sums) # normalize the flat solution
                    InnerProd     = np.dot(resMat, resMat.T)

                    #TODO: Do we need np.asarray here?

                    # add the first one always
                    self.preconditioner.add(
                            np.asarray(self.blist)[0],
                            np.asarray(self.reslist)[0]
                        )

                    cutoff=0.8 # define the cutoff of how orthogonal the 3 solutions should be
                    
                    
                    if np.abs(InnerProd[0,1]) < cutoff \
                    and np.abs(InnerProd[0,2]) < cutoff:
                        if np.abs(InnerProd[1,2]) < cutoff: # if every permutation is orthogonal, add the 3

                            #TODO: Do we need np.asarray here?
                            self.preconditioner.add(
                                    np.asarray(self.blist)[1],
                                    np.asarray(self.reslist)[1]
                                )

                            #TODO: Do we need np.asarray here?
                            self.preconditioner.add(
                                    np.asarray(self.blist)[2],
                                    np.asarray(self.reslist)[2]
                                )

                        elif np.abs(InnerProd[1,2]) >= cutoff: # if 0 ort 1 and 0 ort 2 but 1 not ort 2, add 1
                            #TODO: Do we need np.asarray here?
                            self.preconditioner.add(
                                    np.asarray(self.blist)[1],
                                    np.asarray(self.reslist)[1]
                                )

                    elif np.abs(InnerProd[0,1]) < cutoff : # elif 0 ort 1 add 1
                        #TODO: Do we need np.asarray here?
                        self.preconditioner.add(
                                np.asarray(self.blist)[1],
                                np.asarray(self.reslist)[1]
                            )

                    elif np.abs(InnerProd[0,2]) < cutoff : # elif 0 ort 2 add 2
                        #TODO: Do we need np.asarray here?
                        self.preconditioner.add(
                                np.asarray(self.blist)[2],
                                np.asarray(self.reslist)[2]
                            )

                    # Train if enough data has been collected, if retrain_freq=1, it will always enter
                    if self.preconditioner.counter >= self.retrain_freq:
                        # if self.debug:
                        #     print("retraining")
                        #     print(self.preconditioner.counter)
                        timeLoop = self.preconditioner.retrain_timed() # Train the NN and return runtime
                        # trainTime=float(timeLoop[-1])
                        # TODO: we need a data retention policy for things
                        # like the train time history

                        # save training time and reset auxiliar lists 
                        self.trainTime.append(timeLoop[-1]) 
                        self.trainInfo.append((self.ProbCount, timeLoop[-1])) # Cambiado **
                        self.blist        = []
                        self.reslist      = []
                        self.reslist_flat = []

    def write_diagnostics(self, iter_time, A, target, b):
        '''
        Function to update the information to the lists of the trainer
        - iter_time: runtime of the GMRES
        - A: Laplace operator
        - target: solution of the GMRES (matrix dim x dim)
        - b: random matrix dim x dim with a dipole
        '''
        normb = np.linalg.norm(b)
        # Cambiado **
        if self.debug:
            iter_err = resid(A, target, b) # calculate the residue of the problem
            self.IterErrList.append(iter_err / normb) 
        iter_err_probe = resid(A, np.asarray([target[-1]]), b) / normb# select the residue of the diagnostic_probe matrix
        self.ML_GMRES_Time_list.append(iter_time) 
        self.Err_list.append(iter_err_probe)



def cnn_preconditionerOnline_timed_2D(trainer):

    def my_decorator(func):
        spec = getfullargspec(func) # Save func args names
        name = func.__name__

        @wraps(func)
        def speedup_wrapper(*args, **kwargs):

            # Construct view of the func with its args (in this case, A, b, x0,...)
            trainer.set_args_view(spec, args, kwargs) 

            # Get problem data:
            A, b, x0, e = trainer.get_problem_data()

            # Use NN to generate initial guess:
            b_norm, b_Norm_max = prob_norm(b) # norm of b and maximum of normalized b
            pred_x0            = trainer.predict(A, b, x0, b_norm*b_Norm_max) # b_norm*b_Norm_max = not-normalized maximum

            # Replace the input initial guess witht the NN preconditioner
            args_view = deepcopy(trainer.args_view) # Deepcopy of the args of func
            args_view._replace(trainer.prob_init_name, pred_x0) # Replace initial guess to the predicted one
            new_args, new_kwargs = PreconditionerTrainer.fill_args(args_view) # Update args after replace them

            # Run function (and time it)
            tic = time.perf_counter()
            target = func(*new_args, **new_kwargs) # GMRES
            toc = time.perf_counter()

            # Pick out the last solution from residual list
            res = target[-1]

            # Write diagnostic data (error and time-to solution) to list
            IterTime = (toc-tic)
            trainer.write_diagnostics(IterTime, A, target, b) # Update the lists to print information

            # Add problem to the training set (if it is a valid set)
            run_time = trainer.add_single(res, b, b_norm*b_Norm_max)

            return target

        speedup_wrapper.__signature__ = signature(func) # preserve the original function signature

        return speedup_wrapper

    return my_decorator





##########################33
# Possibly useful snipets

# #original implementation of tensor linop A (prohibitively slow for back prop)
# self.ATesnorOp = mk_laplace_2d_Tensor(10, 10, dx)
# y_pred= self.ATesnorOp(y_pred)
# loss = self.criterion(y_pred-batch_x,0*y_pred)

 ## FD convolutional weights for computing residual
# self.FDpad=torch.nn.ZeroPad2d(1).to(device)
# self.Aweights = torch.tensor([[0., 1., 0.],
#                 [1., -4., 1.],
#                 [0.,  1., 0.]])*(1/dx)**2.0
# self.Aweights = self.Aweights.to(device)
# self.Aweights = self.Aweights.view(1,1,3 ,3 )


# #For restoring original scale of  solutions and RHS
# y_pred=torch.mul(y_pred,batch_Normfactors)
# batch_y =torch.mul(batch_y,batch_Normfactors)
# batch_x =torch.mul(batch_x,batch_Normfactors)
# ResidualLoss = torch.mul(ResidualLoss,batch_Normfactors)

# #Code for writing number of samples to file
# f2=open("NumSamples.txt","ab")
# Temp=np.zeros((1,1))
# Temp[0,0]=self.x.size(0)
# np.savetxt(f2,Temp)
# f2.close()


## Code for writing loss values to files
# f=open("Losses.txt","ab")
# # print(np.asarray(self.loss_val[1:-1]))
# np.savetxt(f,np.asarray(self.loss_val[1:-1]))
# f.close()



# Alternate implementation of loss function
# ResidualLoss = torch.square(ResidualLoss)
# ResidualLoss = torch.sum(ResidualLoss,-1)
# ResidualLoss = torch.sum(ResidualLoss,-1)
# ResidualLoss = torch.sqrt(ResidualLoss)
# ResidualLoss = torch.sum(ResidualLoss)
# ResidualLoss = torch.sqrt(0.0001*ResidualLoss/y_pred.size(0))


# L2Integralloss = torch.square(y_pred-batch_y)
# L2Integralloss = torch.sum(L2Integralloss,-1)
# L2Integralloss = torch.sum(L2Integralloss,-1)
# L2Integralloss = torch.sqrt(L2Integralloss*(self.dx**2.0/self.Area))
# L2Integralloss = torch.sum(L2Integralloss)
# L2Integralloss =  torch.sqrt(L2Integralloss/y_pred.size(0))


# Faster implentation of loss
# ResidualLoss=torch.nn.functional.conv2d(self.FDpad(y_pred.unsqueeze(1)), self.Aweights, bias=None, stride=1)
# ResidualLoss = ResidualLoss.squeeze(1)
# ResidualLoss = ResidualLoss - batch_x
# ResidualLoss= (0.0001*self.criterion(ResidualLoss, 0.0*ResidualLoss))

# loss= torch.sqrt(L2Integralloss+ResidualLoss)



## snippets for restoring  RHS scale during training
# barray=np.ones((InputDim,InputDim))
# barray=barray*b_norm
# bnormList.append(barray)
