# Multivariate Regression with Gross Errors on Manifold-valued Data
## [Xiaowei Zhang](https://web.bii.a-star.edu.sg/~zhangxw/), [Li Cheng](https://web.bii.a-star.edu.sg/~chengli/), [Xudong Shi](https://github.com/shixudongleo), Yu Sun

This is a collection of MATLAB codes that implement a new algorithm for Multivariate Regression on manifolds proposed in the paper. We consider regression problems where given a multivariate observation $\bs{x}\in\bbR^{d}$, the output response *\bs{y}* lies on a Riemannian manifold *\mathcal{M}*. We propose a new regression model to deal with the presence of grossly corrupted manifold-valued responses. 

Full paper here: [http://arxiv.org/abs/](http://arxiv.org/abs/)

We also have a project website [here](https://web.bii.a-star.edu.sg/~zhangxw/palmr-SPD).


## Folders
* `palmr/spd/`: contains implementation of our algorithm *PALMR* for multivariate regression on the manifold of Symmetric Positive Definite (SPD) matrices. It consists of the following subfolders:
  * `src/`: all source codes implementing the algorithm.
  * `results/`: contains results obtained by the authors. These results are used for reproducing tables and figures in the manuscript. Slightly different results may be obtained due to randomness in our codes.
  * `test/`: 
    * `synthetic/`: contains codes to produce results on the synthetic data set.
    * `real_DTI/`:  contains codes to produce results on the real DTI data set.
    
* `riem_mglm_0.12/`: contains implementations of *MGLM* written by [Kim et al.](http://pages.cs.wisc.edu/~hwkim/projects/riem-mglm/). 


## Usage of codes
The main function of our algorithm is *MGLM_Gross_spd.m* contained in `src/` folder.
 
To reproduce results in the manuscript:
 * For synthetic data, we can simply run *SPD_Figure_Example.m* to produce visual results in the manuscript, and run *SPD_nbasis.m*, *SPD_nsample.m*, *SPD_noise_g.m* and *SPD_rate_g.m* to produce results in each column of Fig. 4 in the manuscript.
 
 * For real DTI data, we need to download the real data C-MIND dataset from the [project website](https://web.bii.a-star.edu.sg/~zhangxw/palmr-SPD), and extract the data to folder `palmr/spd/`. Since we used six slices of whole brain DTI data, we use slice 1 as an example. We first run *Slice1_NoError.m*, *Slice1_ManuError.m* and *Slice1_RegError.m* to get the results, and then run *Slice1_Plot_Pred_Error.m* to plot figures and print results. Once we finished the experiments on all 6 slices, we can run *BoxPlotAllSlices.m* to plot the box plots in Fig. 7 of the manuscript.   

## Acknowledgement
* We include the implementation of MGLM for the sake of easy comparison. The implementation is from the authors of MGLM. 