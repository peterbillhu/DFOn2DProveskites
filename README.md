# Geometric data analysis (GDA)-based machine learning for 2D perovskite design

Implementation of the paper "Geometric data analysis (GDA)-based machine learning for 2D perovskite design" by Chuan-Shen Hu, Rishikanta Mayengbam, Min-Chun Wu, Kelin Xia, and Tze Chien Sum

![image](https://github.com/peterbillhu/DFOn2DProveskites/assets/28446650/6bf0f532-f02d-4c3e-834a-27ff44e75fca)

## Requirments

numpy >= 1.21.2

## File Descriptions

The Density Fingerprint algorithm was initially introduced by Edelsbrunner et al. [1]. In this study, we present an implemented algorithm designed for efficiently calculating the density fingerprint of a given atomic system, comprising both the unit cell and motif set. All codes are in the _Algorithms_ folder:

1. fast_density_fingerprint.py: the kernel code file of the project, where the functions _find_relevant_pts_ and _density_fingerprint_ are sufficient to compute the density fingerprint for a given system of the unit cell and motif set.

2. plotting_density_fingerprint.py (Optional): a code file for plotting the density fingerprint. Main function: _plot_density_.

3. unit_cell_tool.py (Optional): a code file for generating unit cell vectors. If one already has the unit cell basis, then this code file is unnecessary for computing the density fingerprint.

## Tutorial

### Density Fingerprint Generation 

We utilize the provided codes to import the proposed algorithms and functions required for generating the density fingerprint.

To compute the density fingerprint, the unit cell and motif of a material are required. Here is a toy example:

```python
Motif = np.array([[5,5,5]])
unit_cell = np.array([[10,0,0], [0,10,0], [0,0,10]])
```
In this example, the rows of the **unit_cell** matrix represent the basis of the unit cell and the **Motif** records the unique central point in the unit cell. We note that the coordinates of motif points are absolute coordinates (not fractional coordinates).

Next, we compute the density fingerprints. First, we need to set parameters for the computation.
```python
## Step 2: compute the density fingerprint:
# Density fingerprint parameters:
# compute \psi_k, k = 0, 1, 2, ..., k_up
k_up = 9
# tolerable density value error
eps = 0.0001
# range of the x-axis of the curves
range_t = (0,1.2)
# fineness of the x-axis of the curves
fine_t = 100
````
Interpretations for the parameters assigned above are as follows:

1. **k_up** is the largest index of discrete density functions. For example, if **k_up** = 9, then the output density functions are $`\psi_0, \psi_1, \dots, \psi_9`$;
2. **eps** is the bias tolerance of density functions Reducing the value of **eps** will result in smoother density functions.
3. **range_t** denotes the domain of the density functions $`\psi_\bullet`$;
4. **fine_t** is the fineness of the x-axis of the density functions $`\psi_\bullet`$. It divides the domain range into **fine_t** grid points and outputs each density functions $`\psi_i`$ as a discrete feature in dimension **fine_t**.

After obtaining the parameters, one can compute the density fingerprint via the following code:

```python
# Compute the density fingerprint
pts_k_up, dists_min_k_up = find_relevant_pts(unit_cell, Motif, k_up)
psi_dict = density_fingerprint(pts_k_up, k_up, unit_cell, range_t, fine_t, eps)  ## psi_dict is a collection of density fingerprint codes
```
**psi_dict** is a collection of NumPy arrays corresponding to the density functions $`\psi_\bullet`$.

### Plot the Density Fingerprint

Let **psi_dict** be the output as above ($`M = \{ (5,5,5) \}`$ and $`U`$ the unit cell with basis $\{ (10,0,0), (0,10,0), (0,0,10) \}$). We can use the following codes to plot the density functions $`\psi_\bullet`$.

First, we import the plotting tool:

```python
from Algorithms.plotting_density_fingerprint import plot_density
```

Second, we use the following code to plot the density functions $`\psi_\bullet`$. In this example, the functions are $`\psi_0, \psi_1, \dots, \psi_9`$.

![image](https://github.com/peterbillhu/DFOn2DProveskites/assets/28446650/cae6197f-4680-4e8a-b0ff-c532ce41552a)


### The Unit Cell Tool

Some material data may only provide the unit cell's _scales_ (a, b, c) and _angles_ $`(\alpha, \beta, \gamma)`$.

```python
# Import the get_unit_cell function
from Algorithms.unit_cell_tool import get_unit_cell
get_unit_cell(scales=np.array([10,10,10]), angles=np.array([90,90,90]))
```

The output Numpy array is as follows:
> array([[1.000000e+01, 0.000000e+00, 0.000000e+00],
       [6.123234e-16, 1.000000e+01, 0.000000e+00],
       [6.123234e-16, 6.123234e-16, 1.000000e+01]])

## Google Colab Notebook Tutorial

To present the function of the proposed algorithms, we provide a Google Colab Notebook tutorial to demonstrate how to generate the density fingerprint of a given unit cell and a motif set. In the demonstration, we exhibit and compute the toy examples as shown in the above section. 

https://colab.research.google.com/drive/1vJKg6GXGYTCD2WQ33GOtdIGE7zOawEos?usp=sharing

For the Google Drive location of the notebook, refer to: 

https://drive.google.com/drive/folders/1TmKzAdcf7fKtqzP3QMLm99QkASWAxR_T?usp=sharing

## Code References

[1] Edelsbrunner, Herbert, et al. "The density fingerprint of a periodic point set." arXiv preprint arXiv:2104.11046 (2021).
