# Geometric data analysis (GDA)-based machine learning for 2D perovskite design

Implementation of the paper "Geometric data analysis (GDA)-based machine learning for 2D perovskite design" by Chuan-Shen Hu, Min-Chun Wu, Kelin Xia, and Tze Chien Sum

![image](https://github.com/peterbillhu/DFOn2DProveskites/assets/28446650/6bf0f532-f02d-4c3e-834a-27ff44e75fca)

## Requirments

numpy >= 1.21.2

## File Descriptions

The Density Fingerprint algorithm was initially introduced by Edelsbrunner et al. [1]. In this study, we present an implemented algorithm designed for efficiently calculating the density fingerprint of a given atomic system, comprising both the unit cell and motif set. All codes are in the _Algorithms_ folder:


## Tutorial

We utilize the provided codes to import the proposed algorithms and functions required for generating the density fingerprint.

```
from Algorithms.fast_density_fingerprint import find_relevant_pts, density_fingerprint, plot_density
```

To compute the density fingerprint, the unit cell and motif of a material are required. Here is a toy example:

```
Motif = np.array([[5,5,5]])
unit_cell = np.array([[10,0,0], [0,10,0], [0,0,10]])
```

In this example, the rows of the unit_cell matrix represent the basis of the unit cell and the Motif records the unique central point in the unit cell. We note that the coordinates of motif points are absolute coordinates (not fractional coordinates).


## Google Notebook Tutorial

To ensure the function of the proposed algorithms, we provide a Google Notebook tutorial to demonstrate how to generate the density fingerprint of a given unit cell and motif set. In the demonstration, we exhibit toy examples as shown in the above section. 

https://colab.research.google.com/drive/1vJKg6GXGYTCD2WQ33GOtdIGE7zOawEos?usp=sharing

## Code References

[1] Edelsbrunner, Herbert, et al. "The density fingerprint of a periodic point set." arXiv preprint arXiv:2104.11046 (2021).
