# Randomly Initialized One-Layer Neural Networks Make Data Linearly Separable
Python code to test the separation capacity of randomly initialized neural networks.

## Creating Your Own Datasets
To make a dataset of three concentric hyperspheres in n dimensions, add the following command to `data_maker.py`:
```
make_three_rings_ND(a, b, c, dim, file_name)
```
where `a` is replaced with the number of points in the innermost hypersphere, `b` with the number of points in the middle hypersphere, `c` with the number of points in the outer hypersphere, `dim` with your desired dimension of the dataset, and `file_name` with the file name to save the dataset as.

## Recreating Experiments on Low-Dimensional Data
In `width_experiments.py`, set `name = 20`, `num_trials = 1000`, `width_lb = 1` and `width_ub = 81`. If you want to visualize the 2D dataset, set `SHOW_DATASET = True`. Then, run `width_experiments.py`.

In `lambda_experiments.py`, set `name = 20`, `num_trials = 1000`, `bias_lb = 0`, `bias_ub = 500`, `num_samples = 200`, and `width = 30`. Then, run `lambda_experiments.py`.

## Recreating Experiments on High-Dimensional Data
In `width_experiments.py`, set `name = 21`, `SHOW_DATASET = False`, `num_trials = 1000`, `width_lb = 101` and `width_ub = 151`. Then, run `width_experiments.py`. To visualize the high-dimensional dataset, run `visualize_data.py`.

## Recreating Experiments on Real-World Dataset
In `lfw_experiments.py`, set `num_trials = 100`, `widths = np.arange(10, 401, 10)`, and `file_name = "lfw_results_400.csv"`. Then, run `lfw_experiments.py`. To visualize the results, set `results = np.loadtxt('lfw_results_400.csv')` in `show_lfw_results.py` and run `show_lfw_results.py`.
