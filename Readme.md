# UWB_TDOA_V1.0

**UWB_TDOA_V1.0** is a Python-based simulation framework for indoor positioning using Ultra-Wideband (UWB) Time Difference of Arrival (TDOA) algorithms. It supports configurable 2D/3D environments, interference simulation (NLOS, multipath, blockage), pre-processing strategies, and multiple localization models including Least Squares and Particle Filters.

---

## 📁 Project Structure
```
UWB_TDOA_V1.0/
├── main.py                   # Main simulation entry point
├── config.py                 # Global configuration parser and defaults
├── datagenerator.py          # TDOA measurement generation with noise/interference
├── dataprocess.py            # Interference-aware preprocessing logic
├── configs/
│   └── model.yaml            # Model configuration file (used for dynamic model loading)
├── model/
│   ├── __init__.py           # Model loader based on YAML mapping
│   ├── least_squares.py      # Least Squares TDOA estimation
│   ├── particle_filter.py    # Particle Filter estimation (optional)
│   ├── fang.py               # Fang algorithm (optional)
│   ├── chan.py               # Chan algorithm (optional)
│   └── taylor.py             # Taylor series-based solver (optional)
├── utils.py                  # Utility functions (e.g. distance, RMSE, MAE)
├── evaluation.py             # Evaluation metrics (RMSE, MAE, per-point error)
└── visualization.py          # 2D/3D visualization of estimated vs ground-truth positions
```

---

## ⚙️ Installation
- Works with Python 3.7+ and standard libraries like NumPy, Matplotlib, SciPy.

1. Clone the repository
```
git clone https://github.com/power-operation/UWB_TDOA.git
```

2. Install dependencies
```
matplotlib==3.9.2 
numpy==1.26.4 
PyYAML==6.0.2 
scikit-learn==1.5.2 
scipy==1.13.1
...
```

---

## 🚀 How to Run

```bash
python main.py --cfg configs/model.yaml \
               --model=least_squares \
               --dimension=2 \
               --nlos=true \
               --multipath=false \
               --blockage=false \
               --process=true \
               --trajectory=sinusoid
```

---

### ✅ Example Output:
```
==> Generating data...
==> Preprocessing TDOA data...
==> Estimating positions using least_squares...
==> Evaluating results...
RMSE: 0.482 meters
MAE: 0.379 meters
==> Visualizing...
```

---

## 🔧 Command-Line Arguments
| Argument        | Description                                                                 |
|---------------- |-----------------------------------------------------------------------------|
| `--cfg`         | Path to YAML model configuration (default: `configs/model.yaml`)            |
| `--model`       | Name of positioning model to use (e.g., `least_squares`, `particle_filter`) |
| `--dimension`   | Set spatial dimension: `2` for 2D or `3` for 3D simulation                  |
| `--nlos`        | Whether to simulate NLOS bias (`true` / `false`)                            |
| `--multipath`   | Whether to simulate multipath effects (`true` / `false`)                    |
| `--blockage`    | Whether to simulate random signal drop/blockage (`true` / `false`)          |
| `--process`     | Whether to enable interference-aware preprocessing (`true` / `false`)       |
| `--trajectory`  | Set visualization trajectory: `line`, `circle`, `sinusoid`, `random`        |

---

## 🧪 Supported Models
Defined via `configs/model.yaml`, each model maps to a specific module and function:
```yaml
models:
  least_squares:
    module: model.least_squares
    function: estimate_positions_least_squares
```
You can add new models to the `model/` directory and register them in `model.yaml`.

---

## 🛠️ Customize & Extend
- Modify `datagenerator.py` to define new interference patterns or data generation logic.
- Implement new preprocessing techniques in `dataprocess.py`.
- Add new estimation models (e.g. machine learning-based) in `model/`.
- Use `evaluation.py` to log and save detailed error statistics.

---

## 📊 Visualization
- The tool supports automatic plotting of estimated vs ground-truth positions.
- For 2D, a scatter plot with anchor locations is shown.
- For 3D, a 3D plot with anchors and estimated trajectories is rendered.

---

## 📌 Notes
- Default random seed is fixed for reproducibility (`np.random.seed(42)`).

---

## 📬 Contact
If you have any question, please feel free to contact the authors. Cunyi Yin: cunyiyin1125@gmail.com.
