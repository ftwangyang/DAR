# DAR(Distance-aware Attention Reshaping)

## Distance-aware Attention Reshaping for Enhancing Generalization of Neural Solvers


## Test DAR on CARP

Under the DAR/CARP/POMO-GE-DAR folder, run

```bash
python test.py
```

## Test DAR on VRPTW

Under the DAR/VRPTW folder, run

```bash
python test.py
```

## Test DAR on ATSP

Under the DAR/ATSP/DAR-MatNet folder, run

```bash
python test_DAR-matnet.py
```
## Test DAR on VRPLIB

Under the DAR/CVRP folder, use the default settings in *config.yml*, run

```bash
python test_vrplib_time.py
```

## Test DAR on TSPLIB

Under the DAR/TSP folder, use the default settings in *config.yml*, and run

```bash
python test_tsplib.py
```

## Test DAR on KP

Under the DAR/KP folder, run

```bash
python Inference.py
```

## Train DAR on TSP, CVRP, VRPTW, ATSP, KP, CARP


```bash
python train.py
```

## Acknowledgments

* https://github.com/gaocrr/ELG
* https://github.com/yd-kwon/MatNet
* https://github.com/yd-kwon/POMO
* https://github.com/RoyalSkye/Omni-VRP
* https://github.com/RoyalSkye/Routing-MVMoE
