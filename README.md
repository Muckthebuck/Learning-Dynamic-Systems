# Learning-Dynamic-Systems

Masters Capstone project: Learning of dynamical systems from a finite number of closed loop data points

Controlling complex systems—such as robots, power grids, or autonomous vehicles—requires designing controllers that ensure stability and accurate tracking of desired states. Traditional model-based control methods depend on precise knowledge of the system’s equations, but real-world systems often experience parameter variations and unmodeled dynamics, making exact modeling impractical. This project explores a data-driven approach using the Sign-Perturbed Sums (SPS) algorithm to estimate system dynamics with finite closed-loop data. The SPS method provides a set of possible system models with exact probabilistic guarantees that the true system lies within this set. By continuously refining this set with new data, the approach enables adaptive control strategies that improve robustness and resilience to uncertainties. This work contributes to more reliable and flexible control systems by reducing dependence on prior knowledge while ensuring performance guarantees, making it valuable for applications in robotics, automation, and engineering systems where adaptability is crucial.

The full report: [Report](./E_21_WERI_009.pdf)

---

## How to run

0. Run Docker if not running
1. Start the redis container for pub sub communicatio

    ```bash
    bash redis_pub_sub.sh --start
    ```

2. Start the SPS for the config you want to run

    ```bash
    bash run_sps.sh --config configs/<config_file.yaml> 
    ```

    for example to run for the armax sim config you would do:

    ```bash
    bash run_sps.sh --config configs/armax.yaml 
    ```

3. Start the Sim

    ```bash
    bash run_sims.sh --config configs/<config_file.yaml> 
    ```

    for example to run for the armax sim config you would do:

    ```bash
    bash run_sims.sh --config configs/armax.yaml 
    ```

### Current supported config files

1. armax.yaml
2. armax_2_order.yaml
3. water_tank.yaml

Rest havent been configured properly yet.

---
Note: If you are not able to run on windows, make sure you have relvant packages..

To parse the YAML configs we are using mikefarah's 'yq': Install from <https://github.com/mikefarah/yq/tree/v4.45.4>
