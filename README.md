# Learning of dynamical systems from a finite number of closed loop data points

Model-based control strategies are widely used to regulate dynamical systems, ensuring trajectory tracking and stability. However, these approaches rely on accurate system models, which may not always be available due to unmodeled dynamics, parameter variations, or environmental uncertainties.

This project investigates the application of the Sign-Perturbed Sums (SPS) algorithm to estimate system dynamics using a finite number of closed-loop data points. The SPS method provides a set of system models with exact probabilistic guarantees that the true system lies within this set. By continuously updating this set in real time with new data, we aim to enhance the robustness and adaptability of control strategies, mitigating performance degradation caused by model uncertainty.

We developed a modular software architecture that integrates SPS seamlessly into real-time control loops. The architecture supports asynchronous operation, allowing the computationally intensive SPS process to run on a separate processor or device—ideal for distributed systems or embedded applications with limited resources. This design supports flexible deployment across various platforms and simplifies integration with existing controllers.

By combining real-time SPS-based model updates with robust control synthesis, this project enables adaptive, data-driven closed-loop control. Bayesian methods are employed to fuse SPS confidence regions, refining model estimates over time and improving both safety and performance in dynamic or uncertain environments.

This framework enhances the reliability of autonomous systems under changing conditions and offers a practical approach for deploying advanced system identification techniques in real-world control applications.




### **TLDR**
We estimate how a system behaves using limited data, even when there's noise or uncertainty. Our method gives a guaranteed range of possible models and updates it as more data comes in. This lets us design controllers that stay reliable and safe, even as conditions change.

The full report: [Report](./E_21_WERI_009.pdf)

## Assumptions


1. **The model orders of** $G(z^{-1};\theta)$ **and** $H(z^{-1};\theta)$ **are correctly specified, and the true system lies within the chosen model class. Additionally, the noise model** $H(z^{-1};\theta)$ **has a stable inverse, and the transfer functions satisfy**  $G(0;\theta) = 0$ **and** $H(0;\theta) = 1$ **for all** $\theta \in \Theta$. *This is the strongest assumption, as it requires correct model structure.*

2. **The noise sequence** $\{N_t\}$ **consists of independent random variables, each with a symmetric distribution about zero.**

3. **The noise sequence** $\{N_t\}$ **is independent of the reference signal** $\{R_t\}$.

4. **The subsystems from** $\{N_t\}$ **and** $\{R_t\}$ **to the output** $\{Y_t\}$ **are asymptotically stable and contain no unstable hidden modes.**

5. **The system initialisation is known for all** $t \leq 0$.
6. Fully observerd state matrix:- or an observer is used. 

---

## How to run
Requirements: Docker, Mosek optimiser (Free academic License)

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
Note: If you are not able to run on windows, make sure you have relvant packages..

To parse the YAML configs we are using mikefarah's 'yq': Install from <https://github.com/mikefarah/yq/tree/v4.45.4>

---
### Current supported config files

1. armax.yaml
2. armax_2_order.yaml
3. water_tank.yaml

Rest havent been configured properly yet.

### If connecting your own data generating model/sim
Please expose the following methods and attributes
Attributes:
```
input_limit: List = [[upper_0, lower_0], ... [upper_i, lower_i]] or None if no limits
state: np.ndarray =  full current state
```
Methods:
```
.set_initial_state(state)
.full_state_to_obs_y(state)
.step(u, t) -> returns observed y
.open_loop_input_sequence(T,f,fs,A,c) or some other method to create complex input signal if needed (modify sim_mod_response in sims/sims.py if thats the case)
.get_reference_x(), if sim has its own refernce, see car_sim
```
System architecture at  [Architecture Diag](plant_uml_diagrams/class_diagrams/arch_diag.pdf)
note: it uses pendulum sim as an example as the sim connected. 

