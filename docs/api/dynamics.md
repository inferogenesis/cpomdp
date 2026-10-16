# Dynamics noise

State-dependent dynamics noise `Q(x)`: the world can diffuse more in some states than others. A constant `Q` needs no object. It goes in the model's `dynamics_noise`. `DynamicsNoiseModel` is the protocol for a `Q(x)` in `dynamics_noise_model`, the twin of `ObservationModel` on the sensor side, and a plain matrix does not satisfy it. `CallableDynamicsNoise` carries the `Q(x)` function, with the same two arguments as `CallableSensor`.

::: cpomdp.DynamicsNoiseModel

::: cpomdp.CallableDynamicsNoise
