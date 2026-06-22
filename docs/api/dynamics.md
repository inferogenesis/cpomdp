# Process noise

State-dependent dynamics noise `Q(x)`: the world can diffuse more in some states than others. `CallableProcessNoise` carries the `Q(x)` function; both it and a plain constant satisfy the `DynamicsNoise` protocol.

::: cpomdp.DynamicsNoise

::: cpomdp.CallableProcessNoise
