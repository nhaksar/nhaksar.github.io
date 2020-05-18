---
layout: post
title:  "Implementing SIR Model in Python"
date:   2020-05-18 17:15:42 -0600
categories: Python
---
Python has a lot of useful stuff for numerically solving dynamic systems. Today, I show how to implement an SIR model in Python, and give a little background on how to code this kind of thing in general.

## Some background
When we talk about the SIR model, we're generally referring to the mathematical model of disease spread first developed by [Kermack et al](doi:10.1098/rspa.1927.0118).
It provides a simple model for the spread of disease through a population where infected people gain immunity after being infected.
The model itself is actually described using a system of differential equations; you can read more about it either in the original paper, or at [this link](https://mathworld.wolfram.com/Kermack-McKendrickModel.html).

## How to code the SIR model
First, we need to import some packages.
```python
# some useful packages
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
```
NumPy is a generally useful package for doing fancy math; Pandas gives access to useful data structures and methods; SciPy is an extension of NumPy that has some specific methods we like, such as 'odeint' and 'fsolve'; and matplotlib is an extremely powerful library for plotting things.

We normalize the size of the population N to 1, which means that S(t), I(t), and R(t) reflect the proportion of people who are susceptible, infected, or "removed" (i.e. recovered) in time t.

Now, we need to create a function that wraps the system of ODEs so we can feed it into 'odeint'.
```python
# define the system of equations that give us our model
# with N normalized to 1, so we have proportions
# params:
#  state = state variables
#  t - time
#  beta - infection rate
#  gamma - recovery rate
# output: next state
def sir_system(state, t, beta, gamma):

    ## draw out state variables
    S, I, R = state

    ## calculate rates for output
    dSdt = -beta*I*S
    dIdt = beta*I*S - gamma*I
    dRdt = gamma*I

    return dSdt, dIdt, dRdt

```
If you didn't read the articles above, this function here gives a very brief summary of the equations that govern the model. You might think of beta as a parameter that determines the rate of infection, while

Having done that, we can make a function that feeds the system, as well as an initial state, a time length, and some parameters, into 'odeint'.
```python
## a function to return the values from integrating the system
def integrate_SIR(init_state, t, beta, gamma):

    ## get times for approximation
    times = np.linspace(0, t, 4*(t+1))

    ## integrate system
    time_states = odeint(sir_system, init_state, times, args=(beta,gamma,))

    return pd.DataFrame({'S(t)': time_states[:,0],
                         'I(t)': time_states[:,1],
                         'R(t)': time_states[:,2]}, index=times)
```

We can also plot the proportions of susceptible, infected, and recovered people over time.
```python
## a function to plot SIR results
def plot_sir(init_state, t, beta, gamma):

    ## get times for approximation
    times = np.linspace(0, t, 4*(t+1))

    ## system integration
    time_states = odeint(sir_system, init_state, times, args=(beta,gamma,))
    S, I, R = time_states.T

    ## plot dynamics
    plt.plot(times, S, 'g', label="S(t)")
    plt.plot(times, I, 'r', label="I(t)")
    plt.plot(times, R, 'b', label="R(t)")

    plt.xlabel("t")
    plt.ylabel("Proportion")
    plt.legend(loc="best")
    plt.show()
```

I give a few examples.
```python
init_state = 0.99, 0.01, 0
integrate_SIR(init_state, 20, 1.2, 1).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.000000</th>
      <th>0.240964</th>
      <th>0.481928</th>
      <th>0.722892</th>
      <th>0.963855</th>
      <th>1.204819</th>
      <th>1.445783</th>
      <th>1.686747</th>
      <th>1.927711</th>
      <th>2.168675</th>
      <th>...</th>
      <th>17.831325</th>
      <th>18.072289</th>
      <th>18.313253</th>
      <th>18.554217</th>
      <th>18.795181</th>
      <th>19.036145</th>
      <th>19.277108</th>
      <th>19.518072</th>
      <th>19.759036</th>
      <th>20.000000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>S(t)</td>
      <td>0.99</td>
      <td>0.987076</td>
      <td>0.984029</td>
      <td>0.980856</td>
      <td>0.977555</td>
      <td>0.974126</td>
      <td>0.970568</td>
      <td>0.966878</td>
      <td>0.963059</td>
      <td>0.959110</td>
      <td>...</td>
      <td>0.685397</td>
      <td>0.683812</td>
      <td>0.682297</td>
      <td>0.680850</td>
      <td>0.679469</td>
      <td>0.678150</td>
      <td>0.676891</td>
      <td>0.675691</td>
      <td>0.674545</td>
      <td>0.673453</td>
    </tr>
    <tr>
      <td>I(t)</td>
      <td>0.01</td>
      <td>0.010459</td>
      <td>0.010930</td>
      <td>0.011411</td>
      <td>0.011903</td>
      <td>0.012404</td>
      <td>0.012913</td>
      <td>0.013428</td>
      <td>0.013949</td>
      <td>0.014474</td>
      <td>...</td>
      <td>0.008181</td>
      <td>0.007836</td>
      <td>0.007503</td>
      <td>0.007181</td>
      <td>0.006870</td>
      <td>0.006570</td>
      <td>0.006280</td>
      <td>0.006001</td>
      <td>0.005733</td>
      <td>0.005475</td>
    </tr>
    <tr>
      <td>R(t)</td>
      <td>0.00</td>
      <td>0.002465</td>
      <td>0.005041</td>
      <td>0.007733</td>
      <td>0.010542</td>
      <td>0.013470</td>
      <td>0.016520</td>
      <td>0.019693</td>
      <td>0.022992</td>
      <td>0.026416</td>
      <td>...</td>
      <td>0.306423</td>
      <td>0.308352</td>
      <td>0.310200</td>
      <td>0.311969</td>
      <td>0.313662</td>
      <td>0.315281</td>
      <td>0.316829</td>
      <td>0.318308</td>
      <td>0.319722</td>
      <td>0.321072</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 84 columns</p>
</div>




```python
init_state = 0.99, 0.01, 0 # 1% of init population infected
plot_sir(init_state, 20, 1.2, 1)
```


![png](https://nhaksar.github.io/assets/2020-05-18/output_6_0.png)



```python
init_state = 0.99, 0.01, 0
integrate_SIR(init_state, 20, 1, 0.1).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.000000</th>
      <th>0.240964</th>
      <th>0.481928</th>
      <th>0.722892</th>
      <th>0.963855</th>
      <th>1.204819</th>
      <th>1.445783</th>
      <th>1.686747</th>
      <th>1.927711</th>
      <th>2.168675</th>
      <th>...</th>
      <th>17.831325</th>
      <th>18.072289</th>
      <th>18.313253</th>
      <th>18.554217</th>
      <th>18.795181</th>
      <th>19.036145</th>
      <th>19.277108</th>
      <th>19.518072</th>
      <th>19.759036</th>
      <th>20.000000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>S(t)</td>
      <td>0.99</td>
      <td>0.987343</td>
      <td>0.984063</td>
      <td>0.980019</td>
      <td>0.975043</td>
      <td>0.968935</td>
      <td>0.961456</td>
      <td>0.952332</td>
      <td>0.941248</td>
      <td>0.927849</td>
      <td>...</td>
      <td>0.000804</td>
      <td>0.000750</td>
      <td>0.000702</td>
      <td>0.000657</td>
      <td>0.000617</td>
      <td>0.000580</td>
      <td>0.000546</td>
      <td>0.000514</td>
      <td>0.000485</td>
      <td>0.000459</td>
    </tr>
    <tr>
      <td>I(t)</td>
      <td>0.01</td>
      <td>0.012388</td>
      <td>0.015336</td>
      <td>0.018968</td>
      <td>0.023434</td>
      <td>0.028915</td>
      <td>0.035618</td>
      <td>0.043789</td>
      <td>0.053703</td>
      <td>0.065668</td>
      <td>...</td>
      <td>0.287550</td>
      <td>0.280757</td>
      <td>0.274120</td>
      <td>0.267638</td>
      <td>0.261306</td>
      <td>0.255121</td>
      <td>0.249081</td>
      <td>0.243182</td>
      <td>0.237421</td>
      <td>0.231794</td>
    </tr>
    <tr>
      <td>R(t)</td>
      <td>0.00</td>
      <td>0.000269</td>
      <td>0.000602</td>
      <td>0.001013</td>
      <td>0.001522</td>
      <td>0.002151</td>
      <td>0.002926</td>
      <td>0.003879</td>
      <td>0.005050</td>
      <td>0.006484</td>
      <td>...</td>
      <td>0.711646</td>
      <td>0.718493</td>
      <td>0.725178</td>
      <td>0.731705</td>
      <td>0.738077</td>
      <td>0.744299</td>
      <td>0.750374</td>
      <td>0.756304</td>
      <td>0.762094</td>
      <td>0.767747</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 84 columns</p>
</div>




```python
init_state = 0.99, 0.01, 0
plot_sir(init_state, 20, 1, 0.1)
```


![png](https://nhaksar.github.io/assets/2020-05-18/output_8_0.png)

I hope that was useful!
