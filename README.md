# PINNs for Damped Harmonic Oscillator

The forced harmonic oscillator is an important physical model that describes how a classical harmonic oscillator behaves under the influence of an external force. The classical harmonic oscillator acts according to Hooke's law and is subject to a constant pulling force. However, in a forced harmonic oscillator, it is possible to study the dynamics that occur when an external force is applied rhythmically to the natural frequency of the system. This system provides an important model for understanding the complexity of vibrations and how they can resonate under the influence of external force. The forced harmonic oscillator is used in many fields, from mechanical systems to electrical circuits, and understanding this model is a fundamental step to explore physical phenomena in more depth and use it in various applications.In order to use PINNs in practice, programming of the forced harmonic oscillator was studied.   

The movement of the damped harmonic oscillator can be visualized as shown in the figure;
![oscillator](https://github.com/saglamzeynep/Harmonic-Oscillator/assets/152716329/aebb5ed9-5c8f-4380-95bd-f9d6b9836bea)  

In this example, we are interested in modeling the displacement u(t) of the mass (green box) over time due to the spring. This is a canonical physics problem in which displacement occurs. Thus the differential equation as a function of oscillator time is defined as;  

$$m\frac{d^2u}{dt^2}+\mu\frac{du}{dt}+ku=0$$


where m is the mass of the oscillator, $\mu$ is the coefficient of friction and k is the spring constant. Since the forced harmonic oscillation situation will be dealt with, it is taken into consideration that the oscillation will stop slowly. Mathematically, this situation is represented by this equation;  

$$\delta > w_0 \quad where, \quad\quad \delta=\frac{\mu}{2m} \quad\quad w_0=\sqrt{\frac{k}{m}}$$

The initial conditions of the system were determined as follows:  

$$u(t=0)=1 \quad \quad and \quad\quad \frac{du}{dt}(t=0)=0$$

Finally, it is necessary to create an  ansatz equation for our application. Ansatz usually refers to an educated guess or presupposition made for the purpose of simplifying a complex problem or developing a more understandable solution or equation. In theoretical physics, it is used as an initial assumption or hypothesis to simplify the equations of complex physical systems. These assumptions are chosen for mathematical convenience or for the purpose of approximating a complex truth to a simpler one, making the result more processable.Here our ansatz equation ;  

$$u(t)=e^{-\delta t}(2Acos(\phi+wt)) \quad\quad, with \quad\quad w=\sqrt{w_0^2+\delta^2}$$

## Simulation of Equation of Motion with PINNs for Harmonic Oscillator

The first goal was to simulate the equation of motion using time data and equation 2.13. Python's pytorch library was used to do this. In addition to the time data produced using the linspace function and equation below, a loss function was created;  

![Opera Snapshot_2023-12-13_141748_github com](https://github.com/saglamzeynep/Harmonic-Oscillator/assets/152716329/57b9d45e-6fbc-44d5-b11a-ea9f35f308e6)

The first two terms in the loss function represent the boundary loss, and tries to ensure that the solution learned by the PINN matches the initial conditions of the system. The third term in the loss function is called the physics loss, and tries to ensure that the PINN solution obeys the underlying differential equation at a set of training points $(t_i)$ sampled over the entire domain. The $\lambda_1$ and $\lambda_2$ values used here were found by trial and error. The purpose of using lambda values is to ensure stability during training.

## Simulation of Equation of Motion with PINNs for Noisy Harmonic Oscillator Data

In this part, the aim was to develop an algorithm that estimates the $\delta$ value by simulating the system based on existing noisy data. We established a similar system with changes to the loss function we used in the previous application. Here loss function;  

![denk2](https://github.com/saglamzeynep/Harmonic-Oscillator/assets/152716329/18de574d-7da4-4597-bdee-756fe2c8d416)  

The main idea here is to treat $\mu$ as a learnable parameter when training PINN. So we both simulate the solution and invert this parameter. There are two terms in the loss function here. The first is the physics loss, which occurs in the same way as above, which ensures that the solution learned by PINN is consistent with known physics. The second term is called data loss and ensures that the solution learned by PINN is appropriate. The reason why there is no boundary loss at this point is that we do not know the boundary conditions.  

## Simulation of Equation of Motion with PINNs for High Frequency Harmonic Oscillator

In the last application, we tried to examine the operation performed in the first application for higher frequencies. When we used exactly the same approach and the same code and increased the frequency value from 20 to 80, the program did not react at all and returned a straight line output. The problem here is the spectral bias. The term "spectral bias" refers to a specific tendency or preference for certain frequencies in the learning and generalization processes of neural networks. Spectral bias of neural networks and the need for more training points makes this a  difficult problem for the PINN to deal with.  
To solve the spectral bias issue, we had to go back to our ansatz equation and make changes. At this point we added the term $sin(\alpha t + \beta )$ to ansatz as a multiplier. The reason for this intervention is our ability to "assumption" by having knowledge of physics. As a result, the new ansatz equation

$$u(t)=e^{-\delta t}(2Acos(\phi+wt))sin(\alpha t+\beta) \:\:, with \:\:\:\: w=\sqrt{w_0^2+\delta^2}$$

After understanding the system and placing it on a physical basis, we train our machine with our approaches and ansatz and achieve the results we want.









