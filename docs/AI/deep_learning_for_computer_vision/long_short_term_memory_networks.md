# Long-Short Term Memory (LSTM) Networks

In practice, we actually will rarely ever use the Vanilla RNN formula. Instead, we will use what we call a Long-Short Term Memory (LSTM) RNN.

## Vanilla RNN Gradient Flow and the Vanishing Gradient Problem

An RNN block takes in input $x_t$ and previous hidden representation $h_{t-1}$ and learns a transformation, which is then passed through $\tanh$ to produce the hidden representation $h_t$ for the next time step and output $y_t$ as shown in the equation below.

$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t)$$

The partial derivative of $h_t$ with respect to $h_{t-1}$ is written as:

$$\frac{\partial h_t}{\partial h_{t-1}} = \tanh'(W_{hh}h_{t-1} + W_{xh}x_t)W_{hh}$$

We update the weights $W_{hh}$ by getting the derivative of the loss at the very last time step $L_t$ with respect to $W_{hh}$:

$$\frac{\partial L_t}{\partial W_{hh}} = \frac{\partial L_t}{\partial h_t}\frac{\partial h_t}{\partial h_{t-1}}\ldots\frac{\partial h_1}{\partial W_{hh}} = \frac{\partial L_t}{\partial h_t}\left(\prod_{t=2}^{T}\frac{\partial h_t}{\partial h_{t-1}}\right)\frac{\partial h_1}{\partial W_{hh}} = \frac{\partial L_t}{\partial h_t}\left(\prod_{t=2}^{T}\tanh'(W_{hh}h_{t-1} + W_{xh}x_t)\right)W_{hh}^{T-1}\frac{\partial h_1}{\partial W_{hh}}$$

**Vanishing gradient:** We see that $\tanh'(W_{hh}h_{t-1} + W_{xh}x_t)$ will almost always be less than 1 because $\tanh$ is always between negative one and one. Thus, as $t$ gets larger (i.e., longer timesteps), the gradient $\frac{\partial L_t}{\partial W}$ will decrease in value and get close to zero. This will lead to the vanishing gradient problem, where gradients at future time steps rarely impact gradients at the very first time step. This is problematic when we model long sequences of inputs because the updates will be extremely slow.

**Removing non-linearity ($\tanh$):** If we remove non-linearity ($\tanh$) to solve the vanishing gradient problem, then we will be left with:

$$\frac{\partial L_t}{\partial W} = \frac{\partial L_t}{\partial h_t}\left(\prod_{t=2}^{T}W_{hh}\right)\frac{\partial h_1}{\partial W} = \frac{\partial L_t}{\partial h_t}W_{hh}^{T-1}\frac{\partial h_1}{\partial W}$$

The **singular values** of a matrix tell us how much the matrix can stretch or compress vectors. For a matrix $W_{hh}$, the **largest singular value** (also called the spectral norm) is the maximum factor by which $W_{hh}$ can stretch any vector. Mathematically, if $\sigma_{\max}$ is the largest singular value of $W_{hh}$, then for any vector $\mathbf{v}$, we have $\|W_{hh}\mathbf{v}\| \leq \sigma_{\max}\|\mathbf{v}\|$. When we repeatedly multiply by $W_{hh}$ (as in $W_{hh}^{T-1}$), the effect depends on whether $\sigma_{\max} > 1$, $\sigma_{\max} = 1$, or $\sigma_{\max} < 1$.

If the largest singular value of $W_{hh}$ is greater than 1, then the gradients will blow up and the model will get very large gradients coming back from future time steps. Exploding gradient often leads to getting gradients that are NaNs. If the largest singular value of $W_{hh}$ is smaller than 1, then we will have the vanishing gradient problem as mentioned above.

## LSTM Formulation

The following is the precise formulation for LSTM. On step $t$, there is a hidden state $h_t$ and a cell state $c_t$. Both $h_t$ and $c_t$ are vectors of size $n$. One distinction of LSTM from Vanilla RNN is that LSTM has this additional $c_t$ cell state, and intuitively it can be thought of as $c_t$ stores long-term information. LSTM can read, erase, and write information to and from this $c_t$ cell. The way LSTM alters $c_t$ cell is through three special gates: $i$, $f$, $o$ which correspond to "input", "forget", and "output" gates. The values of these gates vary from closed (0) to open (1). All $i$, $f$, $o$ gates are vectors of size $n$.

At every timestep we have an input vector $x_t$, previous hidden state $h_{t-1}$, previous cell state $c_{t-1}$, and LSTM computes the next hidden state $h_t$ and next cell state $c_t$ at timestep $t$ as follows:

$$f_t = \sigma(W_{hf}h_{t-1} + W_{xf}x_t)$$

$$i_t = \sigma(W_{hi}h_{t-1} + W_{xi}x_t)$$

$$o_t = \sigma(W_{ho}h_{t-1} + W_{xo}x_t)$$

$$g_t = \tanh(W_{hg}h_{t-1} + W_{xg}x_t)$$

![img](lstm_mformula_1.png)

![img](lstm_mformula_2.png)

where $\odot$ is an element-wise Hadamard product. $g_t$ in the above formulas is an intermediary calculation cache that's later used with $o$ gate in the above formulas.

Since all $f$, $i$, $o$ gate vector values range from 0 to 1, because they were squashed by sigmoid function $\sigma$, when multiplied element-wise, we can see that:

**Forget Gate:** Forget gate $f_t$ at time step $t$ controls how much information needs to be "removed" from the previous cell state $c_{t-1}$. This forget gate learns to erase hidden representations from the previous time steps, which is why LSTM will have two hidden representations $h_t$ and cell state $c_t$. This $c_t$ will get propagated over time and learn whether to forget the previous cell state or not.

**Input Gate:** Input gate $i_t$ at time step $t$ controls how much information needs to be "added" to the next cell state $c_t$ from previous hidden state $h_{t-1}$ and input $x_t$. Instead of $\tanh$, the "input" gate $i$ has a sigmoid function, which converts inputs to values between zero and one. This serves as a switch, where values are either almost always zero or almost always one. This "input" gate decides whether to take the RNN output that is produced by the "gate" gate $g$ and multiplies the output with input gate $i$.

**Output Gate:** Output gate $o_t$ at time step $t$ controls how much information needs to be "shown" as output in the current hidden state $h_t$.

The key idea of LSTM is the cell state, the horizontal line running through between recurrent timesteps. You can imagine the cell state to be some kind of highway of information passing through straight down the entire chain, with only some minor linear interactions. Thus, even when there is a bunch of LSTMs stacked together, we can get an uninterrupted gradient flow where the gradients flow back through cell states instead of hidden states $h$ without vanishing in every time step. This greatly fixes the gradient vanishing/exploding problem we have outlined above.

![img](lstm_highway.png)

## Does LSTM solve the vanishing gradient problem?

LSTM architecture makes it easier for the RNN to preserve information over many recurrent time steps. For example, if the forget gate is set to 1, and the input gate is set to 0, then the information of the cell state will always be preserved over many recurrent time steps. For a Vanilla RNN, in contrast, it's much harder to preserve information in hidden states in recurrent time steps by just making use of a single weight matrix.

LSTMs do not guarantee that there is no vanishing/exploding gradient problems, but it does provide an easier way for the model to learn long-distance dependencies.
