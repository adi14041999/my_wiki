# Convolutional Neural Networks: Architectures, Convolution / Pooling Layers

## Architecture Overview

Regular Neural Nets don't scale well to full images. In CIFAR-10, images are only of size 32x32x3 (32 wide, 32 high, 3 color channels), so a single fully-connected neuron in a first hidden layer of a regular Neural Network would have 32*32*3 = 3072 weights. This amount still seems manageable, but clearly this fully-connected structure does not scale to larger images. For example, an image of more respectable size, e.g. 200x200x3, would lead to neurons that have 120,000 weights. Clearly, this full connectivity is wasteful and the huge number of parameters would quickly lead to overfitting.

Convolutional Neural Networks take advantage of the fact that the input consists of images and they constrain the architecture in a more sensible way. In particular, unlike a regular Neural Network, the layers of a ConvNet have neurons arranged in 3 dimensions: width, height, depth. Note that the word depth here refers to the third dimension of an activation volume, not to the depth of a full Neural Network, which can refer to the total number of layers in a network. For example, the input images in CIFAR-10 are an input volume of activations, and the volume has dimensions 32x32x3 (width, height, depth respectively). As we will soon see, the neurons in a layer will only be connected to a small region of the layer before it, instead of all of the neurons in a fully-connected manner. Moreover, the final output layer would for CIFAR-10 have dimensions 1x1x10, because by the end of the ConvNet architecture we will reduce the full image into a single vector of class scores, arranged along the depth dimension.

A ConvNet is made up of Layers. Every Layer has a simple API: It transforms an input 3D volume to an output 3D volume with some differentiable function that may or may not have parameters.

## Layers used to build ConvNets

We use three main types of layers to build ConvNet architectures: Convolutional Layer, Pooling Layer, and Fully-Connected Layer (exactly as seen in regular Neural Networks). We will stack these layers to form a full ConvNet architecture.

We will go into more details below, but a simple ConvNet for CIFAR-10 classification could have the architecture [INPUT - CONV - RELU - POOL - FC]. In more detail:

- **INPUT [32x32x3]** will hold the raw pixel values of the image, in this case an image of width 32, height 32, and with three color channels R,G,B.
- **CONV layer** will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. This may result in volume such as [32x32x12] if we decided to use 12 filters.
- **RELU layer** will apply an elementwise activation function, such as the max(0,x) thresholding at zero. This leaves the size of the volume unchanged ([32x32x12]).
- **POOL layer** will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16x16x12].
- **FC (i.e. fully-connected) layer** will compute the class scores, resulting in volume of size [1x1x10], where each of the 10 numbers correspond to a class score, such as among the 10 categories of CIFAR-10. As with ordinary Neural Networks and as the name implies, each neuron in this layer will be connected to all the numbers in the previous volume.

![image](convnet.jpeg)

The above image shows the activations of an example ConvNet architecture. The initial volume stores the raw image pixels (left) and the last volume stores the class scores (right). Each volume of activations along the processing path is shown as a column. The last layer volume holds the scores for each class, but here we only visualize the sorted top 5 scores, and print the labels of each one. The full web-based demo is shown in the header of this [website](http://cs231n.stanford.edu/). The architecture shown here is a tiny VGG Net.

### Convolutional Layer

#### Local Connectivity

When dealing with high-dimensional inputs such as images, it is impractical to connect neurons to all neurons in the previous volume. Instead, we will connect each neuron to only a local region of the input volume. The spatial extent of this connectivity is a hyperparameter called the receptive field of the neuron (equivalently this is the filter size). The extent of the connectivity along the depth axis is always equal to the depth of the input volume. It is important to emphasize again this asymmetry in how we treat the spatial dimensions (width and height) and the depth dimension: the connections are local in 2D space (along width and height), but always full along the entire depth of the input volume.

**Example:** For example, suppose that the input volume has size [32x32x3], (e.g. an RGB CIFAR-10 image). If the receptive field (or the filter size) is 5x5, then each neuron in the Conv Layer will have weights to a [5x5x3] region in the input volume, for a total of 75 weights (and +1 bias parameter). Notice that the extent of the connectivity along the depth axis must be 3, since this is the depth of the input volume.

**Example:** Suppose an input volume had size [16x16x20]. Then using an example receptive field size of 3x3, every neuron in the Conv Layer would now have a total of 180 connections to the input volume. Notice that, again, the connectivity is local in 2D space (e.g. 3x3), but full along the input depth (20).

#### Spatial arrangement

We have explained the connectivity of each neuron in the Conv Layer to the input volume, but we haven't yet discussed how many neurons there are in the output volume or how they are arranged. Three hyperparameters control the size of the output volume: the depth, stride and zero-padding.

##### Depth

The depth of the output volume is a hyperparameter: it corresponds to the number of filters we would like to use, each learning to look for something different in the input.
##### Stride 

We must specify the stride with which we slide the filter. When the stride is 1 then we move the filters one pixel at a time. When the stride is 2 (or uncommonly 3 or more, though this is rare in practice) then the filters jump 2 pixels at a time as we slide them around. This will produce smaller output volumes spatially.

##### Zero-padding 

Sometimes it will be convenient to pad the input volume with zeros around the border. The size of this zero-padding is a hyperparameter. The nice feature of zero padding is that it will allow us to control the spatial size of the output volumes (most commonly as we'll see soon we will use it to exactly preserve the spatial size of the input volume so the input and output width and height are the same).

We can compute the spatial size of the output volume as a function of the input volume size ($W$), the receptive field size of the Conv Layer neurons ($F$), the stride with which they are applied ($S$), and the amount of zero padding used ($P$) on the border. You can convince yourself that the correct formula for calculating how many neurons "fit" is given by $(W-F+2P)/S+1$. For example for a 7x7 input and a 3x3 filter with stride 1 and pad 0 we would get a 5x5 output. With stride 2 we would get a 3x3 output.

In general, setting zero padding to be $P=(F-1)/2$ when the stride is $S=1$ ensures that the input volume and output volume will have the same size spatially. It is very common to use zero-padding in this way.

Note again that the spatial arrangement hyperparameters have mutual constraints. For example, when the input has size $W=10$, no zero-padding is used $P=0$, and the filter size is $F=3$, then it would be impossible to use stride $S=2$, since $(W-F+2P)/S+1=(10-3+0)/2+1=4.5$, i.e. not an integer, indicating that the neurons don't "fit" neatly and symmetrically across the input. Therefore, this setting of the hyperparameters is considered to be invalid, and a ConvNet library could throw an exception or zero pad the rest to make it fit, or crop the input to make it fit, or something.

#### Case study (detailed example)

Suppose that the input volume X has shape `X.shape: (11,11,4)`. Suppose further that we use no zero padding ($P=0$), that the filter size is $F=5$, and that the stride is $S=2$. The output volume would therefore have spatial size $(11-5)/2+1 = 4$, giving a volume with width and height of 4. The activation map in the output volume (call it V), would then look as follows (only some of the elements are computed in this example):

```python
V[0,0,0] = np.sum(X[:5,:5,:] * W0) + b0
V[1,0,0] = np.sum(X[2:7,:5,:] * W0) + b0
V[2,0,0] = np.sum(X[4:9,:5,:] * W0) + b0
V[3,0,0] = np.sum(X[6:11,:5,:] * W0) + b0
```

Remember that in numpy, the operation `*` above denotes elementwise multiplication between the arrays. Notice also that the weight vector W0 is the weight vector of that neuron and b0 is the bias. Here, W0 is assumed to be of shape `W0.shape: (5,5,4)`, since the filter size is 5 and the depth of the input volume is 4. Notice that at each point, we are computing the dot product as seen before in ordinary neural networks. Also, we see that we are using the same weight and bias (due to parameter sharing), and the dimensions along the width are increasing in steps of 2 (i.e. the stride). To construct a second activation map in the output volume, we would have:

```python
V[0,0,1] = np.sum(X[:5,:5,:] * W1) + b1
V[1,0,1] = np.sum(X[2:7,:5,:] * W1) + b1
V[2,0,1] = np.sum(X[4:9,:5,:] * W1) + b1
V[3,0,1] = np.sum(X[6:11,:5,:] * W1) + b1
V[0,1,1] = np.sum(X[:5,2:7,:] * W1) + b1
V[2,3,1] = np.sum(X[4:9,6:11,:] * W1) + b1
```

where we see that we are indexing into the second depth dimension in V (at index 1) because we are computing the second activation map, and that a different set of parameters (W1) is now used. In the example above, we are for brevity leaving out some of the other operations the Conv Layer would perform to fill the other parts of the output array V. Additionally, recall that these activation maps are often followed elementwise through an activation function such as ReLU, but this is not shown here.

#### Convolution Demo

Below is a running demo of a CONV layer. Since 3D volumes are hard to visualize, all the volumes (the input volume (in blue), the weight volumes (in red), the output volume (in green)) are visualized with each depth slice stacked in rows. The input volume is of size $W_1=5, H_1=5, D_1=3$, and the CONV layer parameters are $K=2, F=3, S=2, P=1$. That is, we have two filters of size $3 \times 3$, and they are applied with a stride of 2. Therefore, the output volume size has spatial size $(5 - 3 + 2)/2 + 1 = 3$. Moreover, notice that a padding of $P=1$ is applied to the input volume, making the outer border of the input volume zero. The visualization below iterates over the output activations (green), and shows that each element is computed by elementwise multiplying the highlighted input (blue) with the filter (red), summing it up, and then offsetting the result by the bias.

<div class="fig figcenter fighighlight">
  <div id="conv-demo" style="width: 100%; height: 700px; border: 1px solid #ccc; background: #f9f9f9; position: relative; overflow: hidden;">
    <canvas id="conv-canvas" width="800" height="700" style="position: absolute; top: 0; left: 0;"></canvas>
  </div>
</div>

<script>
(function() {
  // Volume class
  function Vol(sx, sy, depth, c) {
    this.sx = sx;
    this.sy = sy;
    this.depth = depth;
    var n = sx * sy * depth;
    this.w = new Array(n);
    if (typeof c === 'undefined') {
      for (var i = 0; i < n; i++) {
        this.w[i] = Math.floor(Math.random() * 3);
      }
    } else {
      for (var i = 0; i < n; i++) {
        this.w[i] = c;
      }
    }
  }
  
  Vol.prototype.get = function(x, y, d) {
    var ix = ((this.sx * y) + x) * this.depth + d;
    return this.w[ix];
  };
  
  Vol.prototype.set = function(x, y, d, v) {
    var ix = ((this.sx * y) + x) * this.depth + d;
    this.w[ix] = v;
  };
  
  // Convolution parameters
  var W1 = 7, H1 = 7, D1 = 3;  // input volume
  var K = 2, F = 3, S = 2;      // 2 filters, 3x3, stride 2
  var cs = 25;                  // cell size
  
  // Create input volume with padding
  var X = new Vol(W1, H1, D1);
  // Zero pad the borders
  for (var d = 0; d < X.depth; d++) {
    for (var x = 0; x < X.sx; x++) {
      for (var y = 0; y < X.sy; y++) {
        if (x === 0 || x === (X.sx - 1) || y === 0 || y === (X.sy - 1)) {
          X.set(x, y, d, 0);
        }
      }
    }
  }
  
  // Create filters and biases
  var Ws = [];
  var bs = [];
  for (var k = 0; k < K; k++) {
    var W = new Vol(F, F, D1);
    for (var q = 0; q < W.w.length; q++) {
      W.w[q] = Math.floor(Math.random() * 3) - 1;
    }
    Ws.push(W);
    var b = new Vol(1, 1, 1);
    b.w[0] = 1 - k;
    bs.push(b);
  }
  
  // Convolution forward pass
  var conv_forward = function(V, Ws, bs, stride) {
    var out_sy = Math.floor((V.sy - Ws[0].sy) / stride + 1);
    var out_sx = Math.floor((V.sx - Ws[0].sx) / stride + 1);
    var A = new Vol(out_sx, out_sy, Ws.length, 0.0);
    
    for (var d = 0; d < Ws.length; d++) {
      var f = Ws[d];
      var x = 0, y = 0;
      for (var ay = 0; ay < out_sy; y += stride, ay++) {
        x = 0;
        for (var ax = 0; ax < out_sx; x += stride, ax++) {
          var a = 0.0;
          for (var fy = 0; fy < f.sy; fy++) {
            var oy = y + fy;
            for (var fx = 0; fx < f.sx; fx++) {
              var ox = x + fx;
              if (oy >= 0 && oy < V.sy && ox >= 0 && ox < V.sx) {
                for (var fd = 0; fd < f.depth; fd++) {
                  a += f.w[((f.sx * fy) + fx) * f.depth + fd] * 
                       V.w[((V.sx * oy) + ox) * V.depth + fd];
                }
              }
            }
          }
          a += bs[d].w[0];
          A.set(ax, ay, d, a);
        }
      }
    }
    return A;
  };
  
  // Canvas rendering
  var canvas = document.getElementById('conv-canvas');
  var ctx = canvas.getContext('2d');
  
  // Render volume function
  function renderVol(V, xoff, yoff, col, title, vid) {
    var pad = 3;
    var dpad = 20;
    var gyoff = 20;
    
    var txt = title + ' (' + V.sx + 'x' + V.sy + 'x' + V.depth + ')';
    
    // Add title
    ctx.fillStyle = 'black';
    ctx.font = '16px Arial';
    ctx.fillText(txt, xoff, yoff - 5);
    
    for (var d = 0; d < V.depth; d++) {
      // Add depth label
      ctx.fillStyle = 'black';
      ctx.font = '16px Courier';
      ctx.fillText(vid + '[:,:,' + d + ']', xoff, yoff + d * (V.sy * (cs + pad) + dpad) + gyoff - 5);
      
      for (var x = 0; x < V.sx; x++) {
        for (var y = 0; y < V.sy; y++) {
          var xcoord = xoff + x * (cs + pad);
          var ycoord = yoff + y * (cs + pad) + d * (V.sy * (cs + pad) + dpad) + gyoff;
          
          var thecol = col;
          if (vid === 'x' && (x === 0 || y === 0 || x === V.sx - 1 || y === V.sy - 1)) {
            thecol = '#DDD';
          }
          
          // Store cell info for highlighting
          if (!window.cellInfo) window.cellInfo = {};
          window.cellInfo[vid + '_' + x + '_' + y + '_' + d] = {
            x: xcoord, y: ycoord, width: cs, height: cs,
            color: thecol, value: V.get(x, y, d)
          };
          
          // Add rectangle
          ctx.fillStyle = thecol;
          ctx.fillRect(xcoord, ycoord, cs, cs);
          
          // Add border
          ctx.strokeStyle = 'black';
          ctx.lineWidth = 1;
          ctx.strokeRect(xcoord, ycoord, cs, cs);
          
          // Add text
          ctx.fillStyle = 'black';
          ctx.font = '16px Arial';
          ctx.fillText(V.get(x, y, d).toFixed(0), xcoord + 5, ycoord + 15);
        }
      }
    }
  }
  
  // Animation variables
  var fxg = 0, fyg = 0, fdg = 0;
  var iid = -1;
  var O;
  
  // Focus cell function
  function focusCell() {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Redraw everything
    var yoff = 20;
    renderVol(X, 10, yoff, '#DDF', 'Input Volume (+pad 1)', 'x');
    
    for (var i = 0; i < Ws.length; i++) {
      renderVol(Ws[i], 270 + i * 170, yoff, '#FDD', 'Filter W' + i, 'w' + i);
      renderVol(bs[i], 270 + i * 170, 350 + yoff, '#FDD', 'Bias b' + i, 'b' + i);
    }
    
    renderVol(O, 600, yoff, '#DFD', 'Output Volume', 'o');
    
    // Add toggle button
    ctx.fillStyle = 'black';
    ctx.font = '16px Arial';
    ctx.fillText('toggle movement', 520, 470);
    
    ctx.fillStyle = 'rgba(200, 200, 200, 0.1)';
    ctx.fillRect(500, 450, 150, 30);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.strokeRect(500, 450, 150, 30);
    
    // Highlight current cells
    var fx = fxg;
    var fy = fyg;
    var fd = fdg;
    
    // Highlight output cell
    var outputCell = window.cellInfo['o_' + fx + '_' + fy + '_' + fd];
    if (outputCell) {
      ctx.strokeStyle = '#0A0';
      ctx.lineWidth = 3;
      ctx.strokeRect(outputCell.x, outputCell.y, outputCell.width, outputCell.height);
    }
    
    // Highlight weights and bias
    for (var i = 0; i < Ws[fd].sx * Ws[fd].sy * Ws[fd].depth; i++) {
      var x = i % Ws[fd].sx;
      var y = Math.floor(i / Ws[fd].sx) % Ws[fd].sy;
      var d = Math.floor(i / (Ws[fd].sx * Ws[fd].sy));
      var weightCell = window.cellInfo['w' + fd + '_' + x + '_' + y + '_' + d];
      if (weightCell) {
        ctx.strokeStyle = '#A00';
        ctx.lineWidth = 3;
        ctx.strokeRect(weightCell.x, weightCell.y, weightCell.width, weightCell.height);
      }
    }
    
    var biasCell = window.cellInfo['b' + fd + '_0_0_0'];
    if (biasCell) {
      ctx.strokeStyle = '#A00';
      ctx.lineWidth = 3;
      ctx.strokeRect(biasCell.x, biasCell.y, biasCell.width, biasCell.height);
    }
    
    // Highlight input cells and draw connections
    for (var d = 0; d < D1; d++) {
      for (var x = 0; x < F; x++) {
        for (var y = 0; y < F; y++) {
          var ix = fx * S + x;
          var iy = fy * S + y;
          var id = d;
          var inputCell = window.cellInfo['x_' + ix + '_' + iy + '_' + id];
          if (inputCell) {
            ctx.strokeStyle = '#00A';
            ctx.lineWidth = 3;
            ctx.strokeRect(inputCell.x, inputCell.y, inputCell.width, inputCell.height);
            
            // Draw connection line
            if (x === 0 && y === 0) {
              var weightCell = window.cellInfo['w' + fd + '_' + x + '_' + y + '_' + d];
              if (weightCell) {
                ctx.strokeStyle = 'black';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(inputCell.x + inputCell.width, inputCell.y + inputCell.height/2);
                ctx.lineTo(weightCell.x, weightCell.y + weightCell.height/2);
                ctx.stroke();
              }
            }
          }
        }
      }
    }
    
    // Cycle to next output cell
    fxg++;
    if (fxg >= O.sx) {
      fxg = 0;
      fyg++;
      if (fyg >= O.sy) {
        fyg = 0;
        fdg++;
        if (fdg >= O.depth) {
          fdg = 0;
        }
      }
    }
  }
  
  // Draw function
  function draw() {
    var yoff = 20;
    
    // Render input volume
    renderVol(X, 10, yoff, '#DDF', 'Input Volume (+pad 1)', 'x');
    
    // Render filters and biases
    for (var i = 0; i < Ws.length; i++) {
      renderVol(Ws[i], 270 + i * 170, yoff, '#FDD', 'Filter W' + i, 'w' + i);
      renderVol(bs[i], 270 + i * 170, 350 + yoff, '#FDD', 'Bias b' + i, 'b' + i);
    }
    
    // Render output
    renderVol(O, 600, yoff, '#DFD', 'Output Volume', 'o');
    
    // Add toggle button
    ctx.fillStyle = 'black';
    ctx.font = '16px Arial';
    ctx.fillText('toggle movement', 520, 470);
    
    ctx.fillStyle = 'rgba(200, 200, 200, 0.1)';
    ctx.fillRect(500, 450, 150, 30);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.strokeRect(500, 450, 150, 30);
    
    // Add click handler
    canvas.addEventListener('click', function(e) {
      var rect = canvas.getBoundingClientRect();
      var x = e.clientX - rect.left;
      var y = e.clientY - rect.top;
      
      if (x >= 500 && x <= 650 && y >= 450 && y <= 480) {
        if (iid === -1) {
          iid = setInterval(focusCell, 1000);
        } else {
          clearInterval(iid);
          iid = -1;
        }
      }
    });
  }
  
  // Initialize
  function start() {
    O = conv_forward(X, Ws, bs, S);
    draw();
    iid = setInterval(focusCell, 1000);
  }
  
  // Start the animation
  start();
})();
</script>

### Pooling Layer

It is common to periodically insert a Pooling layer in-between successive Conv layers in a ConvNet architecture. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting. The Pooling Layer operates independently on every depth slice of the input and resizes it spatially, using the MAX operation. The most common form is a pooling layer with filters of size 2x2 applied with a stride of 2 downsamples every depth slice in the input by 2 along both width and height, discarding 75% of the activations. Every MAX operation would in this case be taking a max over 4 numbers (little 2x2 region in some depth slice). The depth dimension remains unchanged. More generally, the pooling layer:

- Accepts a volume of size $W_1 \times H_1 \times D_1$
- Requires two hyperparameters:
  - their spatial extent $F$,
  - the stride $S$,
- Produces a volume of size $W_2 \times H_2 \times D_2$ where:
  - $W_2 = (W_1 - F)/S + 1$
  - $H_2 = (H_1 - F)/S + 1$
  - $D_2 = D_1$
- Introduces zero parameters since it computes a fixed function of the input
- For Pooling layers, it is not common to pad the input using zero-padding.

It is worth noting that there are only two commonly seen variations of the max pooling layer found in practice: A pooling layer with $F=3, S=2$ (also called overlapping pooling), and more commonly $F=2, S=2$. Pooling sizes with larger receptive fields are too destructive.

![maxpool](pool.jpeg)
![maxpool](maxpool.jpeg)

#### General pooling 

In addition to max pooling, the pooling units can also perform other functions, such as average pooling or even L2-norm pooling. Average pooling was often used historically but has recently fallen out of favor compared to the max pooling operation, which has been shown to work better in practice.

### Fully-connected layer

Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset.

### Batch Normalization

To understand the goal of batch normalization, it is important to first recognize that machine learning methods tend to perform better with input data consisting of uncorrelated features with zero mean and unit variance. When training a neural network, we can preprocess the data before feeding it to the network to explicitly decorrelate its features. This will ensure that the first layer of the network sees data that follows a nice distribution. However, even if we preprocess the input data, the activations at deeper layers of the network will likely no longer be decorrelated and will no longer have zero mean or unit variance, since they are output from earlier layers in the network. Even worse, during the training process the distribution of features at each layer of the network will shift as the weights of each layer are updated.

The authors of [1] hypothesize that the shifting distribution of features inside deep neural networks may make training deep networks more difficult. To overcome this problem, they propose to insert into the network some layers that normalize batches. At training time, such a layer uses a minibatch of data to estimate the mean and standard deviation of each feature. These estimated means and standard deviations are then used to center and normalize the features of the minibatch. A running average of these means and standard deviations is kept during training, and at test time these running averages are used to center and normalize features.

It is possible that this normalization strategy could reduce the representational power of the network, since it may sometimes be optimal for certain layers to have features that are not zero-mean or unit variance. To this end, the batch normalization layer includes learnable shift and scale parameters for each feature dimension.

**Forward pass implementation:**

```python
def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out = None
    if mode == "train":
        """
        In NumPy, axis=0 refers to the first dimension of an array. In a two-dimensional
        array, this corresponds to the rows. When you specify axis=0 for a calculation,
        you are telling NumPy to perform the operation across the rows, collapsing them
        and producing a result for each column. Here, for each feature, we're computing
        the mean and variance across the mini-batch.
        Thus, sample_mean = [mean_feature_0, mean_feature_1, ...]
        """
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_normalized + beta
        running_mean = (1 - momentum) * running_mean + momentum * sample_mean
        running_var = (1 - momentum) * running_var + momentum * sample_var
    elif mode == "test":
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_normalized + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
    
    return out
```

[1] [Sergey Ioffe and Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift", ICML 2015.](https://arxiv.org/abs/1502.03167)

## ConvNet Architectures

### Layer Patterns

The most common form of a ConvNet architecture stacks a few CONV-RELU layers, follows them with POOL layers, and repeats this pattern until the image has been merged spatially to a small size. At some point, it is common to transition to fully-connected layers. The last fully-connected layer holds the output, such as the class scores. In other words, the most common ConvNet architecture follows the pattern:

**INPUT -> [[CONV -> RELU] * N -> POOL?] * M -> [FC -> RELU] * K -> FC**

where the * indicates repetition, and the POOL? indicates an optional pooling layer. Moreover, N >= 0 (and usually N <= 3), M >= 0, K >= 0 (and usually K < 3). For example, here are some common ConvNet architectures you may see that follow this pattern:

- **INPUT -> FC**, implements a linear classifier. Here N = M = K = 0.
- **INPUT -> CONV -> RELU -> FC**
- **INPUT -> [CONV -> RELU -> POOL] * 2 -> FC -> RELU -> FC**. Here we see that there is a single CONV layer between every POOL layer.
- **INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL] * 3 -> [FC -> RELU] * 2 -> FC** Here we see two CONV layers stacked before every POOL layer. This is generally a good idea for larger and deeper networks, because multiple stacked CONV layers can develop more complex features of the input volume before the destructive pooling operation.

#### Prefer a stack of small filter CONV to one large receptive field CONV layer 

Suppose that you stack three 3x3 CONV layers on top of each other (with non-linearities in between, of course). In this arrangement, each neuron on the first CONV layer has a 3x3 view of the input volume. A neuron on the second CONV layer has a 3x3 view of the first CONV layer, and hence by extension a 5x5 view of the input volume. Similarly, a neuron on the third CONV layer has a 3x3 view of the 2nd CONV layer, and hence a 7x7 view of the input volume. Suppose that instead of these three layers of 3x3 CONV, we only wanted to use a single CONV layer with 7x7 receptive fields. These neurons would have a receptive field size of the input volume that is identical in spatial extent (7x7), but with several disadvantages. First, the neurons would be computing a linear function over the input, while the three stacks of CONV layers contain non-linearities that make their features more expressive. Second, if we suppose that all the volumes have $C$ channels (in practice, you might see architectures that increase channels with depth (e.g., 64 → 128 → 256), but for this theoretical comparison, keeping $C$ constant makes the parameter count analysis clear and fair), then it can be seen that the single 7x7 CONV layer would contain $C \times (7 \times 7 \times C) = 49C^2$ parameters, while the three 3x3 CONV layers would only contain $3 \times (C \times (3 \times 3 \times C)) = 27C^2$ parameters.

For the **single 7×7 CONV layer**:

- Each filter has size $7 \times 7 \times C$ (7×7 spatial dimensions × C input channels)

- We have $C$ output channels (filters)

- Total parameters = $C \times (7 \times 7 \times C) = C \times 49C = 49C^2$

For the **three stacked 3×3 CONV layers**:

- Each 3×3 filter has size $3 \times 3 \times C$ 

- We have $C$ output channels (filters) per layer

- Total parameters per layer = $C \times (3 \times 3 \times C) = C \times 9C = 9C^2$

- Total for 3 layers = $3 \times 9C^2 = 27C^2$

Intuitively, stacking CONV layers with tiny filters as opposed to having one CONV layer with big filters allows us to express more powerful features of the input, and with fewer parameters. As a practical disadvantage, we might need more memory to hold all the intermediate CONV layer results if we plan to do backpropagation.

### Layer Sizing Patterns

Until now we've omitted mentions of common hyperparameters used in each of the layers in a ConvNet. We will first state the common rules of thumb for sizing the architectures.

**The input layer** (that contains the image) should be divisible by 2 many times. Common numbers include 32 (e.g. CIFAR-10), 64, 96 (e.g. STL-10), or 224 (e.g. common ImageNet ConvNets), 384, and 512.

**The conv layers** should be using small filters (e.g. 3x3 or at most 5x5), using a stride of $S=1$, and crucially, padding the input volume with zeros in such way that the conv layer does not alter the spatial dimensions of the input. That is, when $F=3$, then using $P=1$ will retain the original size of the input. When $F=5$, $P=2$. For a general $F$, it can be seen that $P=(F-1)/2$ preserves the input size. If you must use bigger filter sizes (such as 7x7 or so), it is only common to see this on the very first conv layer that is looking at the input image.

**The pool layers** are in charge of downsampling the spatial dimensions of the input. The most common setting is to use max-pooling with 2x2 receptive fields (i.e. $F=2$), and with a stride of 2 (i.e. $S=2$). Note that this discards exactly 75% of the activations in an input volume (due to downsampling by 2 in both width and height). Another slightly less common setting is to use 3x3 receptive fields with a stride of 2, but this makes "fitting" more complicated (e.g., a 32x32x3 layer would require zero padding to be used with a max-pooling layer with 3x3 receptive field and stride 2). It is very uncommon to see receptive field sizes for max pooling that are larger than 3 because the pooling is then too lossy and aggressive. This usually leads to worse performance.

#### Why use stride of 1 in CONV? 

Smaller strides work better in practice. Additionally, as already mentioned stride 1 allows us to leave all spatial down-sampling to the POOL layers, with the CONV layers only transforming the input volume depth-wise.

#### Why use padding?

In addition to the aforementioned benefit of keeping the spatial sizes constant after CONV, doing this actually improves performance. If the CONV layers were to not zero-pad the inputs and only perform valid convolutions, then the size of the volumes would reduce by a small amount after each CONV, and the information at the borders would be "washed away" too quickly.

### Case studies

There are several architectures in the field of Convolutional Networks that have a name. The most common are:

[**LeNet:**](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) The first successful applications of Convolutional Networks were developed by Yann LeCun in 1990's. Of these, the best known is the LeNet architecture that was used to read zip codes, digits, etc.

[**AlexNet:**](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) The first work that popularized Convolutional Networks in Computer Vision was the AlexNet, developed by Alex Krizhevsky, Ilya Sutskever and Geoff Hinton. The AlexNet was submitted to the [ImageNet ILSVRC challenge](http://www.image-net.org/challenges/LSVRC/2014/) in 2012 and significantly outperformed the second runner-up (top 5 error of 16% compared to runner-up with 26% error). The Network had a very similar architecture to LeNet, but was deeper, bigger, and featured Convolutional Layers stacked on top of each other (previously it was common to only have a single CONV layer always immediately followed by a POOL layer).

[**ZF Net:**](http://arxiv.org/abs/1311.2901) The ILSVRC 2013 winner was a Convolutional Network from Matthew Zeiler and Rob Fergus. It became known as the ZFNet (short for Zeiler & Fergus Net). It was an improvement on AlexNet by tweaking the architecture hyperparameters, in particular by expanding the size of the middle convolutional layers and making the stride and filter size on the first layer smaller.

**GoogLeNet:** The ILSVRC 2014 winner was a Convolutional Network from [Szegedy et al](http://arxiv.org/abs/1409.4842). from Google. Its main contribution was the development of an Inception Module that dramatically reduced the number of parameters in the network (4M, compared to AlexNet with 60M). Additionally, this paper uses Average Pooling instead of Fully Connected layers at the top of the ConvNet, eliminating a large amount of parameters that do not seem to matter much. There are also several followup versions to the GoogLeNet, most recently [Inception-v4](http://arxiv.org/abs/1602.07261).

[**VGGNet:**](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) The runner-up in ILSVRC 2014 was the network from Karen Simonyan and Andrew Zisserman that became known as the VGGNet. Its main contribution was in showing that the depth of the network is a critical component for good performance. Their final best network contains 16 CONV/FC layers and, appealingly, features an extremely homogeneous architecture that only performs 3x3 convolutions and 2x2 pooling from the beginning to the end. Their [pretrained model](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) is available for plug and play use in Caffe. A downside of the VGGNet is that it is more expensive to evaluate and uses a lot more memory and parameters (140M). Most of these parameters are in the first fully connected layer, and it was since found that these FC layers can be removed with no performance downgrade, significantly reducing the number of necessary parameters.

**ResNet:** [Residual Network](http://arxiv.org/abs/1512.03385) developed by Kaiming He et al. was the winner of ILSVRC 2015. It features special skip connections and a heavy use of [batch normalization](http://arxiv.org/abs/1502.03167). The architecture is also missing fully connected layers at the end of the network. The reader is also referred to Kaiming's presentation ([video](https://www.youtube.com/watch?v=1PGLj-uKT1w), [slides](http://research.microsoft.com/en-us/um/people/kahe/ilsvrc15/ilsvrc2015_deep_residual_learning_kaiminghe.pdf)), and some [recent experiments](https://github.com/gcr/torch-residual-networks) that reproduce these networks in Torch. ResNets are currently by far state of the art Convolutional Neural Network models and are the default choice for using ConvNets in practice (as of May 10, 2016). In particular, also see more recent developments that tweak the original architecture from [Kaiming He et al. Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (published March 2016).

#### VGGNet in detail

Let's break down the VGGNet in more detail as a case study. The whole VGGNet is composed of CONV layers that perform 3x3 convolutions with stride 1 and pad 1, and of POOL layers that perform 2x2 max pooling with stride 2 (and no padding). We can write out the size of the representation at each step of the processing and keep track of both the representation size and the total number of weights:

```
INPUT: [224x224x3]        memory:  224*224*3=150K   weights: 0
CONV3-64: [224x224x64]  memory:  224*224*64=3.2M   weights: (3*3*3)*64 = 1,728
CONV3-64: [224x224x64]  memory:  224*224*64=3.2M   weights: (3*3*64)*64 = 36,864
POOL2: [112x112x64]  memory:  112*112*64=800K   weights: 0
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M   weights: (3*3*64)*128 = 73,728
CONV3-128: [112x112x128]  memory:  112*112*128=1.6M   weights: (3*3*128)*128 = 147,456
POOL2: [56x56x128]  memory:  56*56*128=400K   weights: 0
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*128)*256 = 294,912
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
CONV3-256: [56x56x256]  memory:  56*56*256=800K   weights: (3*3*256)*256 = 589,824
POOL2: [28x28x256]  memory:  28*28*256=200K   weights: 0
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*256)*512 = 1,179,648
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [28x28x512]  memory:  28*28*512=400K   weights: (3*3*512)*512 = 2,359,296
POOL2: [14x14x512]  memory:  14*14*512=100K   weights: 0
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
CONV3-512: [14x14x512]  memory:  14*14*512=100K   weights: (3*3*512)*512 = 2,359,296
POOL2: [7x7x512]  memory:  7*7*512=25K  weights: 0
FC: [1x1x4096]  memory:  4096  weights: 7*7*512*4096 = 102,760,448
FC: [1x1x4096]  memory:  4096  weights: 4096*4096 = 16,777,216
FC: [1x1x1000]  memory:  1000 weights: 4096*1000 = 4,096,000
```

**TOTAL memory:** 24M * 4 bytes ~= 93MB / image (only forward! ~*2 for bwd)

**TOTAL params:** 138M parameters

As is common with Convolutional Networks, notice that most of the memory (and also compute time) is used in the early CONV layers, and that most of the parameters are in the last FC layers. In this particular case, the first FC layer contains 100M weights, out of a total of 140M.

## Additional Resources
Additional resources related to implementation:

1. [Soumith benchmarks for CONV performance](https://github.com/soumith/convnet-benchmarks)

2. [ConvNetJS CIFAR-10 demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html) allows you to play with ConvNet architectures and see the results and computations in real time, in the browser.

3. [Caffe](http://caffe.berkeleyvision.org/), one of the popular ConvNet libraries.

4. [State of the art ResNets in Torch7](http://torch.ch/blog/2016/02/04/resnets.html)

