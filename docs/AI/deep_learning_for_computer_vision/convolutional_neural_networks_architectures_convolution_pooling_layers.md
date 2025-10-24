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

## ConvNet Architectures
Watch YT lec 6 and then continue.
