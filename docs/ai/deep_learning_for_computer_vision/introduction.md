# Introduction

**Machine Learning** is a subset of artificial intelligence that focuses on algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience, without being explicitly programmed for every scenario. At its core, ML is about finding patterns in data and using those patterns to make predictions or decisions. The field encompasses supervised learning (learning from labeled examples), unsupervised learning (finding hidden patterns in unlabeled data), and reinforcement learning (learning through interaction with an environment).

**Deep Learning** is a specialized branch of machine learning that uses artificial neural networks with multiple layers (hence "deep") to model and understand complex patterns in data. These networks are inspired by the structure and function of the human brain, with interconnected nodes (neurons) that process information through weighted connections. Deep learning excels at tasks involving high-dimensional data like images, audio, and text, where traditional machine learning methods often struggle. Key architectures include convolutional neural networks (CNNs) for computer vision, recurrent neural networks (RNNs) and transformers for natural language processing, and generative adversarial networks (GANs) for creating synthetic data. The success of deep learning is largely due to the availability of large datasets, powerful computing resources (especially GPUs), and sophisticated optimization techniques like backpropagation. Deep learning has revolutionized fields like computer vision, natural language processing, speech recognition, and has enabled breakthroughs in areas ranging from protein folding prediction to creative AI applications.

## Hubel and Wiesel's Experiments
The foundation of modern computer vision and convolutional neural networks can be traced back to the groundbreaking work of David Hubel and Torsten Wiesel in the 1950s and 1960s. Their experiments on the visual cortex of cats and monkeys revealed fundamental principles about how the brain processes visual information.

Hubel and Wiesel inserted microelectrodes into the primary visual cortex (V1) of cats and monkeys to record the electrical activity of individual neurons while presenting various visual stimuli. They discovered that neurons in the visual cortex respond selectively to specific features rather than responding to all visual input uniformly.

**Simple Cells**: These neurons respond to oriented edges and lines at specific locations in the visual field. Each simple cell has a preferred orientation (e.g., horizontal, vertical, or diagonal) and responds most strongly when a line or edge of that orientation is presented at a particular location.

**Complex Cells**: These neurons also respond to oriented edges but are less sensitive to the exact position of the stimulus. They maintain their response even when the stimulus is shifted slightly within their receptive field, making them more invariant to position.

**Hierarchical Organization**: Hubel and Wiesel discovered that the visual cortex is organized hierarchically, with simple cells feeding into complex cells, which in turn feed into even more complex feature detectors. This hierarchical processing allows the brain to build increasingly complex representations from simple visual features.

The experiments revealed several fundamental principles that directly inspired the design of convolutional neural networks:

1. **Local Receptive Fields**: Neurons respond to stimuli in small, localized regions of the visual field, not the entire image.

2. **Feature Detection**: The visual system detects specific features (edges, orientations) rather than processing raw pixel values.

3. **Hierarchical Processing**: Complex visual patterns are built up from simpler features through multiple layers of processing.

4. **Translation Invariance**: Higher-level neurons become increasingly invariant to the exact position of features.

5. **Shared Weights**: Similar feature detectors are replicated across different spatial locations.

These insights provided the biological inspiration for convolutional neural networks, where convolutional layers detect local features (like edges), pooling layers provide translation invariance, and multiple layers build up hierarchical representations of increasing complexity. Hubel and Wiesel's work earned them the Nobel Prize in Physiology or Medicine in 1981 and remains foundational to our understanding of both biological and artificial vision systems.

## What solving Computer Vision means

Computer vision aims to enable machines to interpret and understand visual information from the world, essentially replicating the human ability to "see" and make sense of visual data.

**Core Tasks**: Object detection and recognition, scene understanding, motion analysis, 3D reconstruction, and visual reasoning. The ultimate goal is to extract meaningful information from images and videos that can be used for decision-making, navigation, interaction, and understanding.

### How Nature has solved Vision

**Compound Eyes (Insects)**: Insects like bees and dragonflies have compound eyes composed of thousands of individual photoreceptor units (ommatidia). Each unit captures light from a small portion of the visual field, creating a mosaic image. This design provides excellent motion detection and wide field of view, crucial for navigation and predator avoidance. The honeybee's visual system can detect polarized light, enabling navigation using the sun's position even on cloudy days.

**Camera Eyes (Vertebrates)**: Most vertebrates, including humans, have camera-like eyes with a single lens focusing light onto a retina. The retina contains specialized photoreceptor cells (rods for low-light vision, cones for color vision) and complex neural processing layers. This design provides high resolution and excellent color discrimination, enabling detailed object recognition and complex visual tasks.

**Specialized Vision Systems**: Different species have evolved unique adaptations. Mantis shrimp have 12-16 different photoreceptor types (compared to humans' 3), enabling them to see a vast spectrum of colors including ultraviolet and polarized light. Birds of prey have exceptional visual acuity - eagles can spot prey from kilometers away. Nocturnal animals like cats have enhanced low-light vision with reflective tapetum layers.

**Neural Processing**: All these systems share common principles: hierarchical feature detection (from simple edges to complex objects), parallel processing of different visual attributes (motion, color, depth), and extensive neural plasticity allowing adaptation to environmental changes. The visual cortex processes information in specialized regions - V1 for basic features, V2-V4 for intermediate processing, and higher areas for object recognition and scene understanding.