# Spatializing Data

## Data as points in space

All data can be represented as **coordinates in a multi-dimensional space**. Each **dimension** corresponds to a specific **feature** of the data. For example, a location might live in 3D space with axes for latitude, longitude, and altitude; a house listing might be a point in a space whose dimensions are square footage, number of bedrooms, number of bathrooms, year built, and so on. One data point = one vector of feature values = one point in that space. This view is useful because it lets you talk about **distance**, **neighborhoods**, and **direction**— concepts that underpin many ML algorithms (nearest neighbors, clustering, gradient-based optimization).

## Vectors enable ML processing

Machine learning models operate on **numbers**. Before a model can use your data, it must be turned into **numeric vectors**. Each **component** of the vector is the value of one feature, and the **length** of the vector (number of components) is the **dimensionality** of the space. So "spatializing" data really means: choose or construct features, assign each a dimension, and fill in the coordinates. Once data are vectors, models can compute similarities, apply linear or nonlinear transformations, and learn mappings between spaces. Raw text, images, or categorical labels cannot be fed in as-is; they have to be **vectorized** first.

## Universal vectorization requirement

Regardless of **original modality**— text, images, audio, tabular records, graphs— all of it must undergo feature engineering to become **numeric vectors** that ML models can process.