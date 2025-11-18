# Decoder-Only Transformers

![img](d0.png)

![img](d1.png)

![img](d2.png)

![img](d3.png)

![img](d4.png)

![img](d5.png)

![img](d6.png)

![img](d7.png)

![img](d22.png)

![img](d23.png)

![img](d24.png)

![img](d25.png)

**One way to calculate similarities between the Query and the Keys is to calculate something called a Dot Product.**

![img](d27.png)

![img](d28.png)

![img](d29.png)

![img](d30.png)

Now we use the **Query** and **Key** for **What** to calculate the similarity with itself...

![img](d32.png)

![img](d35.png)

![img](d36.png)

![img](d38.png)

![img](d39.png)

Reusing the sets of **Weights** for the **Query**, **Key** and **Value** numbers lets the **Decoder-only Transformer** handle prompts that have different lengths.

![img](d41.png)

Lastly, we need a way to use the encodings we have for each word in the prompt to generate the word that follows it and then to generate a response.

![img](d43.png)

![img](d44.png)

![img](d45.png)

Beacause this is a **Decoder-only Transformer**, we need one thing that can both encode the prompt and generate the output.

Thus, even though we are not yet generating a response, we need to include the parts that will do that.

Also, we can compare the known input to what the model generates when we train the model.

![img](d49.png)

![img](d50.png)

![img](d51.png)

![img](d53.png)

![img](d54.png)

![img](d55.png)

**Note: If we were training the Decoder-only Transformer, then we would use the fact that we made a mistake to modify the Weights and Biases.**

In contrast, when we are using the model to generate responses, we don't really care what words come out.

![img](d59.png)

![img](d60.png)

Let’s review.

![img](d61.png)

![img](d62.png)

![img](d63.png)

![img](d64.png)

![img](d65.png)

![img](d66.png)

![img](d67.png)

![img](d68.png)

![img](d69.png)

However, it is also important to keep track of the relationships between the input sentence and the output.

![img](d71.png)

![img](d72.png)

![img](d73.png)

...and these two sentences have completely opposite meanings.

**So it is super important for the Decoder to keep track of the significant words in the input.**

The nice thing, is that all we have to do to add this ability to our **Decoder-only Transformer** is just include the prompt when we do **Masked Self-Attention** while generating the output.

![img](d77.png)

![img](d78.png)

![img](d79.png)

...and run everything through the Softmax function.

![img](d81.png)

![img](d82.png)

![img](d83.png)

![img](d84.png)

![img](d85.png)

![img](d86.png)

![img](d87.png)

...and the Softmax function we used before.

![img](d89.png)

![img](d90.png)

![img](d91.png)

![img](d92.png)

Now we calculate the **Masked Self-Attention** values.

![img](d94.png)

![img](d95.png)

![img](d96.png)

![img](d97.png)

![img](d98.png)

...and the Softmax function we used before.

![img](d100.png)