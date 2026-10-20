# The basic Transformer

So these two phrases...

**'Squatch eats pizza!!!** and ***Pizza eats 'Squatch!!!**

... use the exact same words, but have very different meanings.

So keeping track of word order is super important.

We'll start by showing how to add **Positional Encoding** to the phrase, **'Squatch eats pizza!!!**

**Note:** There are a bunch of ways to do **Postional Encoding**, but we're going to talk about one popular method.

![img](t5.png)

![img](t6.png)

![img](t7.png)

![img](t8.png)

![img](t9.png)

![img](t10.png)

![img](t11.png)

![img](t12.png)

![img](t13.png)

![img](t14.png)

![img](t15.png)

![img](t16.png)

![img](t17.png)

![img](t18.png)

... and we end up with word embeddings plus **Positional Encoding** for the whole sentence...

![img](t20.png)

Thus, **Positional Encoding** allows a Transformer to keep track of word order.

![img](t22.png)

...let's talk about how a Transformer keeps track of the relationships among words.

For example, if the input sentence was this...

![img](t25.png)

![img](t26.png)

![img](t27.png)

**The good news is that Transformers have something called Self-Attention, which is a mechanism to correctly associate the word 'it' with 'pizza'. In general, Self-Attention works by seeing how similar each word is to all the words in the sentence, including itself.**

![img](t30.png)

![img](t31.png)

![img](t32.png)

![img](t33.png)

![img](t34.png)

![img](t35.png)

![img](t36.png)

![img](t37.png)

![img](t38.png)

![img](t39.png)

![img](t40.png)

![img](t41.png)

![img](t42.png)

![img](t43.png)

![img](t44.png)

**One way to calculate similarities between the Query and the Keys is to calculate something called a Dot Product.**

![img](t46.png)

**...tells us Let's is much more similar to itself than it is to the word Go.**

![img](t48.png)

![img](t49.png)

![img](t51.png)

![img](t52.png)

![img](t53.png)

![img](t54.png)

![img](t55.png)

![img](t56.png)

![img](t57.png)

![img](t58.png)

![img](t59.png)

![img](t60.png)

![img](t61.png)

**First, the new Self-Attention values for each word contain input from all of the other words, and this helps give each word context**.

![img](t64.png)

![img](t65.png)

![img](t66.png)

And that is all we need to do encode the input for this simple Transformer.

![img](t68.png)

![img](t70.png)

![img](t71.png)

**However, this time we create embedding values for the output vocabulary, which consists of the Spanish words...** 

![img](t73.png)

![img](t74.png)

![img](t75.png)

![img](t76.png)

![img](t77.png)

![img](t78.png)

![img](t79.png)

**Note: The sets of Weights we used to calculate the Decoder's Self-Attention Query, Key and Value are different from the sets we used in the Encoder.**

![img](t81.png)

**Now, so far we have talked about how Self-Attention helps the Transformer keep track of how words are related within a sentence. However, since we're translating a sentence, we need to keep track of the relationships between the input sentence and the output.**

![img](t84.png)

![img](t85.png)

![img](t86.png)

...and these two sentences have completely opposite meanings.

**So it is super important for the Decoder to keep track of the significant words in the input.**

**So, the main idea of Encoder-Decoder Attention is to allow the Decoder to keep track of the significant words in the input.**

![img](t90.png)

![img](t91.png)

![img](t92.png)

![img](t93.png)

![img](t94.png)

![img](t95.png)

Now that we know what percentage of each input word to use when determining what should be the first translated word...

...we calculate **Values** for each input word.

![img](t98.png)

![img](t99.png)

![img](t100.png)

**Note: The weights we use to calculate the Queries, Keys and Values for Encoder-Decoder Attention are different from the sets of Weights we use for Self-Attention.** 

![img](t102.png)

![img](t103.png)

![img](t104.png)

![img](t105.png)

![img](t106.png)

![img](t107.png)

First, we get the **Word Embeddings** for **vamos**...

...then we add the **Positional Encoding**. 

![img](t110.png)

![img](t111.png)

![img](t112.png)

![img](t113.png)

![img](t114.png)

![img](t115.png)

![img](t116.png)

![img](t117.png)

![img](t118.png)