# Nov 2020. [Charles Isbell- Computing, Interactive AI, and Race in America | Lex Fridman Podcast no. 135](https://www.youtube.com/watch?v=LAyZ8IYfGxQ)
1 Nov 2020.

## Interactive AI

Isbell's research area of **interactive AI** is about building agents that have to operate *among* other intelligent agents (including humans) rather than solving an isolated prediction or control problem against a fixed, static environment. In his own framing, the goal is understanding how to build autonomous agents that must live and interact with large numbers of other intelligent agents, some of whom may be human.

A few things distinguish this from "plain" machine learning or reinforcement learning:

- **The environment talks back:** Standard supervised learning assumes a fixed data distribution, and standard RL usually assumes stationary environment dynamics. In an interactive-AI setting, the other agents (people, other bots) are adapting to *you* while you adapt to *them*, so it's a moving target, closer to a social negotiation than a static optimization problem.
- **Reward signals come from social feedback, not clean labels:** A core thread in his work is balancing multiple sources of reward in social environments— reward that comes from how people actually react to an agent, which is noisy, delayed, and sometimes contradictory.

His signature example is **Cobot**, a reinforcement-learning social agent he built and deployed in **LambdaMOO** (a long-running text-based virtual community) in the late 1990s/early 2000s. Cobot lived among real human users for years, and its reward signal came from how the community actually treated it (being talked to, ignored, told off, and so on), making it one of the earliest real-world, long-duration RL deployments among actual people rather than in a simulator.

Other threads under the same umbrella from his lab: scalable coordination between multiple agents, discovering structure and activities from unstructured interaction data, and adaptive collaborative tools— all cases where the AI problem is inseparable from the humans or other agents it's embedded with.

This area is explicitly recognized in his honors: both his ACM Fellowship and AAAI Fellowship citations credit him specifically for contributions to interactive AI.

**Sources:**

- [Charles Isbell's faculty page](https://faculty.cc.gatech.edu/~isbell/DrIsbell.html)
- [Charles Lee Isbell Jr.— Wikipedia](https://en.wikipedia.org/wiki/Charles_Lee_Isbell_Jr.)
- [Cobot: A Social Reinforcement Learning Agent (NeurIPS 2001 paper)](http://papers.neurips.cc/paper/2118-cobot-a-social-reinforcement-learning-agent.pdf)
- [Interactive AI, Plus Improving ML Education— TWIML AI Podcast with Charles Isbell](https://twimlai.com/podcast/twimlai/interactive-ai-plus-improving-ml-education-charles-isbell/)
