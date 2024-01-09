# PySuperTuxKart AI Training Project

## Overview

PySuperTuxKart is a kart racing game adapted for AI training in sensor-based tasks, focusing on competitive AI development in the 2v2 Supertux soccer match. This repository documents our approach, challenges faced, and findings while creating a model competitive against other AI opponents.

## Motivation

Our primary goals were to explore the effectiveness of combining imitation learning and self-play in creating a competitive AI. Specifically, we aimed to investigate:

- The synergy between self-play and imitation learning for robust results.
- Identification of model architectures conducive to learning in a self-play environment.

## Methodology

### Environment Approach and Challenges

We developed a custom gym environment mirroring the game's Match class to generate both off-policy and on-policy trajectories. However, limitations in parallelism significantly impacted self-play speed.

### Model Approaches and Challenges

#### Imitation with DAgger
- Attempted imitating the high-performing Jurgen model using DAgger.
- Challenges in correcting trajectory deviations; the model struggled to learn from Jurgen's actions effectively.

#### Imitation using LSTM
- Implemented imitation learning with an LSTM model.
- Struggled with latency issues, overfitting, and inadequate model performance despite adjustments in input sequences and model architecture.

#### Imitation with A2C
- Utilized imitation learning followed by A2C training.
- Challenges with brake activation convergence and reward sparsity; slow training cycle due to bugs impacted the model's competitiveness.

### Results

Our DAgger-based model scored 91/100 on the local grader after refining using DAgger. However, other models failed to achieve competitive baselines.

## Conclusion

Despite limitations in achieving our intended results, the project provided invaluable insights into AI model development. Simplifying model complexity and prioritizing the data pipeline could yield more effective results.

## Next Steps

- Explore advanced reinforcement learning methodologies like PPO.
- Enhance the input features of the best agent.
- Improve self-play environment, parallelize training, and optimize training loops.
- Consider alternate approaches like Actor-Critic methods or evolutionary computation strategies for faster iterations.
