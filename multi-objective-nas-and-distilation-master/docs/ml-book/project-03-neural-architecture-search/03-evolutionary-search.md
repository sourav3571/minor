# 3. From random search to evolution

An evolutionary optimizer maintains candidate configurations (a population), measures fitness, chooses parents, mutates/crosses configurations to produce children, and keeps useful candidates across generations. For multi-objective fitness, NSGA-II sorts candidates into non-dominated fronts and uses crowding distance to preserve diversity. Project 4 builds the dominance calculation; Project 5 provides a small transparent evolutionary loop.

Random search is the baseline, not a strawman. Evolution adds tuning choices: population size, number of generations, mutation probabilities, duplicate handling, and fitness evaluation. Compare methods with the same total candidate-training budget. Otherwise an evolutionary search may appear better simply because it evaluated more models.

Training candidates from scratch is costly. Successive halving/early stopping, weight sharing, predictors, or one-shot supernets can reduce cost, but each introduces assumptions that must be independently checked. Start with a small transparent search, log all attempts, and measure overhead before claiming efficiency.
