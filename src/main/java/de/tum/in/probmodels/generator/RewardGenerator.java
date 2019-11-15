package de.tum.in.probmodels.generator;

import javax.annotation.Nullable;

public interface RewardGenerator<S> {
  double stateReward(S state);

  double transitionReward(S state, @Nullable Object transitionIdentifier);
}
