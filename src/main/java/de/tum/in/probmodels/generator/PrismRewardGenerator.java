package de.tum.in.probmodels.generator;

import de.tum.in.probmodels.util.PrismWrappedException;
import javax.annotation.Nullable;
import parser.State;
import prism.ModelGenerator;
import prism.PrismException;

public class PrismRewardGenerator implements RewardGenerator<State> {
  private final int rewardIndex;
  private final ModelGenerator generator;

  public PrismRewardGenerator(int rewardIndex, ModelGenerator generator) {
    this.rewardIndex = rewardIndex;
    this.generator = generator;
  }

  @Override
  public double stateReward(State state) {
    try {
      double reward = generator.getStateReward(rewardIndex, state);
      if (Double.isNaN(reward)) {
        throw new IllegalStateException("NaN reward on state " + state);
      }
      return reward;
    } catch (PrismException e) {
      throw new PrismWrappedException(e);
    }
  }

  @Override
  public double transitionReward(State state, @Nullable Object label) {
    try {
      // Note: getStateActionReward expects to receive null as label in some cases
      double reward = generator.getStateActionReward(rewardIndex, state, label);
      if (Double.isNaN(reward)) {
        String message = String.format("NaN reward on state %s / transition %s", state, label);
        throw new IllegalStateException(message);
      }
      return reward;
    } catch (PrismException e) {
      throw new PrismWrappedException(e);
    }
  }
}
