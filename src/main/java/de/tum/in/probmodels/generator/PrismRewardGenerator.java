package de.tum.in.probmodels.generator;

import de.tum.in.probmodels.util.PrismWrappedException;
import javax.annotation.Nullable;
import parser.State;
import prism.ModelGenerator;
import prism.PrismException;

public class PrismRewardGenerator implements RewardGenerator<State> {
  private final int rewardIndex;
  private final ModelGenerator generator;

  PrismRewardGenerator(int rewardIndex, ModelGenerator generator) {
    this.rewardIndex = rewardIndex;
    this.generator = generator;
  }

  @Override
  public double stateReward(State state) {
    try {
      return generator.getStateReward(rewardIndex, state);
    } catch (PrismException e) {
      throw new PrismWrappedException(e);
    }
  }

  @Override
  public double transitionReward(State state, @Nullable Object label) {
    try {
      // Note: getStateActionReward expects to receive null as label in some cases
      return generator.getStateActionReward(rewardIndex, state, label);
    } catch (PrismException e) {
      throw new PrismWrappedException(e);
    }
  }
}
