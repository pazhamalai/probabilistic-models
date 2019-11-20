package de.tum.in.probmodels.model;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.probmodels.util.annotation.Tuple;
import explicit.rewards.MDPRewards;
import explicit.rewards.MDPRewardsSimple;
import it.unimi.dsi.fastutil.ints.IntIterator;
import java.util.function.IntFunction;
import java.util.function.IntUnaryOperator;
import org.immutables.value.Value;

@Value.Immutable
@Tuple
public abstract class RestrictedModel<T extends Model> {
  public abstract T model();

  public abstract IntUnaryOperator stateMapping();

  public abstract IntFunction<NatBitSet> stateActions();

  public MDPRewards buildMdpRewards(MDPRewards originalRewards) {
    int states = model().getNumStates();

    MDPRewardsSimple rewards = new MDPRewardsSimple(states);
    for (int stateNumber = 0; stateNumber < states; stateNumber++) {
      int originalState = stateMapping().applyAsInt(stateNumber);
      double stateReward = originalRewards.getStateReward(originalState);
      rewards.setStateReward(stateNumber, stateReward);

      NatBitSet actions = stateActions().apply(stateNumber);
      int index = 0;
      IntIterator iterator = actions.iterator();
      while (iterator.hasNext()) {
        int originalIndex = iterator.nextInt();

        double transitionReward = originalRewards.getTransitionReward(stateNumber, originalIndex);
        rewards.setTransitionReward(stateNumber, index, transitionReward);
        index += 1;
      }
    }

    return rewards;
  }
}
