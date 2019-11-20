package de.tum.in.probmodels.model;

import static com.google.common.base.Preconditions.checkArgument;

import de.tum.in.naturals.set.BoundedNatBitSet;
import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.naturals.set.NatBitSets;
import de.tum.in.probmodels.util.annotation.Tuple;
import explicit.rewards.MDPRewards;
import explicit.rewards.MDPRewardsSimple;
import it.unimi.dsi.fastutil.ints.IntIterator;
import it.unimi.dsi.fastutil.ints.IntSet;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;
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

  public static <T extends Model> RestrictedModel<T> buildRestrictedModel(T model,
      T newModel, IntSet states, boolean omitSelfLoops) {
    checkArgument(newModel.getNumStates() == 0);
    int[] originalToRestrictedStates = new int[model.getNumStates()];
    Arrays.fill(originalToRestrictedStates, -1);
    states.forEach((int allowedState) ->
        originalToRestrictedStates[allowedState] = newModel.addState());

    int restrictedStates = newModel.getNumStates();
    assert restrictedStates == states.size();
    int[] restrictedToOriginalStates = new int[restrictedStates];
    NatBitSet[] restrictedActions = new NatBitSet[restrictedStates];

    for (int originalState = 0; originalState < model.getNumStates(); originalState++) {
      if (originalToRestrictedStates[originalState] == -1) {
        continue;
      }
      int restrictedState = originalToRestrictedStates[originalState];
      restrictedToOriginalStates[restrictedState] = originalState;

      List<Action> actions = model.getActions(originalState);
      BoundedNatBitSet removedActions = NatBitSets.boundedSet(actions.size(), 0);
      ListIterator<Action> iterator = actions.listIterator();
      while (iterator.hasNext()) {
        int index = iterator.nextIndex();
        Action action = iterator.next();

        Distribution distribution = new Distribution();
        int source = originalState;
        action.distribution().forEach((target, probability) -> {
          int restrictedDestination = originalToRestrictedStates[target];
          if (!(restrictedDestination < 0 || (omitSelfLoops && target == source))) {
            distribution.add(restrictedDestination, probability);
          }
        });
        if (distribution.isEmpty()) {
          removedActions.set(index);
        } else {
          model.addChoice(restrictedState, Action.of(distribution.scale(), action.label()));
        }
      }
      restrictedActions[restrictedState] = NatBitSets.compact(removedActions.complement());
      /* if (addedActions == 0) {
        newModel.addDeadlockState(restrictedState);
      } */
    }
    model.getInitialStates().forEach((int initialState) -> {
      int restrictedInitialState = originalToRestrictedStates[initialState];
      if (restrictedInitialState != -1) {
        newModel.addInitialState(restrictedInitialState);
      }
    });

    IntUnaryOperator stateMapping = i -> restrictedToOriginalStates[i];
    IntFunction<NatBitSet> stateActions = i -> restrictedActions[i];
    return RestrictedModelTuple.create(newModel, stateMapping, stateActions);
  }
}
