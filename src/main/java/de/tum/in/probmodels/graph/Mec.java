package de.tum.in.probmodels.graph;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.naturals.set.NatBitSets;
import de.tum.in.probmodels.model.Distribution;
import de.tum.in.probmodels.model.Model;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntIterator;
import it.unimi.dsi.fastutil.ints.IntSet;
import java.util.List;
import java.util.Objects;
import java.util.function.IntFunction;

public final class Mec {
  public final NatBitSet states; // Set of state numbers corresponding to the original model that belong to the mec
  public final Int2ObjectMap<IntSet> actions; // Map from state to a set of indices. These indices give a subset of state's list of actions that can be obtained from the original model.
  // This subset gives a set of actions that are a part of the mec.

  private Mec(NatBitSet s, Int2ObjectMap<IntSet> a) {
    states = s;
    actions = a;
  }

  // Creates and returns an mec object from the given model and the given set of states.
  public static Mec create(Model model, NatBitSet states) {
    Int2ObjectMap<IntSet> actions = new Int2ObjectOpenHashMap<>(states.size());
    boolean changed = true;

    // This loop iteratively removes actions that may point to a state outside the mec.
    // Further it removes states that may not have a valid action. (All actions for a state may lie outside the given set of states.
    while (changed) {
      changed = false;

      IntIterator stateIterator = states.iterator();
      NatBitSet toRemove = NatBitSets.set();
      while (stateIterator.hasNext()) {
        int state = stateIterator.nextInt();

        List<Distribution> distributions = model.getChoices(state);
        int choiceCount = distributions.size();
        // If there are no actions from a state, remove the state from the mec
        if (choiceCount == 0) {
          toRemove.set(state);
          changed = true;
          continue;
        }

        IntFunction<IntSet> constructor = k -> NatBitSets.boundedFilledSet(choiceCount);
        // Indices of all actions from a state. If actions map already has a value, that is returned, otherwise all indices from 0 to choiceCount-1 are returned.
        IntSet stateActions = actions.computeIfAbsent(state, constructor);
        // Indices of actions that are to be removed (Empty for now)
        NatBitSet removeActions = NatBitSets.boundedSet(choiceCount);

        // Remove an action if it's support consists of state outside the remaining states
        stateActions.forEach((int action) -> {
          if (!states.containsAll(distributions.get(action).support())) {
            removeActions.set(action);
          }
        });
        //  changed set to true there is a removed action
        changed |= stateActions.removeAll(removeActions);

        // if all of the state's actions are removed, remove the state from the mec
        if (stateActions.isEmpty()) {
          toRemove.set(state);
          stateActions.remove(state);
          changed = true;
        }
      }
      // Do the remove operation
      states.andNot(toRemove);
    }

    return new Mec(states, actions);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    Mec other = (Mec) o;
    return Objects.equals(states, other.states);
  }

  @Override
  public String toString() {
    return states.toString();
  }

  @Override
  public int hashCode() {
    return states.hashCode() * 31;
  }

  public int size() {
    return states.size();
  }
}
