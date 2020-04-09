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
  public final NatBitSet states;
  public final Int2ObjectMap<IntSet> actions;

  private Mec(NatBitSet s, Int2ObjectMap<IntSet> a) {
    states = s;
    actions = a;
  }

  public static Mec create(Model model, NatBitSet states) {
    Int2ObjectMap<IntSet> actions = new Int2ObjectOpenHashMap<>(states.size());
    boolean changed = true;

    while (changed) {
      changed = false;

      IntIterator stateIterator = states.iterator();
      NatBitSet toRemove = NatBitSets.set();
      while (stateIterator.hasNext()) {
        int state = stateIterator.nextInt();

        List<Distribution> distributions = model.getChoices(state);
        int choiceCount = distributions.size();
        if (choiceCount == 0) {
          toRemove.set(state);
          changed = true;
          continue;
        }

        IntFunction<IntSet> constructor = k -> NatBitSets.boundedFilledSet(choiceCount);
        IntSet stateActions = actions.computeIfAbsent(state, constructor);
        NatBitSet removeActions = NatBitSets.boundedSet(choiceCount);

        stateActions.forEach((int action) -> {
          if (!states.containsAll(distributions.get(action).support())) {
            removeActions.set(action);
          }
        });
        changed |= stateActions.removeAll(removeActions);

        if (stateActions.isEmpty()) {
          toRemove.set(state);
          stateActions.remove(state);
          changed = true;
        }
      }
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
