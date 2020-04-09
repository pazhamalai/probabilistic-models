package de.tum.in.probmodels.model;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import explicit.DTMC;
import explicit.SuccessorsIterator;
import it.unimi.dsi.fastutil.ints.IntIterator;
import it.unimi.dsi.fastutil.ints.IntIterators;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import prism.ModelType;
import prism.Pair;

public class MarkovChain extends DefaultModel implements DTMC {
  private final Map<Integer, Distribution> transitions = new HashMap<>();
  private int numTransitions = 0;

  @Override
  public void clearState(int i) {
    Distribution distribution = transitions.remove(i);
    if (distribution != null) {
      numTransitions -= distribution.size();
    }
  }

  @Override
  public ModelType getModelType() {
    return ModelType.DTMC;
  }

  @Override
  public int getNumTransitions() {
    return numTransitions;
  }

  @Override
  public IntIterator getSuccessorsIterator(int s) {
    Distribution distribution = transitions.get(s);
    if (distribution == null) {
      return IntIterators.EMPTY_ITERATOR;
    }
    return distribution.support().iterator();
  }

  @Override
  public SuccessorsIterator getSuccessors(int s) {
    return SuccessorsIterator.from(getSuccessorsIterator(s), true);
  }

  @Override
  public boolean isSuccessor(int s1, int s2) {
    Distribution distribution = transitions.get(s1);
    if (distribution == null) {
      return false;
    }
    return distribution.contains(s2);
  }

  @Override
  public boolean allSuccessorsInSet(int s, BitSet set) {
    Distribution distribution = transitions.get(s);
    if (distribution == null) {
      return true;
    }
    return distribution.isSubsetOf(set);
  }

  @Override
  public boolean someSuccessorsInSet(int s, BitSet set) {
    Distribution distribution = transitions.get(s);
    if (distribution == null) {
      return false;
    }
    return distribution.containsOneOf(set);
  }

  @Override
  public int getNumTransitions(int s) {
    Distribution distribution = transitions.get(s);
    return distribution == null ? 0 : distribution.size();
  }

  @Override
  @Deprecated
  public Iterator<Map.Entry<Integer, Double>> getTransitionsIterator(int s) {
    Distribution distribution = transitions.get(s);
    return distribution == null ? Collections.emptyIterator() : distribution.objectIterator();
  }

  @Override
  public Iterator<Map.Entry<Integer, Pair<Double, Object>>>
  getTransitionsAndActionsIterator(int s) {
    throw new UnsupportedOperationException();
  }

  /**
   * Get the transitions (a distribution) for state s.
   */
  public Distribution getTransitions(int s) {
    return transitions.get(s);
  }

  @Override
  public List<Distribution> getChoices(int state) {
    Distribution transitions = getTransitions(state);
    return transitions == null ? Collections.emptyList() : Collections.singletonList(transitions);
  }

  @Override
  public void forEachChoice(int state, Consumer<Distribution> action) {
    Distribution transitions = getTransitions(state);
    if (transitions != null) {
      action.accept(transitions);
    }
  }

  @Override
  public void addChoice(int state, Distribution distribution) {
    Distribution oldValue = transitions.put(state, checkNotNull(distribution));
    checkArgument(oldValue == null, "MarkovChain can only have one distribution");
    numTransitions += distribution.size();
  }

  @Override
  public void addChoice(int state, Action action) {
    checkArgument(action.label() == null);
    addChoice(state, action.distribution());
  }

  @Override
  public void setChoice(int state, int action, Distribution distribution) {
    checkArgument(action == 0);
    Distribution oldValue = transitions.put(state, checkNotNull(distribution));
    if (oldValue != null) {
      numTransitions -= oldValue.size();
    }
    numTransitions += distribution.size();
  }

  @Override
  public Distribution getChoice(int state, int action) {
    checkArgument(action == 0);
    return getTransitions(state);
  }

  @Override
  public void setActions(int state, List<Action> actions) {
    checkArgument(actions.size() <= 1);
    if (actions.isEmpty()) {
      clearState(state);
    } else {
      Action action = actions.get(0);
      checkArgument(action.label() == null);
      setChoice(state, 0, action.distribution());
    }
  }

  @Override
  public List<Action> getActions(int state) {
    Distribution distribution = transitions.get(state);
    return distribution == null
        ? Collections.emptyList()
        : Collections.singletonList(Action.of(distribution));
  }

  @Override
  public int getNumChoices(int state) {
    return transitions.containsKey(state) ? 1 : 0;
  }
}
