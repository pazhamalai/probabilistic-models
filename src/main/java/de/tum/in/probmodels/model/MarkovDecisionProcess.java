package de.tum.in.probmodels.model;

import com.google.common.collect.Lists;
import de.tum.in.probmodels.model.distribution.Distribution;
import explicit.NondetModelSimple;
import explicit.SuccessorsIterator;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import prism.ModelType;
import prism.PrismLog;
import strat.MDStrategy;

public class MarkovDecisionProcess extends DefaultModel
    implements explicit.MDP, NondetModelSimple {
  private final Map<Integer, List<Action>> transitions = new HashMap<>();
  private int numTransitions = 0;

  private List<Action> getTransitions(int state) {
    return transitions.computeIfAbsent(state, k -> new ArrayList<>());
  }

  @Override
  public void clearState(int s) {
    transitions.remove(s);
  }

  @Override
  public void addChoice(int s, Distribution distribution) {
    assert s < getNumStates();
    List<Action> distributions = getTransitions(s);
    distributions.add(Action.of(distribution));
    numTransitions += 1;
  }

  @Override
  public void addChoice(int state, Action action) {
    assert state < getNumStates();
    List<Action> distributions = getTransitions(state);
    distributions.add(action);
    numTransitions += 1;
  }

  @Override
  public ModelType getModelType() {
    return null;
  }

  @Override
  public int getNumTransitions() {
    return numTransitions;
  }

  @Override
  public int getNumChoices(int s) {
    assert s < getNumStates();
    return transitions.getOrDefault(s, Collections.emptyList()).size();
  }

  @Override
  public int getMaxNumChoices() {
    return transitions.values().stream().mapToInt(List::size).max().orElse(-1);
  }

  @Override
  public int getNumChoices() {
    return transitions.values().stream().mapToInt(List::size).sum();
  }

  @Override
  public Object getAction(int s, int i) {
    return transitions.get(s).get(i).label();
  }

  @Override
  public boolean areAllChoiceActionsUnique() {
    return false;
  }

  @Override
  public boolean allSuccessorsInSet(int s, int i, BitSet set) {
    return transitions.get(s).get(i).distribution().isSubsetOf(set);
  }

  @Override
  public boolean someSuccessorsInSet(int s, int i, BitSet set) {
    return transitions.get(s).get(i).distribution().containsOneOf(set);
  }

  @Override
  public Iterator<Integer> getSuccessorsIterator(int s, int i) {
    return transitions.get(s).get(i).distribution().support().iterator();
  }

  @Override
  public SuccessorsIterator getSuccessors(int s, int i) {
    return SuccessorsIterator.from(getSuccessorsIterator(s, i), true);
  }

  @Override
  public explicit.Model constructInducedModel(MDStrategy strategy) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void exportToDotFileWithStrat(PrismLog out, BitSet mark, int[] strategy) {
    throw new UnsupportedOperationException();
  }

  @Override
  public int getNumTransitions(int s, int i) {
    return transitions.get(s).get(i).distribution().size();
  }

  @Override
  @Deprecated
  public Iterator<Map.Entry<Integer, Double>> getTransitionsIterator(int s, int i) {
    return transitions.get(s).get(i).distribution().objectIterator();
  }

  @Override
  public Distribution getChoice(int state, int action) {
    return transitions.get(state).get(action).distribution();
  }

  @Override
  public void setActions(int state, List<Action> actions) {
    transitions.put(state, actions);
  }

  @Override
  public List<Action> getActions(int state) {
    return transitions.getOrDefault(state, Collections.emptyList());
  }

  @Override
  public void setChoice(int state, int action, Distribution distribution) {
    transitions.get(state).set(action, Action.of(distribution));
  }

  @Override
  public List<Distribution> getChoices(int state) {
    List<Action> actions = transitions.get(state);
    if (actions == null) {
      return Collections.emptyList();
    }
    return Lists.transform(actions, Action::distribution);
  }

  @Override
  public void forEachChoice(int state, Consumer<Distribution> action) {
    List<Action> actions = transitions.get(state);
    if (actions == null) {
      return;
    }
    actions.forEach(a -> action.accept(a.distribution()));
  }
}
