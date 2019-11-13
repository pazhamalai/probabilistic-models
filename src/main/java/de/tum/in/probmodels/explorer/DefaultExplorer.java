package de.tum.in.probmodels.explorer;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.naturals.set.NatBitSets;
import de.tum.in.probmodels.generator.Choice;
import de.tum.in.probmodels.generator.Generator;
import de.tum.in.probmodels.model.Action;
import de.tum.in.probmodels.model.Distribution;
import de.tum.in.probmodels.model.Model;
import de.tum.in.probmodels.model.StateToIndex;
import de.tum.in.probmodels.util.Util;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntCollection;
import it.unimi.dsi.fastutil.ints.IntList;
import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import java.util.List;
import prism.PrismException;

public class DefaultExplorer<S, M extends Model> implements Explorer<S, M> {
  private final StateToIndex<S> stateMap = new StateToIndex<>();
  // All states which are in the partial model and explored
  private final NatBitSet exploredStates = NatBitSets.set();
  private final M model;
  private final Generator<S> generator;

  public DefaultExplorer(M model, Generator<S> generator) throws PrismException {
    this.model = model;
    this.generator = generator;
    IntList initialStateIds = new IntArrayList();
    for (S initialState : generator.initialStates()) {
      int stateId = getStateId(initialState);
      exploreState(stateId);
      initialStateIds.add(stateId);
    }
    model.setInitialStates(initialStateIds);
  }

  @Override
  public NatBitSet exploredStates() {
    return exploredStates;
  }

  @Override
  public M model() {
    return model;
  }

  @Override
  public S exploreState(int stateId) throws PrismException {
    assert stateMap.check(stateId) && !isExploredState(stateId);
    exploredStates.set(stateId);

    S state = stateMap.getState(stateId);
    assert state != null;

    for (Choice<S> choice : generator.choices(state)) {
      Distribution distribution = new Distribution();
      for (Object2DoubleMap.Entry<S> transition : choice.transitions().object2DoubleEntrySet()) {
        int target = getStateId(transition.getKey());
        double probability = transition.getDoubleValue();
        distribution.add(target, probability);
      }
      assert distribution.isEmpty() || Util.isOne(distribution.sum()) : distribution;

      model.addChoice(stateId, Action.of(distribution, choice.label()));
    }
    return state;
  }

  @Override
  public boolean isExploredState(int stateId) {
    return exploredStates.contains(stateId);
  }

  @Override
  public final int getStateId(S state) {
    assert state != null;

    int stateId = stateMap.getStateId(state);
    if (stateId != -1) {
      return stateId;
    }

    int newStateId = model.addState();
    assert newStateId == stateMap.size();
    stateMap.addState(state, newStateId);
    return newStateId;
  }

  @Override
  public IntCollection initialStates() {
    return model.getInitialStates();
  }

  @Override
  public int exploredStateCount() {
    return exploredStates.size();
  }

  @Override
  public List<Distribution> getChoices(int stateId) {
    assert isExploredState(stateId);
    return model.getChoices(stateId);
  }

  @Override
  public List<Action> getActions(int stateId) {
    assert isExploredState(stateId);
    return model.getActions(stateId);
  }

  @Override
  public S getState(int stateId) {
    return stateMap.getState(stateId);
  }
}
