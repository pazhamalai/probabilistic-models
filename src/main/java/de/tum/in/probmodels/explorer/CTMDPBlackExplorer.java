package de.tum.in.probmodels.explorer;

import de.tum.in.probmodels.generator.Choice;
import de.tum.in.probmodels.generator.Generator;
import de.tum.in.probmodels.model.*;
import de.tum.in.probmodels.util.Sample;
import de.tum.in.probmodels.util.Util;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import it.unimi.dsi.fastutil.ints.*;
import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectArrayList;

import java.util.List;

public class CTMDPBlackExplorer<S, M extends Model> extends BlackExplorer<S, M>{

  private final Int2ObjectMap<ObjectArrayList<Int2DoubleMap>> stateTransitionRates = new Int2ObjectOpenHashMap<>();

  public final Int2ObjectMap<Int2ObjectMap<DoubleArrayList>> transitionTimes = new Int2ObjectOpenHashMap<>();

  public static <S, M extends Model> CTMDPBlackExplorer<S, M> of(M model, Generator<S> generator,
                                                                 boolean removeSelfLoops) {
    CTMDPBlackExplorer<S, M> explorer = new CTMDPBlackExplorer<>(model, generator, removeSelfLoops);
    IntList initialStateIds = new IntArrayList();
    for (S initialState : generator.initialStates()) {
      int stateId = explorer.getStateId(initialState);
      explorer.exploreState(stateId);
      initialStateIds.add(stateId);
    }
    model.setInitialStates(initialStateIds);
    return explorer;
  }

  CTMDPBlackExplorer(M model, Generator<S> generator, boolean removeSelfLoops) {
    super(model, generator, removeSelfLoops);
  }

  @Override
  public M model() {
    // Write uniformization code here.
    return model;
  }

  /**
   * Update sampled counts for a state-action-successor triplet. If update is true, the learned distributions are
   * immediately updated. Returns whether a new action has been sampled more than actionCountFilter number of times.
   * @param state
   * @param actionIndex
   * @param successor
   * @param update
   */
  public boolean updateCounts(int state, int actionIndex, int successor, boolean update){
    if (actionCountFilterActive) {
      actionIndex = unfilteredActionIndexMap.get(state).get(actionIndex);
    }
    Int2LongMap transitionCounts = stateTransitionCounts.get(state).get(actionIndex);
    transitionCounts.put(successor, transitionCounts.getOrDefault(successor, 0)+1);

    boolean newTrans = false;

    if (transitionCounts.get(successor)==1){
      numTrans++;
    }

    long actionCount = getActionCounts(state, actionIndex);
    if(actionCount>actionCountFilter && actionCount-1<=actionCountFilter){
      newTrans = true;
    }

    if (update) {
      List<Action> currActions = model.getActions(state);
      Action currAction = currActions.get(actionIndex);

      Distribution distribution = getDistributionFromCounts(state, transitionCounts);
      currActions.set(actionIndex, Action.of(distribution, currAction.label()));

      model.setActions(state, currActions);
    }

    stateActionChange.get(state).put(actionIndex, !update);

    double stayTime = Sample.sampleExponential(stateTransitionRates.get(state).get(actionIndex).values().stream().reduce(0d, Double::sum));
    transitionTimes.get(state).get(actionIndex).add(stayTime);

    return newTrans;
  }

  @Override
  public void simulateActionRepeatedly(int stateId, int filteredIndex, double requiredSamples){
    int realIndex = filteredIndex;
    if (actionCountFilterActive) {
      realIndex = unfilteredActionIndexMap.get(stateId).get(filteredIndex);
    }
    Action action = stateActions.get(stateId).get(realIndex);
    long actionCounts = getActionCounts(stateId, realIndex);
    Int2IntMap actionTransitionCounts = new Int2IntOpenHashMap();
    Int2ObjectMap<DoubleArrayList> stateTransitionTimes = transitionTimes.get(stateId);
    for(int succ: action.distribution().support()) {
      actionTransitionCounts.put(succ, 0);
    }
    while (actionCounts<requiredSamples) {
      int succ = action.distribution().sample();
      actionTransitionCounts.put(succ, actionTransitionCounts.get(succ)+1);
      double stayTime = Sample.sampleExponential(stateTransitionRates.get(stateId).get(realIndex).get(succ));
      stateTransitionTimes.get(realIndex).add(stayTime);
      actionCounts++;
    }
    for(int succ: action.distribution().support()) {
      long currValue = stateTransitionCounts.get(stateId).get(realIndex).get(succ);
      stateTransitionCounts.get(stateId).get(realIndex)
          .put(succ, currValue+actionTransitionCounts.get(succ));
    }
    List<Action> currActions = model.getActions(stateId);
    Distribution distribution = getDistributionFromCounts(stateId, stateTransitionCounts.get(stateId).get(realIndex));
    currActions.set(filteredIndex, Action.of(distribution, action.label()));

    model.setActions(stateId, currActions);
  }

  @Override
  public S exploreState(int stateId) {
    assert stateMap.check(stateId) && !isExploredState(stateId);
    exploredStates.add(stateId);

    // adds a state into the partial model
    S state = stateMap.getState(stateId);
    assert state != null;

    ObjectArrayList<Int2LongMap> stateActionCounts = new ObjectArrayList<>();
    ObjectArrayList<Action> stateChoices = new ObjectArrayList<>();

    Int2ObjectMap<DoubleArrayList> stateTransitionTimes = new Int2ObjectOpenHashMap<>();
    ObjectArrayList<Int2DoubleMap> stateTransitionRates = new ObjectArrayList<>();

    int actionCount = -1;
    for (Choice<S> choice : generator.choices(state)) {
      actionCount++;
      DistributionBuilder builder = Distributions.defaultBuilder();
      Int2DoubleMap rateMap = new Int2DoubleOpenHashMap();

      double rateSum = 0d;
      for (Object2DoubleMap.Entry<S> transition : choice.transitions().object2DoubleEntrySet()) {
        int target = getStateId(transition.getKey());
        double rate = transition.getDoubleValue();
        rateMap.put(target, rate);
        rateSum += rate;
      }

      for (Object2DoubleMap.Entry<S> transition : choice.transitions().object2DoubleEntrySet()) {
        int target = getStateId(transition.getKey());
        double rate = transition.getDoubleValue();
        builder.add(target, rate/rateSum);
      }

      stateTransitionTimes.put(actionCount, new DoubleArrayList());
      // scale the distribution if any values in the original support were skipped
      Distribution distribution = builder.scaled();
      assert distribution.isEmpty() || Util.isOne(distribution.sum()) : distribution;
      // Real distribution added to stateChoices
      stateChoices.add(Action.of(distribution, choice.label()));
      stateTransitionRates.add(rateMap);

      stateActionCounts.add(new Int2LongOpenHashMap());

      // Empty distribution added to model
      DistributionBuilder emptyBuilder = Distributions.defaultBuilder();
      model.addChoice(stateId, Action.of(emptyBuilder.build(), choice.label()));
    }

    stateTransitionCounts.put(stateId, stateActionCounts);
    stateActions.put(stateId, stateChoices);
    stateActionChange.put(stateId, new Int2BooleanOpenHashMap());

    this.transitionTimes.put(stateId, stateTransitionTimes);
    this.stateTransitionRates.put(stateId, stateTransitionRates);

    exploredActionsCount += stateChoices.size();

    return state;
  }
}
