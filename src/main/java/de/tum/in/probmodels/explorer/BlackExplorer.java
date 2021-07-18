package de.tum.in.probmodels.explorer;

import de.tum.in.probmodels.generator.Choice;
import de.tum.in.probmodels.generator.Generator;
import de.tum.in.probmodels.model.*;
import de.tum.in.probmodels.util.Sample;
import de.tum.in.probmodels.util.Util;
import it.unimi.dsi.fastutil.ints.*;
import it.unimi.dsi.fastutil.objects.*;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiPredicate;

public class BlackExplorer<S, M extends Model> implements Explorer<S, M>{
  // A mapping to and from state numbers in partial model to state object in generator.
  private final StateToIndex<S> stateMap = new StateToIndex<>();
  // All states which are in the partial model and explored
  private final IntSet exploredStates = new IntOpenHashSet();
  private final M model;
  private final Generator<S> generator;
  private final boolean removeSelfLoops;

  private final Int2ObjectMap<ObjectArrayList<Int2IntMap>> stateTransitionCounts = new Int2ObjectOpenHashMap<>();
  private final Int2ObjectMap<ObjectArrayList<Action>> stateActions = new Int2ObjectOpenHashMap<>();
  private final Int2ObjectMap<Int2BooleanMap> stateActionChange = new Int2ObjectOpenHashMap<>();

  private final Int2ObjectMap<List<Action>> unfilteredActionsCache = new Int2ObjectOpenHashMap<>();

  private int exploredActionsCount = 0;
  private boolean actionCountFilterActive = false;
  private double actionCountFilter;

  private int numTrans = 0;

  // Creates and returns a default explorer object from a generator. Explores all initial states
  public static <S, M extends Model> BlackExplorer<S, M> of(M model, Generator<S> generator,
      boolean removeSelfLoops) {
    BlackExplorer<S, M> explorer = new BlackExplorer<>(model, generator, removeSelfLoops);
    IntList initialStateIds = new IntArrayList();
    for (S initialState : generator.initialStates()) {
      int stateId = explorer.getStateId(initialState);
      explorer.exploreState(stateId);
      initialStateIds.add(stateId);
    }
    model.setInitialStates(initialStateIds);
    return explorer;
  }

  BlackExplorer(M model, Generator<S> generator, boolean removeSelfLoops) {
    this.model = model;
    this.generator = generator;
    this.removeSelfLoops = removeSelfLoops;
  }

  @Override
  public IntSet exploredStates() {
    return IntSets.unmodifiable(exploredStates);
  }

  @Override
  public M model() {
    return model;
  }

  public int getNumExploredActions(){
    return exploredActionsCount;
  }

  public int getNumTrans(){
    return numTrans;
  }

  public void updateCountParams(double mecConfidence, double pMin){
    this.actionCountFilter = Math.log(mecConfidence)/Math.log(1-pMin);
  }

  public boolean updateCounts(int state, int actionIndex, int successor, boolean update){
    Int2IntMap transitionCounts = stateTransitionCounts.get(state).get(actionIndex);
    transitionCounts.put(successor, transitionCounts.getOrDefault(successor, 0)+1);

    boolean newTrans = false;

    if (transitionCounts.get(successor)==1){
      numTrans++;
    }

    int actionCount = getActionCounts(state, actionIndex);
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

    stateActionChange.get(state).put(actionIndex, true);

    return newTrans;
  }

  public int getActionCounts(int stateId, int actionIndex){
    Int2IntMap transitionCounts = stateTransitionCounts.get(stateId).get(actionIndex);
    return transitionCounts.values().stream().mapToInt(s -> s).sum();
  }

  private Distribution getDistributionFromCounts(int stateId, Int2IntMap transitionCounts){

    double actionCounts = transitionCounts.values().stream().mapToInt(s -> s).sum();

    DistributionBuilder builder = Distributions.defaultBuilder();

    boolean skippedAny = false;
    for (int target : transitionCounts.keySet()) {
      double probability = transitionCounts.get(target) / actionCounts;
      if (removeSelfLoops && target == stateId) {
        skippedAny = true;
      } else {
        builder.add(target, probability);
      }
    }

    // scale the distribution if any values in the original support were skipped
    Distribution distribution = skippedAny ? builder.scaled() : builder.build();
    assert distribution.isEmpty() || Util.isOne(distribution.sum()) : distribution;
    return distribution;

  }

  public void updateModel(BiPredicate<Integer, Integer> changed){

    for (int stateId: exploredStates) {
      List<Action> actionList = new ArrayList<>();
      for (int actionIndex=0; actionIndex<stateTransitionCounts.get(stateId).size(); actionIndex++){
        Action action = model.getActions(stateId).get(actionIndex);
        if (changed.test(stateId, actionIndex)) {

          Int2IntMap transitionCounts = stateTransitionCounts.get(stateId).get(actionIndex);
          Distribution distribution = getDistributionFromCounts(stateId, transitionCounts);
          action = Action.of(distribution, action.label());
        }
        actionList.add(action);
      }
      model.setActions(stateId, actionList);
    }

  }

  public void updateModelCounts(){

    updateModel((stateId, actionIndex) -> stateActionChange.get((int) stateId).get((int) actionIndex));

    for (int key: stateActionChange.keySet()){
      for (int actionIndex=0; actionIndex<stateActionChange.size(); actionIndex++) {
        stateActionChange.get(key).put(actionIndex, false);
      }
    }

  }

  @Override
  public S exploreState(int stateId) {
    assert stateMap.check(stateId) && !isExploredState(stateId);
    exploredStates.add(stateId);

    // adds a state into the partial model
    S state = stateMap.getState(stateId);
    assert state != null;

    ObjectArrayList<Int2IntMap> stateActionCounts = new ObjectArrayList<>();
    ObjectArrayList<Action> stateChoices = new ObjectArrayList<>();

    for (Choice<S> choice : generator.choices(state)) {
      DistributionBuilder builder = Distributions.defaultBuilder();

      boolean skippedAny = false;
      for (Object2DoubleMap.Entry<S> transition : choice.transitions().object2DoubleEntrySet()) {
        int target = getStateId(transition.getKey());
        double probability = transition.getDoubleValue();
        if (removeSelfLoops && target == stateId) {
          skippedAny = true;
        } else {
          builder.add(target, probability);
        }
      }
      // scale the distribution if any values in the original support were skipped
      Distribution distribution = skippedAny ? builder.scaled() : builder.build();
      assert distribution.isEmpty() || Util.isOne(distribution.sum()) : distribution;
      stateChoices.add(Action.of(distribution, choice.label()));

      Int2IntMap actionCounts = new Int2IntOpenHashMap();
      stateActionCounts.add(actionCounts);

      DistributionBuilder emptyBuilder = Distributions.defaultBuilder();
      model.addChoice(stateId, Action.of(emptyBuilder.build(), choice.label()));
    }

    stateTransitionCounts.put(stateId, stateActionCounts);
    stateActions.put(stateId, stateChoices);
    stateActionChange.put(stateId, new Int2BooleanOpenHashMap());

    exploredActionsCount += stateChoices.size();

    return state;
  }

  public int sampleNextAction(int stateId){
    int[] choiceIndices = new int[model.getNumChoices(stateId)];
    for (int i = 0; i < model.getNumChoices(stateId); i++) {
      choiceIndices[i] = i;
    }
    return Sample.sampleUniform(choiceIndices, model.getNumChoices(stateId));
  }

  public int sampleState(int stateId, int actionIndex){
    Action action = stateActions.get(stateId).get(actionIndex);
    return action.distribution().sample();
  }

  public void activateActionCountFilter(){

    actionCountFilterActive = true;

    for (int i: exploredStates) {
      unfilteredActionsCache.put(i, model.getActions(i));
      model.setActions(i, getActions(i));
    }
  }

  public void deactivateActionCountFilter(){
    actionCountFilterActive = false;

    for (int i: exploredStates){
      model.setActions(i, unfilteredActionsCache.get(i));
    }

    unfilteredActionsCache.clear();
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
    List<Distribution> choices = model.getChoices(stateId);
    if (actionCountFilterActive) {
      List<Distribution> finalChoices = new ArrayList<>();

      for (int i = 0; i < choices.size(); i++) {
        if(getActionCounts(stateId, i) > actionCountFilter){
          finalChoices.add(choices.get(i));
        }
      }

      choices = finalChoices;
    }

    return choices;
  }

  @Override
  public List<Action> getActions(int stateId) {
    assert isExploredState(stateId);
    List<Action> actions = model.getActions(stateId);
    if (actionCountFilterActive) {
      List<Action> finalActions = new ArrayList<>();

      for (int i = 0; i < actions.size(); i++) {
        if(getActionCounts(stateId, i) > actionCountFilter){
          finalActions.add(actions.get(i));
        }
      }

      actions = finalActions;
    }

    return actions;
  }

  @Override
  public S getState(int stateId) {
    return stateMap.getState(stateId);
  }

  @Override
  public String toString() {
    return String.format("DefaultExplorer(%s, %s, %s)", model.getClass(), generator,
            removeSelfLoops ? "inline" : "normal");
  }

}
