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

/**
 * Class to facilitate black box exploration. It keeps hold of counts for how many times each state-action-triplet is
 * sampled and accordingly maintains learned distributions.
 * @param <S>
 * @param <M>
 */
public class BlackExplorer<S, M extends Model> implements Explorer<S, M>{
  // A mapping to and from state numbers in partial model to state object in generator.
  protected final StateToIndex<S> stateMap = new StateToIndex<>();
  // All states which are in the partial model and explored
  protected final IntSet exploredStates = new IntOpenHashSet();
  protected final M model;
  protected final Generator<S> generator;
  protected final boolean removeSelfLoops;

  // This holds the counts for haw many times every state-action-successor triplet has been sampled. They can be accessed
  // by first using the stateIndex and then the actionIndex as keys.
  protected final Int2ObjectMap<ObjectArrayList<Int2IntMap>> stateTransitionCounts = new Int2ObjectOpenHashMap<>();
  // This holds the real set of actions for the model. Successors are sampled using these distributions.
  protected final Int2ObjectMap<ObjectArrayList<Action>> stateActions = new Int2ObjectOpenHashMap<>();
  // This holds whether the counts of a state-action pair have been changed or not.
  protected final Int2ObjectMap<Int2BooleanMap> stateActionChange = new Int2ObjectOpenHashMap<>();

  // This holds all the actions of the model regardless of whether they pass the actionCountFilter. They are used to
  // restore the original set of actions after deactivateActionCountFilter() is called.
  private final Int2ObjectMap<List<Action>> unfilteredActionsCache = new Int2ObjectOpenHashMap<>();

  // This holds a mapping from the indices of actions in the filtered model to the indices in the filtered model.
  // This helps during to update counts when the model is filtered.
  protected final Int2ObjectMap<Int2IntMap> unfilteredActionIndexMap = new Int2ObjectOpenHashMap<>();

  protected int exploredActionsCount = 0;
  protected boolean actionCountFilterActive = false;
  protected double actionCountFilter;

  protected int numTrans = 0;

  public Int2ObjectMap<ObjectArrayList<Int2IntMap>> getStateTransitionCounts() {
    return stateTransitionCounts;
  }

  // It returns the original state actions as per the MDP.
  public Int2ObjectMap<ObjectArrayList<Action>> getOriginalStateActions() {
    return stateActions;
  }

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

  /**
   * Updates to actionCountFilter according to mecConfidence and pMin.
   * @param mecConfidence
   * @param pMin
   */
  public void updateCountParams(double mecConfidence, double pMin){
    this.actionCountFilter = Math.log(mecConfidence)/Math.log(1-pMin);
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

    stateActionChange.get(state).put(actionIndex, !update);

    return newTrans;
  }

  /**
   * @param stateId
   * @param actionIndex
   * @return Returns the number of times a state action pair has been sampled.
   */
  public int getActionCounts(int stateId, int actionIndex){
    if (actionCountFilterActive) {
      actionIndex = unfilteredActionIndexMap.get(stateId).get(actionIndex);
    }
    Int2IntMap transitionCounts = stateTransitionCounts.get(stateId).get(actionIndex);
    return transitionCounts.values().stream().mapToInt(s -> s).sum();
  }

  /**
   * @param stateId
   * @param transitionCounts
   * @return Returns the distribution of an action for a state from the transitionCounts.
   */
  protected Distribution getDistributionFromCounts(int stateId, Int2IntMap transitionCounts){

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

  /**
   * Updates the model according to the latest state-action-successor counts only if changed is satisfied for a state-
   * action pair.
   * @param changed
   */
  public void updateModel(BiPredicate<Integer, Integer> changed){

    for (int stateId: exploredStates) {
      List<Action> actionList = new ArrayList<>();
      for (int actionIndex=0; actionIndex<stateTransitionCounts.get(stateId).size(); actionIndex++){
        if (actionCountFilterActive) {
          actionIndex = unfilteredActionIndexMap.get(stateId).get(actionIndex);
        }
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

  /**
   * Updates to model counts of the entire model according to the latest state-action-successor counts.
   */
  public void updateModelCounts(){

    updateModel((stateId, actionIndex) -> stateActionChange.get((int) stateId).get((int) actionIndex));

    for (int stateId: stateActionChange.keySet()){
      for (int actionIndex=0; actionIndex<stateActionChange.size(); actionIndex++) {
        if (actionCountFilterActive) {
          actionIndex = unfilteredActionIndexMap.get(stateId).get(actionIndex);
        }
        stateActionChange.get(stateId).put(actionIndex, false);
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
      // Real distribution added to stateChoices
      stateChoices.add(Action.of(distribution, choice.label()));

      Int2IntMap actionCounts = new Int2IntOpenHashMap();
      stateActionCounts.add(actionCounts);

      // Empty distribution added to model
      DistributionBuilder emptyBuilder = Distributions.defaultBuilder();
      model.addChoice(stateId, Action.of(emptyBuilder.build(), choice.label()));
    }

    stateTransitionCounts.put(stateId, stateActionCounts);
    stateActions.put(stateId, stateChoices);
    stateActionChange.put(stateId, new Int2BooleanOpenHashMap());

    exploredActionsCount += stateChoices.size();

    return state;
  }

  /**
   * Uniformly samples an action from the model choices.
   * @param stateId
   * @return Returns the index of the sampled action corresponding to the choices in the model variable.
   */
  public int sampleNextAction(int stateId){
    int[] choiceIndices = new int[model.getNumChoices(stateId)];
    for (int i = 0; i < model.getNumChoices(stateId); i++) {
      choiceIndices[i] = i;
    }
    return Sample.sampleUniform(choiceIndices, model.getNumChoices(stateId));
  }

  /**
   * Samples a successor state from the distribution of the action.
   * @param stateId
   * @param actionIndex: index of the action corresponding to the model variable.
   * @return StateID of the sampled successor state.
   */
  public int simulateAction(int stateId, int actionIndex){
    if (actionCountFilterActive) {
      actionIndex = unfilteredActionIndexMap.get(stateId).get(actionIndex);
    }
    Action action = stateActions.get(stateId).get(actionIndex);
    return action.distribution().sample();
  }

  public void simulateActionRepeatedly(int stateId, int filteredIndex, double requiredSamples){
    int realIndex = filteredIndex;
    if (actionCountFilterActive) {
      realIndex = unfilteredActionIndexMap.get(stateId).get(filteredIndex);
    }
    Action action = stateActions.get(stateId).get(realIndex);
    int actionCounts = getActionCounts(stateId, realIndex);
    Int2IntMap actionTransitionCounts = new Int2IntOpenHashMap();
    for(int succ: action.distribution().support()) {
      actionTransitionCounts.put(succ, 0);
    }
    while (actionCounts<requiredSamples) {
      int succ = action.distribution().sample();
      actionTransitionCounts.put(succ, actionTransitionCounts.get(succ)+1);
      actionCounts++;
    }
    for(int succ: action.distribution().support()) {
      int currValue = stateTransitionCounts.get(stateId).get(realIndex).get(succ);
      stateTransitionCounts.get(stateId).get(realIndex)
              .put(succ, currValue+actionTransitionCounts.get(succ));
    }
    List<Action> currActions = model.getActions(stateId);
    Distribution distribution = getDistributionFromCounts(stateId, stateTransitionCounts.get(stateId).get(realIndex));
    currActions.set(filteredIndex, Action.of(distribution, action.label()));

    model.setActions(stateId, currActions);
  }

  /**
   * Activates the action count filter. Updates the model variable to hold only those actions that pass the actionCountFilter.
   * Caches to original action for later use.
   */
  public void activateActionCountFilter(){

    for (int i: exploredStates) {
      unfilteredActionsCache.put(i, model.getActions(i));
      model.setActions(i, filterActions(i));
    }
    actionCountFilterActive = true;
  }

  /**
   * Deactivates the action count filter. Updates the model variable to hold the original set of actions.
   */
  public void deactivateActionCountFilter(){
    actionCountFilterActive = false;

    for (int i: exploredStates){
      model.setActions(i, unfilteredActionsCache.get(i));
    }

    unfilteredActionsCache.clear();
    unfilteredActionIndexMap.clear();
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

  /**
   * Returns a list of distributions of the corresponding actions for a state. If actionCountFilter is active, only
   * those actions' distributions are added which have been sampled more than actionCountFilter number of times.
   * @param stateId
   * @return a list of distributions of choices.
   */
  @Override
  public List<Distribution> getChoices(int stateId) {
    assert isExploredState(stateId);

    return model.getChoices(stateId);
  }

  /**
   * Returns a list of actions for a state. If actionCountFilter is active, only those actions are added which have been
   * sampled more than actionCountFilter number of times. unfilteredActionIndexMap is also populated.
   * @param stateId
   * @return a list of actions.
   */
  @Override
  public List<Action> getActions(int stateId) {
    assert isExploredState(stateId);

    return model.getActions(stateId);
  }

  public List<Action> filterActions(int stateId) {
    List<Action> actions = model.getActions(stateId);
    List<Action> finalActions = new ArrayList<>();
    unfilteredActionIndexMap.put(stateId, new Int2IntOpenHashMap());

    for (int i = 0; i < actions.size(); i++) {
      if(getActionCounts(stateId, i) > actionCountFilter){
        unfilteredActionIndexMap.get(stateId).put(finalActions.size(), i);
        finalActions.add(actions.get(i));
      }
    }

    return finalActions;
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
