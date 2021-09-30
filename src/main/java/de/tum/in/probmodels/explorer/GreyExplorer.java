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

/**
 * Class to facilitate grey box exploration. It keeps hold of counts for how many times each state-action-triplet is
 * sampled and accordingly maintains learned distributions.
 *
 * @param <S>
 * @param <M>
 */
public class GreyExplorer<S, M extends Model> implements Explorer<S, M> {
    // A mapping to and from state numbers in partial model to state object in generator.
    private final StateToIndex<S> stateMap = new StateToIndex<>();
    // All states which are in the partial model and explored
    private final IntSet exploredStates = new IntOpenHashSet();
    private final M model;
    private final Generator<S> generator;
    private final boolean removeSelfLoops;

    // When a (state, action) pair is fully explored, this boolean is set to true
    private boolean isNewFullyExploredActionAvailable = false;

    // This holds the counts for haw many times every state-action-successor triplet has been sampled. They can be accessed
    // by first using the stateIndex and then the actionIndex as keys.
    private final Int2ObjectMap<ObjectArrayList<Int2IntMap>> stateTransitionCounts = new Int2ObjectOpenHashMap<>();
    // This holds the real set of actions for the model. Successors are sampled using these distributions.
    private final Int2ObjectMap<ObjectArrayList<Action>> stateActions = new Int2ObjectOpenHashMap<>();

    private int exploredActionsCount = 0;

    private int numTrans = 0;

    private final Int2ObjectMap<List<Action>> unfilteredActionsCache = new Int2ObjectOpenHashMap<>();

    private final Int2ObjectMap<Int2IntMap> unfilteredActionIndexMap = new Int2ObjectOpenHashMap<>();

    private boolean isFullyExploredActionFlag = false;

    public void activateFullyExploredActionsModel() {
        for (int i : exploredStates) {
            unfilteredActionsCache.put(i, getActions(i));
            model.setActions(i, filterActions(i));
        }

        isFullyExploredActionFlag = true;
    }

    public void deactivateFullyExploredActionsModel() {
        isFullyExploredActionFlag = false;

        for (int i: exploredStates){
            model.setActions(i, unfilteredActionsCache.get(i));
        }

        unfilteredActionsCache.clear();
        unfilteredActionIndexMap.clear();
    }

    private List<Action> filterActions(int state) {
        List<Action> actions = getActions(state);
        List<Action> filteredActions = new ArrayList<>();
        unfilteredActionIndexMap.put(state, new Int2IntOpenHashMap());

        for (int i = 0; i < actions.size(); i++) {
            if (actions.get(i).distribution().size() == 0) {
                continue;
            }

            if (isStateActionExplored(state, i)) {
                unfilteredActionIndexMap.get(state).put(filteredActions.size(), i);
                filteredActions.add(actions.get(i));
            }
        }

        return filteredActions;
    }

    // Creates and returns a default explorer object from a generator. Explores all initial states
    public static <S, M extends Model> GreyExplorer<S, M> of(M model, Generator<S> generator,
                                                             boolean removeSelfLoops) {
        GreyExplorer<S, M> explorer = new GreyExplorer<>(model, generator, removeSelfLoops);
        IntList initialStateIds = new IntArrayList();
        for (S initialState : generator.initialStates()) {
            int stateId = explorer.getStateId(initialState);
            explorer.exploreState(stateId);
            initialStateIds.add(stateId);
        }
        model.setInitialStates(initialStateIds);
        return explorer;
    }

    GreyExplorer(M model, Generator<S> generator, boolean removeSelfLoops) {
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

    public int getNumExploredActions() {
        return exploredActionsCount;
    }

    public int getNumTrans() {
        return numTrans;
    }

    /**
     * Update sampled counts for a state-action-successor triplet. If update is true, the learned distributions are
     * immediately updated. Returns whether a new action has been sampled more than actionCountFilter number of times.
     */

    public void updateCounts(int state, int actionIndex, int successor) {

        // TODO this might not be needed. Check.
        if (isFullyExploredActionFlag) {
            actionIndex = unfilteredActionIndexMap.get(state).get(actionIndex);
        }

        int newTransitionCount = incrementTransitionCount(state, actionIndex, successor);

//        Logger.getLogger("GreyExplorer").log(Level.INFO, state + " " + actionIndex + " " + successor + " " + newTransitionCount);

        updateStateActionDistributionInModel(state, actionIndex);

        // Checks this particular transition (state, actionIndex, successor) is being visited for the first time
        if (newTransitionCount == 1) {
            numTrans++;

            // We update whether this action is fully explored.
            isNewFullyExploredActionAvailable = isNewFullyExploredActionAvailable || isStateActionExplored(state, actionIndex);
        }
    }

    /**
     * @return Returns the number of times a state action pair has been sampled.
     */
    public int getActionCounts(int stateId, int actionIndex) {
        if (isFullyExploredActionFlag) {
            actionIndex = unfilteredActionIndexMap.get(stateId).get(actionIndex);
        }
        Int2IntMap transitionCounts = stateTransitionCounts.get(stateId).get(actionIndex);
        return transitionCounts.values().stream().mapToInt(s -> s).sum();
    }

    /**
     * @return Returns the distribution of an action for a state from the transitionCounts.
     */
    private Distribution getDistributionFromCounts(int stateId, Int2IntMap transitionCounts) {

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

        exploredActionsCount += stateChoices.size();

        return state;
    }

    /**
     * Uniformly samples an action from the model choices.
     *
     * @return Returns the index of the sampled action corresponding to the choices in the model variable.
     */
    public int sampleNextAction(int stateId) {
        int[] choiceIndices = new int[model.getNumChoices(stateId)];
        for (int i = 0; i < model.getNumChoices(stateId); i++) {
            choiceIndices[i] = i;
        }
        return Sample.sampleUniform(choiceIndices, model.getNumChoices(stateId));
    }

    /**
     * Samples a successor state from the distribution of the action.
     *
     * @param actionIndex: index of the action corresponding to the model variable.
     * @return StateID of the sampled successor state.
     */
    public int simulateAction(int stateId, int actionIndex) {
        if (isFullyExploredActionFlag) {
            actionIndex = unfilteredActionIndexMap.get(stateId).get(actionIndex);
        }
        Action action = stateActions.get(stateId).get(actionIndex);
        return action.distribution().sample();
    }

    // TODO this method itself might not be needed. Because greybox doesn't require actions to be visited
    // TODO requiredSample number of times
    public void simulateActionRepeatedly(int stateId, int filteredIndex, double requiredSamples) {
        int realIndex = filteredIndex;
        if (isFullyExploredActionFlag) {
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
     *
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
     *
     * @return a list of actions.
     */
    @Override
    public List<Action> getActions(int stateId) {
        assert isExploredState(stateId);

        return model.getActions(stateId);
    }

    @Override
    public S getState(int stateId) {
        return stateMap.getState(stateId);
    }

    @Override
    public String toString() {
        return String.format("GreyExplorer(%s, %s, %s)", model.getClass(), generator,
                removeSelfLoops ? "inline" : "normal");
    }


    /**
     * @return true if all the successors of the (stateId, exploredAction) has been visited.
     */
    public boolean isStateActionExplored(int stateId, int actionIndex) {
        int actualSuccessors = getActualSuccessorsOfStateAction(stateId, actionIndex);

        Action exploredAction = getActions(stateId).get(actionIndex);
        int exploredSuccessors = exploredAction.distribution().size();

        return actualSuccessors == exploredSuccessors;
    }


    /**
     * Given a state and an action, this method returns the actual number of successors belonging to the action.
     *
     * @param actionIndex Maybe a partially explored action. Only partial successors might have been explored
     * @return The actual successors of this action, as per the original MDP.
     */
    public int getActualSuccessorsOfStateAction(int stateId, int actionIndex) {

        // We find the same action in stateActions variable.
        Action actualAction = stateActions.get(stateId).get(actionIndex);

        assert actualAction != null;
        assert actualAction.distribution() != null;

        return actualAction.distribution().size();
    }


    public boolean isNewFullyExploredActionAvailable() {
        return isNewFullyExploredActionAvailable;
    }

    public void resetFullyExploredActionFlag() {
        isNewFullyExploredActionAvailable = false;
    }

    private int incrementTransitionCount(int state, int actionIndex, int successor) {
        Int2IntMap transitionCounts = stateTransitionCounts.get(state).get(actionIndex);

        int newTransitionCount = transitionCounts.getOrDefault(successor, 0) + 1;
        transitionCounts.put(successor, newTransitionCount);
        return newTransitionCount;
    }

    private void updateStateActionDistributionInModel(int state, int actionIndex) {
        Int2IntMap transitionCounts = stateTransitionCounts.get(state).get(actionIndex);

        List<Action> currActions = model.getActions(state);
        Action currAction = currActions.get(actionIndex);

        Distribution distribution = getDistributionFromCounts(state, transitionCounts);
        currActions.set(actionIndex, Action.of(distribution, currAction.label()));

        model.setActions(state, currActions);
    }
}
