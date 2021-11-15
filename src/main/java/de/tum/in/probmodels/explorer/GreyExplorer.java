package de.tum.in.probmodels.explorer;

import de.tum.in.probmodels.generator.Generator;
import de.tum.in.probmodels.model.*;
import it.unimi.dsi.fastutil.ints.*;

import java.util.ArrayList;
import java.util.List;


public class GreyExplorer<S, M extends Model> extends BlackExplorer<S, M> {
    // When a (state, action) pair is fully explored, this boolean is set to true
    private boolean isNewFullyExploredActionAvailable = false;

    GreyExplorer(M model, Generator<S> generator, boolean removeSelfLoops) {
        super(model, generator, removeSelfLoops);
    }

    /**
     * We only keep actions for which all the successors has been explored.
     */
    @Override
    public List<Action> filterActions(int stateId) {
        List<Action> actions = getActions(stateId);
        List<Action> filteredActions = new ArrayList<>();
        unfilteredActionIndexMap.put(stateId, new Int2IntOpenHashMap());

        for (int i = 0; i < actions.size(); i++) {
            if (actions.get(i).distribution().size() == 0) {
                continue;
            }

            if (isStateActionExplored(stateId, i)) {
                unfilteredActionIndexMap.get(stateId).put(filteredActions.size(), i);
                filteredActions.add(actions.get(i));
            }
        }

        return filteredActions;
    }

    @Override
    public boolean updateCounts(int state, int actionIndex, int successor) {
        int oldNumTrans = numTrans;
        boolean result =  super.updateCounts(state, actionIndex, successor);

        // This means a new transition has been visited
        if (oldNumTrans < numTrans) {
            isNewFullyExploredActionAvailable = isNewFullyExploredActionAvailable || isStateActionExplored(state, actionIndex);
        }
        return result;
    }

    public boolean isNewFullyExploredActionAvailable() {
        return isNewFullyExploredActionAvailable;
    }

    public void resetFullyExploredActionFlag() {
        isNewFullyExploredActionAvailable = false;
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

    public int getOriginalNumSuccessors(int state, int action) {
        return stateActions.get(state).get(action).distribution().support().size();
    }
}
