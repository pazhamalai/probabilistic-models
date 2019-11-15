package de.tum.in.probmodels.model;

import de.tum.in.probmodels.generator.Choice;
import de.tum.in.probmodels.generator.Generator;
import java.util.ArrayDeque;
import java.util.Queue;
import parser.State;

public final class ModelBuilder {
  private ModelBuilder() {
  }

  public static StateToIndex<State> build(Model model, Generator<State> gen) {
    Queue<State> queue = new ArrayDeque<>(gen.initialStates());
    StateToIndex<State> stateIndices = new StateToIndex<>();

    for (State initialState : gen.initialStates()) {
      int id = model.addState();
      stateIndices.addState(initialState, id);
      model.addInitialState(id);
    }

    while (!queue.isEmpty()) {
      State state = queue.poll();
      assert stateIndices.contains(state);
      int stateId = stateIndices.getStateId(state);

      for (Choice<State> choice : gen.choices(state)) {
        Distribution distribution = new Distribution();
        for (var entry : choice.transitions().object2DoubleEntrySet()) {
          State successor = entry.getKey();
          if (!stateIndices.contains(successor)) {
            queue.add(successor);
            stateIndices.addState(successor, model.addState());
          }
          int successorId = stateIndices.getStateId(successor);
          distribution.add(successorId, entry.getDoubleValue());
        }
        model.addChoice(stateId, distribution);
      }
    }
    return stateIndices;
  }
}
