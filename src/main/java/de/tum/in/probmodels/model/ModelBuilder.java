package de.tum.in.probmodels.model;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import de.tum.in.naturals.set.BoundedNatBitSet;
import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.naturals.set.NatBitSets;
import de.tum.in.naturals.unionfind.IntArrayUnionFind;
import de.tum.in.naturals.unionfind.IntUnionFind;
import de.tum.in.probmodels.generator.Choice;
import de.tum.in.probmodels.generator.Generator;
import de.tum.in.probmodels.graph.Mec;
import it.unimi.dsi.fastutil.ints.IntSet;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.List;
import java.util.ListIterator;
import java.util.Queue;
import java.util.function.IntConsumer;
import java.util.function.IntFunction;
import java.util.function.IntUnaryOperator;
import java.util.function.Supplier;
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
        DistributionBuilder builder = Distributions.defaultBuilder();
        for (var entry : choice.transitions().object2DoubleEntrySet()) {
          State successor = entry.getKey();
          if (!stateIndices.contains(successor)) {
            queue.add(successor);
            stateIndices.addState(successor, model.addState());
          }
          int successorId = stateIndices.getStateId(successor);
          builder.add(successorId, entry.getDoubleValue());
        }
        model.addChoice(stateId, builder.build());
      }
    }
    return stateIndices;
  }

  public static <T extends Model> QuotientModel<T> buildQuotient(T model,
      Supplier<T> quotientModelConstructor, List<NatBitSet> equivalence) {
    T quotientModel = quotientModelConstructor.get();
    checkArgument(quotientModel.getNumStates() == 0);
    int numStates = model.getNumStates();
    IntUnionFind collapseUF = new IntArrayUnionFind(numStates);

    NatBitSet[] stateClass = new NatBitSet[numStates];
    equivalence.forEach(states -> {
      int representative = states.firstInt();
      states.forEach((int state) -> {
        collapseUF.union(representative, state);
        stateClass[state] = states;
      });
    });

    int[] stateToQuotientArray = new int[numStates];
    Arrays.fill(stateToQuotientArray, -1);
    for (int state = 0; state < numStates; state++) {
      int representative = collapseUF.find(state);
      int quotientState = stateToQuotientArray[representative];
      if (quotientState > 0) {
        stateToQuotientArray[state] = quotientState;
      } else {
        int newQuotientState = quotientModel.addState();
        stateToQuotientArray[representative] = newQuotientState;
        stateToQuotientArray[state] = newQuotientState;
      }
    }
    assert quotientModel.getNumStates() == collapseUF.componentCount();

    Multimap<Integer, Object> selfLoops = HashMultimap.create();
    IntUnaryOperator stateToQuotientState = i -> stateToQuotientArray[i];
    for (int state = 0; state < numStates; state++) {
      int quotientState = stateToQuotientArray[state];
      for (Action action : model.getActions(state)) {
        Distribution quotientDistribution = action.distribution().map(stateToQuotientState).build();
        if (quotientDistribution.size() == 1 && quotientDistribution.contains(quotientState)) {
          if (selfLoops.put(quotientState, action.label())) {
            quotientModel.addChoice(quotientState, Action.of(quotientDistribution, action.label()));
          }
        } else {
          quotientModel.addChoice(quotientState, Action.of(quotientDistribution, action.label()));
        }
      }
    }
    model.getInitialStates().forEach((int initialState) ->
        quotientModel.addInitialState(stateToQuotientArray[initialState]));

    NatBitSet[] quotientToClassArray = new NatBitSet[quotientModel.getNumStates()];
    for (int state = 0; state < numStates; state++) { // NOPMD
      int quotientState = stateToQuotientArray[state];
      assert quotientToClassArray[quotientState] == null
          || quotientToClassArray[quotientState] == stateClass[state];
      NatBitSet clazz = stateClass[state];
      if (clazz == null) {
        quotientToClassArray[quotientState] = NatBitSets.singleton(state);
      } else {
        quotientToClassArray[quotientState] = clazz;
      }
      assert quotientToClassArray[quotientState].contains(state);
    }

    IntFunction<NatBitSet> quotientToClass = i -> quotientToClassArray[i];
    return QuotientModelTuple.create(quotientModel, quotientToClass, stateToQuotientState);
  }

  public static <T extends Model> RestrictedModel<T> buildRestrictedModel(T model,
      Supplier<T> restrictedModelConstructor, IntSet states) {
    T newModel = restrictedModelConstructor.get();
    checkArgument(newModel.getNumStates() == 0);
    int[] originalToRestrictedStates = new int[model.getNumStates()];
    Arrays.fill(originalToRestrictedStates, -1);
    states.forEach((int allowedState) ->
        originalToRestrictedStates[allowedState] = newModel.addState());

    int restrictedStates = newModel.getNumStates();
    assert restrictedStates == states.size();
    int[] restrictedToOriginalStates = new int[restrictedStates];
    NatBitSet[] restrictedActions = new NatBitSet[restrictedStates];

    for (int originalState = 0; originalState < model.getNumStates(); originalState++) {
      if (originalToRestrictedStates[originalState] == -1) {
        continue;
      }
      int restrictedState = originalToRestrictedStates[originalState];
      restrictedToOriginalStates[restrictedState] = originalState;

      List<Action> actions = model.getActions(originalState);
      BoundedNatBitSet removedActions = NatBitSets.boundedSet(actions.size(), 0);
      ListIterator<Action> iterator = actions.listIterator();
      while (iterator.hasNext()) {
        int index = iterator.nextIndex();
        Action action = iterator.next();

        DistributionBuilder builder = Distributions.defaultBuilder();
        action.distribution().forEach((target, probability) -> {
          int restrictedDestination = originalToRestrictedStates[target];
          if (restrictedDestination >= 0) {
            builder.add(restrictedDestination, probability);
          }
        });
        if (builder.isEmpty()) {
          removedActions.set(index);
        } else {
          model.addChoice(restrictedState, Action.of(builder.scaled(), action.label()));
        }
      }
      restrictedActions[restrictedState] = NatBitSets.compact(removedActions.complement());
      /* if (addedActions == 0) {
        newModel.addDeadlockState(restrictedState);
      } */
    }
    model.getInitialStates().forEach((int initialState) -> {
      int restrictedInitialState = originalToRestrictedStates[initialState];
      if (restrictedInitialState != -1) {
        newModel.addInitialState(restrictedInitialState);
      }
    });

    IntUnaryOperator stateMapping = i -> restrictedToOriginalStates[i];
    IntFunction<NatBitSet> stateActions = i -> restrictedActions[i];
    return RestrictedModelTuple.create(newModel, stateMapping, stateActions);
  }

  public static <T extends Model> RestrictedModel<T> buildMecRestrictedModel(T model,
      Supplier<T> restrictedModelConstructor, Mec mec) {
    T newModel = restrictedModelConstructor.get();
    checkArgument(newModel.getNumStates() == 0);
    int[] originalToRestrictedStates = new int[model.getNumStates()];
    Arrays.fill(originalToRestrictedStates, -1);
    mec.states.forEach((int allowedState) ->
            originalToRestrictedStates[allowedState] = newModel.addState());

    int restrictedStates = newModel.getNumStates();
    assert restrictedStates == mec.states.size();
    int[] restrictedToOriginalStates = new int[restrictedStates];
    NatBitSet[] restrictedActions = new NatBitSet[restrictedStates];

    for (int originalState = 0; originalState < model.getNumStates(); originalState++) {
      if (originalToRestrictedStates[originalState] == -1) {
        continue;
      }
      int restrictedState = originalToRestrictedStates[originalState];
      restrictedToOriginalStates[restrictedState] = originalState;

      IntSet mecActions = mec.actions.get(originalState);
      NatBitSet actions = NatBitSets.setWithExpectedLength(mecActions.size());
      mecActions.forEach((IntConsumer) actions::set);
      restrictedActions[restrictedState] = NatBitSets.compact(actions);
      /* if (addedActions == 0) {
        newModel.addDeadlockState(restrictedState);
      } */
    }
    newModel.addInitialState(mec.states.firstInt());

    IntUnaryOperator stateMapping = i -> restrictedToOriginalStates[i];
    IntFunction<NatBitSet> stateActions = i -> restrictedActions[i];
    return RestrictedModelTuple.create(newModel, stateMapping, stateActions);
  }
}
