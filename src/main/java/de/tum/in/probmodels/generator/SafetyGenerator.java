package de.tum.in.probmodels.generator;

import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.function.Predicate;

public class SafetyGenerator<S> implements Generator<ProductState<S, Boolean>> {
  private static final ProductState<?, Boolean> ERROR = ProductState.of(null, false);

  private final Generator<S> system;
  private final Predicate<S> safety;

  public SafetyGenerator(Generator<S> system, Predicate<S> safety) {
    this.system = system;
    this.safety = safety;
  }

  @Override
  public Collection<ProductState<S, Boolean>> initialStates() {
    Collection<S> systemInitialStates = system.initialStates();
    Collection<ProductState<S, Boolean>> productInitialStates =
        new ArrayList<>(systemInitialStates.size());
    for (S systemInitialState : systemInitialStates) {
      productInitialStates.add(productState(systemInitialState));
    }
    return productInitialStates;
  }

  private ProductState<S, Boolean> productState(S state) {
    return safety.test(state) ? ProductState.of(state, true) : error();
  }

  @SuppressWarnings("unchecked")
  private ProductState<S, Boolean> error() {
    return (ProductState<S, Boolean>) ERROR;
  }

  @Override
  public Collection<Choice<ProductState<S, Boolean>>> choices(ProductState<S, Boolean> state) {
    if (state.equals(ERROR)) {
      return Collections.emptyList();
    }
    assert state.automaton() : state;

    S systemState = state.system();
    Collection<Choice<S>> systemChoices = system.choices(systemState);
    Collection<Choice<ProductState<S, Boolean>>> productChoices =
        new ArrayList<>(systemChoices.size());

    for (Choice<S> systemChoice : systemChoices) {
      Object2DoubleMap<S> systemTransitions = systemChoice.transitions();
      // TODO Efficient / Lazy
      Object2DoubleMap<ProductState<S, Boolean>> productTransitions =
          new Object2DoubleOpenHashMap<>();
      for (Object2DoubleMap.Entry<S> transition : systemTransitions.object2DoubleEntrySet()) {
        S systemSuccessor = transition.getKey();
        ProductState<S, Boolean> successor = productState(systemSuccessor);
        // Need to merge, since multiple transitions may lead to the error state
        productTransitions.mergeDouble(successor, transition.getDoubleValue(), Double::sum);
      }
      productChoices.add(Choice.of(systemChoice.label(), productTransitions));
    }

    return productChoices;
  }
}
