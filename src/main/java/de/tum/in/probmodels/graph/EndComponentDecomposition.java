package de.tum.in.probmodels.graph;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.naturals.set.NatBitSets;
import de.tum.in.probmodels.model.Distribution;
import de.tum.in.probmodels.model.Model;
import it.unimi.dsi.fastutil.ints.Int2ObjectFunction;
import it.unimi.dsi.fastutil.ints.IntIterator;
import it.unimi.dsi.fastutil.ints.IntIterators;
import it.unimi.dsi.fastutil.ints.IntSet;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.stream.Collectors;

public final class EndComponentDecomposition {
  private EndComponentDecomposition() {
    // Empty
  }

  public static List<Mec> computeComponents(Model model, NatBitSet restriction) {
    Deque<Mec> workList = new ArrayDeque<>();
    workList.add(Mec.create(model, restriction));

    List<Mec> mecs = new ArrayList<>();
    while (!workList.isEmpty()) {
      Mec mec = workList.remove();
      assert restriction.containsAll(mec.states);

      Int2ObjectFunction<IntIterator> successorFunction = state -> {
        assert mec.states.contains(state) && restriction.contains(state);

        List<Distribution> choices = model.getChoices(state);
        if (choices == null) {
          return IntIterators.EMPTY_ITERATOR;
        }
        IntSet allowedActions = mec.actions.get(state);
        if (allowedActions == null || allowedActions.isEmpty()) {
          return IntIterators.EMPTY_ITERATOR;
        }
        if (allowedActions.size() == 1) {
          return choices.get(allowedActions.iterator().nextInt()).support().iterator();
        }
        // Make successors unique - cheap due to bulk OR
        NatBitSet union = NatBitSets.set();
        allowedActions.forEach((int index) -> union.or(choices.get(index).support()));
        return union.iterator();
      };

      List<Mec> preMecs =
          SccDecomposition.computeSccs(successorFunction, mec.states, s -> true, true)
              .stream()
              .map(scc -> Mec.create(model, scc))
              .filter(m -> !m.states.isEmpty())
              .collect(Collectors.toList());

      if (preMecs.size() == 1) {
        Mec refinedMec = preMecs.get(0);
        assert !refinedMec.states.isEmpty();
        if (mec.equals(refinedMec)) {
          assert !mecs.contains(refinedMec);
          mecs.add(refinedMec);
          continue;
        }
      }
      workList.addAll(preMecs);
    }

    return mecs;
  }
}
