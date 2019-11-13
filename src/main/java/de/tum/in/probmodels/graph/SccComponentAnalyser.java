package de.tum.in.probmodels.graph;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.probmodels.model.CollapseModel;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class SccComponentAnalyser implements ComponentAnalyser {
  private static final Logger logger = Logger.getLogger(SccComponentAnalyser.class.getName());

  @Override
  public List<NatBitSet> findComponents(CollapseModel<?> model, NatBitSet states) {
    logger.log(Level.FINE, "\nStarting BSCC search");

    List<NatBitSet> bsccs = new ArrayList<>(SccDecomposition.computeSccs(model::getSuccessors,
        model.getInitialStates(), false));
    bsccs.removeIf(scc -> scc.intStream().anyMatch(state ->
        model.someSuccessorsMatch(state, successor -> !scc.contains(successor))));

    return bsccs;
  }
}
