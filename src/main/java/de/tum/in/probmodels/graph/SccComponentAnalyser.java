package de.tum.in.probmodels.graph;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.probmodels.model.Model;
import it.unimi.dsi.fastutil.ints.IntSet;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class SccComponentAnalyser implements ComponentAnalyser {
  private static final Logger logger = Logger.getLogger(SccComponentAnalyser.class.getName());

  @Override
  public List<NatBitSet> findComponents(Model model, IntSet states) {
    logger.log(Level.FINE, "\nStarting BSCC search");

    List<NatBitSet> sccs = SccDecomposition.computeSccs(model::getSuccessors,
        model.getInitialStates(), states::contains, false);
    List<NatBitSet> bsccs = new ArrayList<>(sccs);
    bsccs.removeIf(scc -> scc.intStream().anyMatch(state ->
        model.someSuccessorsMatch(state, successor -> !scc.contains(successor))));

    return bsccs;
  }
}
