package de.tum.in.probmodels.graph;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.probmodels.model.Model;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class MecComponentAnalyser implements ComponentAnalyser {
  private static final Logger logger = Logger.getLogger(MecComponentAnalyser.class.getName());

  @Override
  public List<NatBitSet> findComponents(Model model, NatBitSet states) {
    logger.log(Level.FINE, "\nStarting MECs search");

    List<Mec> mecs = EndComponentDecomposition.computeComponents(model, states);
    if (mecs.isEmpty()) {
      logger.log(Level.FINE, "Found no MECs");
      return Collections.emptyList();
    }
    logger.log(Level.FINE, "Found MECs {0}", mecs);

    List<NatBitSet> mecStates = new ArrayList<>(mecs.size());
    mecs.forEach(m -> mecStates.add(m.states));

    return mecStates;
  }
}
