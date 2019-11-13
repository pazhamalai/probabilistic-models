package de.tum.in.probmodels.graph;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.probmodels.model.CollapseModel;
import java.util.List;

@FunctionalInterface
public interface ComponentAnalyser {
  List<NatBitSet> findComponents(CollapseModel<?> model, NatBitSet states);
}
