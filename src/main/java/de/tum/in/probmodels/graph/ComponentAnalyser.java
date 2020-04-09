package de.tum.in.probmodels.graph;

import de.tum.in.naturals.set.NatBitSet;
import de.tum.in.probmodels.model.Model;
import it.unimi.dsi.fastutil.ints.IntSet;
import java.util.List;

@FunctionalInterface
public interface ComponentAnalyser {
  List<NatBitSet> findComponents(Model model, IntSet states);
}
